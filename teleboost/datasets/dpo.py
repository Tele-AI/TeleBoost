# Copyright (c) 2025 TeleAI-infra Team (TeleTron)
# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DPO dataset base class for OSS users.

Subclass this and implement ``__getitem__`` to integrate a *pre-encoded*
data source (lmdb / webdataset / huggingface datasets / latents on disk /
etc.) with teleboost's training loop.

This is the consumer-side contract, after the text/VAE/image encoders have
run.  Producer-backed raw-video datasets intentionally use a different
contract: two top-level preference branches, each containing ``images`` and
prompt fields.  The distributed Wan encoder converts that raw contract into
the encoded contract documented below.  See
``CSVPreferenceDPODataset`` for the built-in raw-video implementation.

Schema contract — each ``__getitem__(idx)`` must return a dict shaped like:

    {
        "context":  Tensor[S_text, D_text],   # T5 / text-encoder output
        "chosen":   {
            "latents":           Tensor[C, T_c, H_c, W_c],   # VAE-encoded video
            "img_clip_feature":  Tensor[N_clip, D_clip],     # CLIP image feature
            "img_emb_y":         Tensor[C, T_c, H_c, W_c],   # reference frame latent
        },
        "rejected": {  # same keys; T_r / H_r / W_r MAY differ from chosen
            "latents":           Tensor[C, T_r, H_r, W_r],
            "img_clip_feature":  Tensor[N_clip, D_clip],
            "img_emb_y":         Tensor[C, T_r, H_r, W_r],
        },
    }

Notes
-----
* All tensors should be CPU and bf16 (or fp32 — teleboost auto-casts).
* Batch dim is ADDED by the DataLoader collator; do NOT prepend B here.
* Chosen and rejected MAY have different temporal/spatial shapes. The
  training loop runs each branch through a separate forward pass
  (`_run_branch` in pretrain_dpo_i2v.py), so shape mismatch is supported
  by design — verified end-to-end with mismatched-shape FakeDataset
  (see tests/).
* If you do not need split-DPO's per-branch backward (you want a single
  preference loss instead), still return both branches and the framework
  handles the rest.

Minimal working subclass:

    class MyDPODataset(DPODatasetBase):
        def __init__(self, manifest_csv, vae, text_encoder, clip):
            self.rows = pd.read_csv(manifest_csv).to_dict("records")
            self.vae, self.text_encoder, self.clip = vae, text_encoder, clip

        def __len__(self): return len(self.rows)

        def __getitem__(self, idx):
            from teleboost.datasets.video_io import load_video

            row = self.rows[idx]
            chosen_video  = load_video(row["chosen_path"])
            reject_video  = load_video(row["rejected_path"])
            with torch.no_grad():
                ctx        = self.text_encoder(row["prompt"])
                ch_lat     = self.vae.encode(chosen_video)
                rj_lat     = self.vae.encode(reject_video)
                ch_clip    = self.clip(chosen_video[0])
                rj_clip    = self.clip(reject_video[0])
            return {
                "context": ctx,
                "chosen":   {"latents": ch_lat, "img_clip_feature": ch_clip,
                             "img_emb_y": ch_lat[:, :1].clone()},
                "rejected": {"latents": rj_lat, "img_clip_feature": rj_clip,
                             "img_emb_y": rj_lat[:, :1].clone()},
            }

Then register it:

    from teleboost.datasets.build import DATASETS
    DATASETS.register_module(MyDPODataset)

And select it via your config-path:

    config = dict(dataset=dict(type="MyDPODataset", manifest_csv="...", ...))
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class DPODatasetBase(torch.utils.data.Dataset):
    """Abstract base class — subclass and implement ``__getitem__``.

    See module docstring for the schema each ``__getitem__`` must return.
    """

    # Subclass interface ────────────────────────────────────────────────
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Mapping:
        raise NotImplementedError

    # Optional helper subclasses can call to validate output schema in tests.
    @staticmethod
    def _validate_item(item: Mapping, allow_mismatched_shapes: bool = True) -> None:
        """Lightweight schema check. Call from a unit test, not the training
        loop — this is O(item) every call and not free.

        Set ``allow_mismatched_shapes=False`` if you want to enforce that
        chosen and rejected have identical shapes (uncommon — most DPO
        setups allow per-branch lengths).
        """
        required_top = {"context", "chosen", "rejected"}
        missing = required_top - set(item.keys())
        if missing:
            raise ValueError(f"DPO item missing top-level keys: {missing}")
        for branch in ("chosen", "rejected"):
            sub = item[branch]
            if not isinstance(sub, Mapping):
                raise TypeError(f"item['{branch}'] must be a Mapping; got {type(sub)}")
            req = {"latents", "img_clip_feature", "img_emb_y"}
            miss = req - set(sub.keys())
            if miss:
                raise ValueError(f"item['{branch}'] missing keys: {miss}")
        if not allow_mismatched_shapes:
            ch_lat = item["chosen"]["latents"]
            rj_lat = item["rejected"]["latents"]
            if ch_lat.shape != rj_lat.shape:
                raise ValueError(f"chosen.latents shape {ch_lat.shape} != rejected.latents shape {rj_lat.shape}; either pad to a common shape in your dataset or set allow_mismatched_shapes=True.")


# CSV-backed raw-video preference pairs for producer-side DPO encoding. This
# implementation intentionally does not run VAE, CLIP, or text encoders.


@dataclass(frozen=True)
class _PreferenceRow:
    prompt: str
    chosen_path: Path
    rejected_path: Path
    manifest: Path
    line_number: int


class CSVPreferenceDPODataset(DPODatasetBase):
    """Load preference-pair videos from one or more CSV manifests.

    ``chosen_video_key`` and ``rejected_video_key`` name the two output
    branches expected by the encoder/training loop.  By default they also
    name the corresponding CSV columns.  Set ``chosen_path_key`` and
    ``rejected_path_key`` when a manifest uses different column names while
    retaining canonical output branch names.

    Relative video paths resolve against ``dataset_base_path`` when it is
    set, otherwise against the directory containing their manifest shard.
    Manifests, headers, cells, and (by default) every referenced video path
    are validated during construction so a bad shard fails before training.
    """

    output_stage = "raw_video"

    def __init__(
        self,
        data_path_list: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
        dataset_metadata_path: str | os.PathLike | None = None,
        dataset_base_path: str | os.PathLike = "",
        dataset_repeat: int = 1,
        chosen_video_key: str = "chosen",
        rejected_video_key: str = "rejected",
        chosen_path_key: str | None = None,
        rejected_path_key: str | None = None,
        prompt_key: str = "prompt",
        transforms: Sequence[dict | Callable] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_interval: int = 1,
        resize_mode: str = "center_crop",
        check_video_paths: bool = True,
        video_loader: Callable[[str], Sequence[Any]] | None = None,
        time_division_factor: int | None = None,
        time_division_remainder: int = 0,
        height_division_factor: int | None = None,
        width_division_factor: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        if not isinstance(dataset_repeat, int) or isinstance(dataset_repeat, bool) or dataset_repeat <= 0:
            raise ValueError(f"dataset_repeat must be a positive integer; got {dataset_repeat!r}")
        if not isinstance(chosen_video_key, str) or not isinstance(rejected_video_key, str) or not chosen_video_key or not rejected_video_key or chosen_video_key == rejected_video_key:
            raise ValueError("chosen_video_key and rejected_video_key must be distinct non-empty strings")

        self.chosen_video_key = chosen_video_key
        self.rejected_video_key = rejected_video_key
        self.chosen_path_key = chosen_path_key or chosen_video_key
        self.rejected_path_key = rejected_path_key or rejected_video_key
        self.prompt_key = prompt_key
        required_columns = (self.chosen_path_key, self.rejected_path_key, self.prompt_key)
        if any(not isinstance(key, str) or not key.strip() for key in required_columns):
            raise ValueError(f"CSV column keys must be non-empty strings; got {required_columns!r}")
        if len(set(required_columns)) != len(required_columns):
            raise ValueError(f"chosen, rejected, and prompt CSV columns must be distinct; got {required_columns!r}")

        self.dataset_repeat = dataset_repeat
        self.height = self._optional_positive_int("height", height)
        self.width = self._optional_positive_int("width", width)
        self.num_frames = self._optional_positive_int("num_frames", num_frames)
        self.frame_interval = self._positive_int("frame_interval", frame_interval)
        self.max_pixels = self._optional_positive_int("max_pixels", max_pixels)
        if not isinstance(check_video_paths, bool):
            raise ValueError(f"check_video_paths must be a boolean; got {check_video_paths!r}")
        if video_loader is not None and not callable(video_loader):
            raise TypeError(f"video_loader must be callable or None; got {type(video_loader).__name__}")
        self.check_video_paths = check_video_paths
        self.video_loader = video_loader

        if not isinstance(resize_mode, str) or resize_mode not in {"center_crop", "stretch", "none"}:
            raise ValueError("resize_mode must be one of: 'center_crop', 'stretch', 'none'")
        if (self.height is None) != (self.width is None):
            raise ValueError("height and width must either both be set or both be omitted")
        self.resize_mode = resize_mode

        self._validate_divisibility(
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
        )

        if self.max_pixels is not None and self.height is not None and self.width is not None:
            pixels = self.height * self.width
            if pixels > self.max_pixels:
                raise ValueError(f"configured frame size {self.width}x{self.height} ({pixels} pixels) exceeds max_pixels={self.max_pixels}")

        base_path = os.fspath(dataset_base_path) if dataset_base_path is not None else ""
        self.dataset_base_path = Path(base_path).expanduser().resolve() if base_path else None
        manifests = self._manifest_paths(data_path_list, dataset_metadata_path)
        self.rows = self._read_manifests(manifests)
        if not self.rows:
            raise ValueError(f"DPO manifests contain no preference rows: {[str(path) for path in manifests]}")

        # Lazy import keeps importing teleboost.datasets lightweight.  The
        # transform registry imports torchvision, which is only needed when a
        # real raw-video dataset is instantiated.
        from teleboost.datasets.contract import Compose

        self.pipeline = Compose(transforms)

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
        return value

    @classmethod
    def _optional_positive_int(cls, name: str, value: int | None) -> int | None:
        return None if value is None else cls._positive_int(name, value)

    def _validate_divisibility(
        self,
        *,
        time_division_factor: int | None,
        time_division_remainder: int,
        height_division_factor: int | None,
        width_division_factor: int | None,
    ) -> None:
        if time_division_factor is not None:
            factor = self._positive_int("time_division_factor", time_division_factor)
            if not isinstance(time_division_remainder, int) or isinstance(time_division_remainder, bool):
                raise ValueError(f"time_division_remainder must be an integer; got {time_division_remainder!r}")
            if not 0 <= time_division_remainder < factor:
                raise ValueError(f"time_division_remainder must be in [0, {factor}); got {time_division_remainder}")
            if self.num_frames is not None and (self.num_frames - time_division_remainder) % factor:
                raise ValueError(f"num_frames={self.num_frames} must satisfy (num_frames - {time_division_remainder}) % {factor} == 0")
        for name, size, factor_value in (
            ("height", self.height, height_division_factor),
            ("width", self.width, width_division_factor),
        ):
            if factor_value is None:
                continue
            factor = self._positive_int(f"{name}_division_factor", factor_value)
            if size is not None and size % factor:
                raise ValueError(f"{name}={size} must be divisible by {factor}")

    @staticmethod
    def _manifest_paths(
        data_path_list: str | os.PathLike | Sequence[str | os.PathLike] | None,
        dataset_metadata_path: str | os.PathLike | None,
    ) -> list[Path]:
        if isinstance(data_path_list, (str, os.PathLike)):
            candidates = [data_path_list]
        elif data_path_list is None:
            candidates = []
        else:
            candidates = list(data_path_list)
        if not candidates and dataset_metadata_path:
            candidates = [dataset_metadata_path]
        if not candidates:
            raise ValueError("CSVPreferenceDPODataset requires a non-empty data_path_list or dataset_metadata_path")

        manifests = []
        for candidate in candidates:
            if not isinstance(candidate, (str, os.PathLike)) or not os.fspath(candidate).strip():
                raise ValueError(f"manifest paths must be non-empty path strings; got {candidate!r}")
            path = Path(candidate).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"DPO manifest does not exist or is not a file: {path}")
            manifests.append(path)
        return manifests

    @staticmethod
    def _required_cell(row: Mapping[str, Any], key: str, manifest: Path, line_number: int) -> str:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{manifest}:{line_number}: required CSV column {key!r} is empty")
        value = value.strip()
        if "\x00" in value:
            raise ValueError(f"{manifest}:{line_number}: CSV column {key!r} contains a NUL byte")
        return value

    def _resolve_video_path(self, value: str, manifest: Path, line_number: int, column: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            root = self.dataset_base_path or manifest.parent
            path = root / path
        path = path.resolve()
        if self.check_video_paths and not path.is_file():
            raise FileNotFoundError(f"{manifest}:{line_number}: video from column {column!r} does not exist or is not a file: {path}")
        return path

    def _read_manifests(self, manifests: Sequence[Path]) -> list[_PreferenceRow]:
        rows: list[_PreferenceRow] = []
        required = {self.chosen_path_key, self.rejected_path_key, self.prompt_key}
        for manifest in manifests:
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                if not fieldnames:
                    raise ValueError(f"DPO manifest has no CSV header: {manifest}")
                duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
                if duplicates:
                    raise ValueError(f"DPO manifest has duplicate CSV columns {duplicates}: {manifest}")
                missing = required - set(fieldnames)
                if missing:
                    raise ValueError(f"DPO manifest {manifest} is missing required columns {sorted(missing)}; available columns: {fieldnames}")

                for line_number, row in enumerate(reader, start=2):
                    if None in row:
                        raise ValueError(f"{manifest}:{line_number}: row has more cells than the CSV header")
                    if not any(isinstance(value, str) and value.strip() for value in row.values()):
                        continue
                    prompt = self._required_cell(row, self.prompt_key, manifest, line_number)
                    chosen_value = self._required_cell(row, self.chosen_path_key, manifest, line_number)
                    rejected_value = self._required_cell(row, self.rejected_path_key, manifest, line_number)
                    rows.append(
                        _PreferenceRow(
                            prompt=prompt,
                            chosen_path=self._resolve_video_path(chosen_value, manifest, line_number, self.chosen_path_key),
                            rejected_path=self._resolve_video_path(rejected_value, manifest, line_number, self.rejected_path_key),
                            manifest=manifest,
                            line_number=line_number,
                        )
                    )
        return rows

    def __len__(self) -> int:
        return len(self.rows) * self.dataset_repeat

    def _load_frames(self, path: Path, row: _PreferenceRow, branch: str) -> list[Any]:
        try:
            if self.video_loader is None:
                from .video_io import load_video

                frames = load_video(str(path))
            else:
                frames = self.video_loader(str(path))
        except Exception as exc:
            raise RuntimeError(f"{row.manifest}:{row.line_number}: failed to load {branch} video {path}: {exc}") from exc
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
            raise ValueError(f"{row.manifest}:{row.line_number}: {branch} video loader returned no frame sequence for {path}")
        return list(frames)

    def _sample_frames(self, frames: Sequence[Any]) -> list[Any]:
        if self.num_frames is None:
            return list(frames)
        if self.num_frames == 1:
            return [frames[0]]
        last = len(frames) - 1
        # Endpoint-preserving uniform sampling also handles short videos by
        # repeating frames deterministically, keeping DataLoader shapes fixed.
        indices = [round(i * last / (self.num_frames - 1)) for i in range(self.num_frames)]
        return [frames[index] for index in indices]

    def _resize_frames(self, frames: Sequence[Any], row: _PreferenceRow, branch: str) -> list[Any]:
        from PIL import Image, ImageOps

        prepared = []
        target_size = None if self.width is None or self.height is None else (self.width, self.height)
        resampling = getattr(Image, "Resampling", Image)
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                try:
                    frame = Image.fromarray(np.asarray(frame))
                except Exception as exc:
                    raise TypeError(f"{row.manifest}:{row.line_number}: {branch} frame {frame_index} cannot be converted to a PIL image (got {type(frame).__name__})") from exc
            frame = frame.convert("RGB")
            if target_size is not None and frame.size != target_size:
                if self.resize_mode == "center_crop":
                    frame = ImageOps.fit(frame, target_size, method=resampling.BICUBIC, centering=(0.5, 0.5))
                elif self.resize_mode == "stretch":
                    frame = frame.resize(target_size, resample=resampling.BICUBIC)
                elif self.resize_mode == "none":
                    raise ValueError(f"{row.manifest}:{row.line_number}: {branch} frame {frame_index} has size {frame.size}, expected {target_size} with resize_mode='none'")
            prepared.append(frame)
        return prepared

    @staticmethod
    def _first_frame_tensor(frame: Any) -> torch.Tensor:
        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an RGB first frame, got array shape {array.shape}")
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().unsqueeze(0)

    def _prepare_branch(self, path: Path, row: _PreferenceRow, branch: str) -> dict[str, Any]:
        frames = self._load_frames(path, row, branch)
        frames = self._sample_frames(frames)
        frames = self._resize_frames(frames, row, branch)
        first_frame = self._first_frame_tensor(frames[0])
        raw = {
            "video": frames,
            self.prompt_key: row.prompt,
            "frame_interval": self.frame_interval,
            # Keep both names because the production packing config carries
            # input_image while the Wan encoder consumes raw_first_image.
            "raw_first_image": first_frame,
            "input_image": first_frame.clone(),
        }
        result = self.pipeline(raw)
        if result is None:
            raise RuntimeError(f"{row.manifest}:{row.line_number}: transform pipeline rejected {branch} video {path}")
        if not isinstance(result, Mapping):
            raise TypeError(f"transform pipeline must return a Mapping; got {type(result).__name__}")
        result = dict(result)
        self._validate_raw_branch(result, row, branch)
        return result

    @staticmethod
    def _validate_raw_branch(branch_data: Mapping[str, Any], row: _PreferenceRow, branch: str) -> None:
        missing = {"images", "struct_prompt"} - set(branch_data)
        if missing:
            raise ValueError(f"{row.manifest}:{row.line_number}: transformed {branch} branch is missing encoder inputs {sorted(missing)}")
        images = branch_data["images"]
        if not torch.is_tensor(images) or images.ndim != 4:
            shape = tuple(images.shape) if torch.is_tensor(images) else None
            raise TypeError(f"{row.manifest}:{row.line_number}: transformed {branch}.images must be a CPU Tensor[T,C,H,W]; got {type(images).__name__} shape={shape}")
        if images.device.type != "cpu":
            raise ValueError(f"{row.manifest}:{row.line_number}: transformed {branch}.images must remain on CPU")
        prompts = branch_data["struct_prompt"]
        if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)) or not prompts:
            raise TypeError(f"{row.manifest}:{row.line_number}: transformed {branch}.struct_prompt must be a non-empty sequence of strings")
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError(f"{row.manifest}:{row.line_number}: transformed {branch}.struct_prompt contains an empty/non-string prompt")

    def __getitem__(self, idx: int) -> dict[str, dict[str, Any]]:
        if not isinstance(idx, int):
            raise TypeError(f"dataset index must be an integer; got {type(idx).__name__}")
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(idx)
        row = self.rows[idx % len(self.rows)]
        return {
            self.chosen_video_key: self._prepare_branch(row.chosen_path, row, "chosen"),
            self.rejected_video_key: self._prepare_branch(row.rejected_path, row, "rejected"),
        }
