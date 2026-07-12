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
from functools import partial
from typing import Any

import torch

from teleboost.engines.teletron.distributed.base_encoder import BaseEncoder
from teleboost.models.wan.teletron.models.dit import WanTeletronImageEncoder, WanTeletronPrompter, WanTeletronTextEncoder, WanTeletronVideoVAE, WanTeletronVideoVAE_2_2
from teleboost.models.wan.teletron.models.dit.diffsynth_wan_video_vae import (
    WanVideoVAE as DiffSynthWanVideoVAE,
)
from teleboost.models.wan.teletron.models.dit.diffsynth_wan_video_vae import (
    WanVideoVAE38 as DiffSynthWanVideoVAE38,
)
from teleboost.models.wan.teletron.wan_teletron_encoder_utils import (
    get_context,
    get_depth_latents,
    get_fake_latents,
    get_img_clip_feature,
    get_img_emb_y,
    get_latents,
    get_noise,
    get_unprompt_emb,
)
from teleboost.engines.teletron import set_config

WORK_FN = {
    "context": get_context,
    "img_clip_feature": get_img_clip_feature,
    "img_emb_y": get_img_emb_y,
    "latents": get_latents,
    "noise": get_noise,
    "fake_latents": get_fake_latents,
    "prompt_emb": get_context,
    "unprompt_emb": get_unprompt_emb,
    "depth_latents": get_depth_latents,
}

PROPERTY_DIMS = {
    "context": 3,
    "img_clip_feature": 3,
    "img_emb_y": 5,
    "latents": 5,
    "noise": 5,
    "fake_latents": 5,
    "prompt_emb": 3,
    "unprompt_emb": 3,
    "depth_latents": 5,
}


class WanTeletronEncoder(BaseEncoder):
    """Concrete encoder implementation for the Wan video model."""

    @staticmethod
    def get_output_schema() -> list[str]:
        """Return the fixed names and order of this encoder's output tensors."""
        return set_config().get("model_config", None).get("encoder", None).get("encoder_schema", ["context", "latents"])

    def __init__(self, device: torch.device):
        super().__init__(device)
        encoder_model_config = set_config().get("model_config", None).get("encoder", None)
        if encoder_model_config is None:
            raise ValueError("Encoder model config not found.")

        self.vae_path = encoder_model_config.get("vae", None).get("path", None)
        self.vae_type = encoder_model_config.get("vae", None).get("type", "WanTeletronVideoVAE_2_1")
        self.tiler_kwargs = encoder_model_config.get("vae", None).get("tiler_kwargs", {})
        self.vae_compile = encoder_model_config.get("vae", None).get("torch_compile", False)
        self.compression_cfg = encoder_model_config.get("vae", None).get("compression", None)
        if self.tiler_kwargs is None:
            self.tiler_kwargs = dict(
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )
        self.text_encoder_path = encoder_model_config.get("text_encoder", None).get("path", None)
        self.tokenizer_path = encoder_model_config.get("text_encoder", None).get("tokenizer_path", None)

        if encoder_model_config.get("image_encoder", None) is not None:
            self.image_encoder_path = encoder_model_config.get("image_encoder", None).get("path", None)
        else:
            self.image_encoder_path = None
        if encoder_model_config.get("image_encoder", None) is not None:
            self.image_encoder_compile = encoder_model_config.get("image_encoder", None).get("torch_compile", False)
        else:
            self.image_encoder_compile = None

        if encoder_model_config.get("depth_model", None) is not None:
            self.depth_model_path = encoder_model_config.get("depth_model", None).get("path", None)
        else:
            self.depth_model_path = None

        if not self.vae_path or not self.text_encoder_path or not self.tokenizer_path:
            raise ValueError("WanTeletronEncoder requires the 'text_encoder_path' and 'tokenizer_path' arguments.")

        # Initialize model components to None; they are loaded in setup()
        self.text_encoder = None
        self.image_encoder = None
        self.vae = None
        self.prompter = None
        self.depth_model = None
        self.work_fn = WORK_FN

    def setup(self) -> None:
        """Load all required teletron model components onto the target device."""
        print(f"Setting up WanTeletronEncoder on device {self.device}...")
        print(f"Init VAE params: type={self.vae_type} path={self.vae_path} tiler_kwargs={self.tiler_kwargs} torch_compile={self.vae_compile}")
        print(f"Loading VAE model... {self.vae_path}")
        if self.vae_type in ("DiffSynthWanVideoVAE", "diffsynth_wan_video_vae"):
            self.vae = DiffSynthWanVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.vae_compile and hasattr(self.vae, "model") and hasattr(self.vae.model, "encode"):
                self.vae.model.encode = torch.compile(self.vae.model.encode, dynamic=True)
                print("torch.compile DiffSynth VAE model... ")
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 8, 8)
        elif self.vae_type in ("DiffSynthWanVideoVAE38", "diffsynth_wan_video_vae38"):
            self.vae = DiffSynthWanVideoVAE38().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.vae_compile and hasattr(self.vae, "model") and hasattr(self.vae.model, "encode"):
                self.vae.model.encode = torch.compile(self.vae.model.encode, dynamic=True)
                print("torch.compile DiffSynth VAE38 model... ")
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 8, 8)
        elif self.vae_type == "WanTeletronVideoVAE_2_1":
            self.vae = WanTeletronVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.vae_compile:
                self.vae.model.encode = torch.compile(self.vae.model.encode, dynamic=True)
                print("torch.compile VAE model... ")
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 8, 8)
        else:
            self.vae = WanTeletronVideoVAE_2_2().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 16, 16)
        self.vae.model.load_state_dict(torch.load(self.vae_path, map_location="cpu", weights_only=True), strict=True)

        print(f"Loading Text Encoder model... {self.text_encoder_path}")
        self.text_encoder = WanTeletronTextEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(torch.load(self.text_encoder_path, map_location="cpu", weights_only=True), strict=True)
        self.prompter = WanTeletronPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.tokenizer_path)

        if self.image_encoder_path is not None:
            print(f"Loading Image Encoder model... {self.image_encoder_path}")
            self.image_encoder = WanTeletronImageEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.image_encoder.model.load_state_dict(torch.load(self.image_encoder_path, map_location="cpu", weights_only=True), strict=False)

        if self.depth_model_path is not None:
            print(f"Loading Depth Model... {self.depth_model_path}")
            from video_depth_anything.video_depth import VideoDepthAnything

            self.depth_model = VideoDepthAnything().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.depth_model.load_state_dict(torch.load(self.depth_model_path, map_location="cpu", weights_only=True), strict=True)

        if self.image_encoder_compile:
            self.image_encoder.encode_image = torch.compile(self.image_encoder.encode_image)
            print("torch.compile Image Encoder model... ")
        for key, val in self.work_fn.items():
            self.work_fn[key] = self.prepare_work_fn(key, val)

        print("WanTeletronEncoder setup complete.")

    def prepare_work_fn(self, target, work_fn):
        if target == "context":
            return partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
        elif target == "img_clip_feature":
            return partial(work_fn, image_encoder=self.image_encoder, dtype=torch.bfloat16)
        elif target == "img_emb_y":
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16, compression=self.compression, tiler_kwargs=self.tiler_kwargs)
        elif target == "latents":
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        elif target == "noise":
            return partial(work_fn, dtype=torch.bfloat16, compression=self.compression)
        elif target == "fake_latents":
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        elif target == "prompt_emb":
            return partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
        elif target == "unprompt_emb":
            if not getattr(self, "unprompt_emb", None):
                self.unprompt_emb = partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
            return self.unprompt_emb
        elif target == "depth_latents":
            return partial(work_fn, depth_model=self.depth_model, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        else:
            return work_fn

    def _is_dpo_batch(self, raw_batch: dict) -> bool:
        dataset_config = set_config().get("dataset", {})
        chosen_key = dataset_config.get("chosen_video_key", "chosen")
        rejected_key = dataset_config.get("rejected_video_key", "rejected")
        return isinstance(raw_batch, dict) and chosen_key in raw_batch and rejected_key in raw_batch

    def _encode_single(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        schema = self.get_output_schema()
        out = {}
        for key in schema:
            out[key] = self.work_fn[key](batch=raw_batch)
        return out

    def _encode_dpo(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        # Cache the streams on self to avoid recreating them on every call
        if not hasattr(self, "_chosen_stream"):
            self._chosen_stream = torch.cuda.Stream(device=self.device)
            self._rejected_stream = torch.cuda.Stream(device=self.device)

        dataset_config = set_config().get("dataset", {})
        chosen_key = dataset_config.get("chosen_video_key", "chosen")
        rejected_key = dataset_config.get("rejected_video_key", "rejected")

        schema = self.get_output_schema()
        out = {}

        # 1. Prompt encoding (default stream)
        shared_input = raw_batch[chosen_key]
        for key in ["context", "prompt_emb", "unprompt_emb"]:
            if key in schema:
                out[key] = self.work_fn[key](batch=shared_input)

        # Ensure prompt encoding on the default stream finishes before the parallel streams read it
        current_stream = torch.cuda.current_stream()
        self._chosen_stream.wait_stream(current_stream)
        self._rejected_stream.wait_stream(current_stream)

        # 2. Chosen/Rejected branches in parallel
        branch_outputs = {chosen_key: {}, rejected_key: {}}

        # Inner per-branch processing logic
        def _proc_branch(branch_name, stream):
            with torch.cuda.stream(stream):
                branch_input = raw_batch[branch_name]
                for key in schema:
                    if key not in ["context", "prompt_emb", "unprompt_emb"]:
                        branch_outputs[branch_name][key] = self.work_fn[key](batch=branch_input)

        # The per-branch logic stays as originally written; note the wait_stream above
        _proc_branch(chosen_key, self._chosen_stream)
        _proc_branch(rejected_key, self._rejected_stream)

        # 3. Synchronize
        self._chosen_stream.synchronize()
        self._rejected_stream.synchronize()

        out.update(branch_outputs)
        return out

    def encode(self, raw_batch: dict[str, Any]) -> list[Any] | list[list[Any]]:
        """
        Encode a data batch with the teletron models.

        Args:
            raw_batch: A single data sample (dict) or a batch of data samples (list of dicts).

        Returns:
            If the input is a single sample, a list of encoded tensors.
            If the input is a list of samples, a list of lists with the encoded results per sample.
        """
        if not self._is_dpo_batch(raw_batch):
            res_batch = self._encode_single(raw_batch)
        else:
            res_batch = self._encode_dpo(raw_batch)
        # dumper = get_dumper()
        # dumper.dump(stage="wan_teletron_encoder_output", obj=res_batch, data_id=None)
        return res_batch
