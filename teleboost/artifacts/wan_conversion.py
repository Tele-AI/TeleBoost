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
"""Convert a Wan-AI Hugging Face checkpoint into a TeleTron --load directory.

Reads HF safetensors (single file or sharded; ``*.safetensors`` glob accepted),
renames Wan-style attention / embedding parameter names to the TeleBoost
(teletron) naming convention used by ``ParallelWanTeletronModel``, and writes a
megatron-format checkpoint directory consumable by ``--load``.

Wan → TeleBoost rename rules (applied in order; only DiT keys, encoder /
VAE / CLIP weights are loaded separately by the encoder pipeline):

    .k.        -> .key.
    .q.        -> .query.
    .v.        -> .value.
    .o.        -> .out_proj.
    .norm_q.   -> .norm_query.
    .norm_k.   -> .norm_key.
    .k_img.    -> .img_key.
    .v_img.    -> .img_value.
    .norm_k_img. -> .norm_image_key.
    patch_embedding.  -> patch_emb.
    time_embedding.   -> time_emb.
    text_embedding.   -> text_emb.
    time_projection.  -> time_proj.

Output layout (megatron "release" format):

    <dst>/
      latest_checkpointed_iteration.txt   ("release")
      release/
        mp_rank_00/
          model_optim_rng.pt              { args: None, checkpoint_version: 3.0,
                                            model: <state_dict> }

Pass ``--load <dst>`` to ``pretrain_dpo_i2v.py`` / ``train_dpo.sh`` to bootstrap
training from these weights.

Examples
--------
    # Wan2.1-T2V-1.3B — single safetensors file
    teleboost-convert-wan-to-teletron \\
        --src /models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \\
        --dst /path/to/Wan2.1-T2V-1.3B-teletron

    # Wan2.2-I2V-A14B — 6 sharded safetensors under high_noise_model/
    teleboost-convert-wan-to-teletron \\
        --src '/models/Wan2.2-I2V-A14B/high_noise_model/*.safetensors' \\
        --dst /path/to/Wan2.2-I2V-A14B-high-teletron

    # Pre-renamed weights already in teletron naming (skip the rename step)
    teleboost-convert-wan-to-teletron --src ... --dst ... --no-rename

Limitations
-----------
* Tensor-parallel size is fixed at 1 (``mp_rank_00``). For TP > 1 you must
  shard the resulting state_dict yourself.
* No optimizer / RNG state is populated — this is a *release* checkpoint
  for starting a fresh fine-tune, not a mid-run resume.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_WAN_TO_TELEAI_RENAMES = [
    (re.compile(r"\.k\."), ".key."),
    (re.compile(r"\.q\."), ".query."),
    (re.compile(r"\.v\."), ".value."),
    (re.compile(r"\.o\."), ".out_proj."),
    (re.compile(r"\.norm_q\."), ".norm_query."),
    (re.compile(r"\.norm_k\."), ".norm_key."),
    (re.compile(r"\.k_img\."), ".img_key."),
    (re.compile(r"\.v_img\."), ".img_value."),
    (re.compile(r"\.norm_k_img\."), ".norm_image_key."),
    (re.compile(r"^patch_embedding\."), "patch_emb."),
    (re.compile(r"^time_embedding\."), "time_emb."),
    (re.compile(r"^text_embedding\."), "text_emb."),
    (re.compile(r"^time_projection\."), "time_proj."),
]

# Inverse rules — applied most-specific-first so .norm_key. doesn't get
# pre-empted by .key.  (Forward rules are written with literal-dot regex
# anchors so the order doesn't matter there; the inverse direction is
# stricter because the renamed forms can be substrings of each other.)
_TELEAI_TO_WAN_RENAMES = [
    (re.compile(r"\.norm_image_key\."), ".norm_k_img."),
    (re.compile(r"\.norm_query\."), ".norm_q."),
    (re.compile(r"\.norm_key\."), ".norm_k."),
    (re.compile(r"\.img_key\."), ".k_img."),
    (re.compile(r"\.img_value\."), ".v_img."),
    (re.compile(r"\.key\."), ".k."),
    (re.compile(r"\.query\."), ".q."),
    (re.compile(r"\.value\."), ".v."),
    (re.compile(r"\.out_proj\."), ".o."),
    (re.compile(r"^patch_emb\."), "patch_embedding."),
    (re.compile(r"^time_emb\."), "time_embedding."),
    (re.compile(r"^text_emb\."), "text_embedding."),
    (re.compile(r"^time_proj\."), "time_projection."),
]


def _expand_paths(src: str | Iterable[str]) -> list[str]:
    """Expand a single path / glob / list of paths into a sorted file list."""
    if isinstance(src, str):
        sources = [src]
    else:
        sources = list(src)
    expanded: list[str] = []
    for s in sources:
        if any(c in s for c in "*?[]"):
            matched = sorted(glob.glob(s))
            if not matched:
                raise FileNotFoundError(f"glob {s!r} matched no files")
            expanded.extend(matched)
        else:
            if not os.path.exists(s):
                raise FileNotFoundError(s)
            expanded.append(s)
    duplicates = sorted({path for path in expanded if expanded.count(path) > 1})
    if duplicates:
        raise ValueError(f"checkpoint sources contain duplicate paths: {duplicates}")
    return expanded


def load_state_dict(paths: list[str]) -> OrderedDict[str, torch.Tensor]:
    """Load and concat state dicts from one or more .safetensors / .pt files."""
    import safetensors.torch
    import torch

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for p in paths:
        if p.endswith(".safetensors"):
            with open(p, "rb") as fh:
                shard = safetensors.torch.load(fh.read())
        else:
            shard = torch.load(p, map_location="cpu", weights_only=True)
        duplicate_keys = sorted(set(state_dict).intersection(shard))
        if duplicate_keys:
            raise ValueError(f"checkpoint shard {p} repeats {len(duplicate_keys)} parameter keys: {duplicate_keys[:5]}")
        before = len(state_dict)
        state_dict.update(shard)
        print(f"  loaded {p}: {len(shard)} keys (+{len(state_dict) - before})")
    return state_dict


def rename_wan_to_teletron(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Apply Wan → teletron naming rules to every key."""
    renamed: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in _WAN_TO_TELEAI_RENAMES:
            new_key = pattern.sub(replacement, new_key)
        if new_key in renamed:
            raise ValueError(f"Wan → TeleTron rename collision at {new_key!r}")
        renamed[new_key] = value
    return renamed


def rename_teletron_to_wan(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Apply teletron → Wan naming rules (inverse of rename_wan_to_teletron)."""
    renamed: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in _TELEAI_TO_WAN_RENAMES:
            new_key = pattern.sub(replacement, new_key)
        if new_key in renamed:
            raise ValueError(f"TeleTron → Wan rename collision at {new_key!r}")
        renamed[new_key] = value
    return renamed


def load_teletron_release(dst: str) -> OrderedDict[str, torch.Tensor]:
    """Load the model state_dict back from a megatron 'release' directory."""
    import torch

    ckpt_path = os.path.join(dst, "release", "mp_rank_00", "model_optim_rng.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return OrderedDict((k, v) for k, v in payload["model"].items())


def roundtrip_check(
    src_state_dict: OrderedDict[str, torch.Tensor],
    teletron_state_dict: OrderedDict[str, torch.Tensor],
) -> tuple[bool, list[str]]:
    """Verify HF → teletron → HF reverses to bit-exact identity.

    Returns (ok, diagnostics). ``ok`` is True iff every key matches and
    every tensor is element-wise equal (no dtype / device mismatch).
    """
    import torch

    reversed_sd = rename_teletron_to_wan(teletron_state_dict)
    diagnostics: list[str] = []
    src_keys = set(src_state_dict.keys())
    rev_keys = set(reversed_sd.keys())
    missing = src_keys - rev_keys
    extra = rev_keys - src_keys
    if missing:
        diagnostics.append(f"missing {len(missing)} keys after roundtrip: {sorted(missing)[:5]}...")
    if extra:
        diagnostics.append(f"extra {len(extra)} keys after roundtrip: {sorted(extra)[:5]}...")
    common = src_keys & rev_keys
    mismatched = []
    for k in common:
        a, b = src_state_dict[k], reversed_sd[k]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            mismatched.append(k)
    if mismatched:
        diagnostics.append(f"{len(mismatched)} tensors differ: {mismatched[:5]}...")
    ok = not (missing or extra or mismatched)
    return ok, diagnostics


def save_teletron_release(
    state_dict: OrderedDict[str, torch.Tensor],
    dst: str,
    checkpoint_version: float = 3.0,
) -> str:
    """Write a megatron 'release' checkpoint directory at ``dst``.

    Returns the path to the saved ``model_optim_rng.pt`` for convenience.
    """
    import torch

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("release")
    mp_dir = os.path.join(dst, "release", "mp_rank_00")
    os.makedirs(mp_dir, exist_ok=True)
    ckpt_path = os.path.join(mp_dir, "model_optim_rng.pt")
    payload = OrderedDict(
        args=None,
        checkpoint_version=checkpoint_version,
        model=OrderedDict((k, v) for k, v in state_dict.items()),
    )
    torch.save(payload, ckpt_path)
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="teleboost-convert-wan-to-teletron",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        required=True,
        nargs="+",
        help=("One or more paths / glob patterns pointing at HF safetensors or .pt files. Globs are expanded and sorted before loading."),
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output directory for the TeleBoost --load checkpoint.",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help=("Skip the Wan → teletron key rename step. Use this when the input weights already use teletron naming (e.g., a previous TeleBoost release checkpoint)."),
    )
    parser.add_argument(
        "--roundtrip-check",
        action="store_true",
        help=("After writing the teletron checkpoint, reload it, apply the inverse rename, and verify it matches the source state_dict key-for-key and bit-exact. Reports OK / lists divergences."),
    )
    args = parser.parse_args()

    paths = _expand_paths(args.src)
    print(f"loading {len(paths)} file(s)...")
    state_dict = load_state_dict(paths)
    print(f"  {len(state_dict)} total keys")
    src_state_dict = OrderedDict(state_dict) if args.roundtrip_check else None

    if not args.no_rename:
        state_dict = rename_wan_to_teletron(state_dict)
        print("renamed keys: Wan → teletron naming")

    out = save_teletron_release(state_dict, args.dst)
    print(f"DONE: wrote {out}")
    print(f"      pass --load {args.dst} to load these weights")

    if args.roundtrip_check:
        print()
        print("=== roundtrip check: teletron -> Wan, compare to source ===")
        reloaded = load_teletron_release(args.dst)
        ok, diagnostics = roundtrip_check(src_state_dict, reloaded)
        if ok:
            print(f"OK: {len(reloaded)} keys, all tensors bit-exact match source")
        else:
            print("FAIL:")
            for d in diagnostics:
                print(f"  {d}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
