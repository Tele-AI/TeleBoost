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
"""Prepare Wan training data: ensure every prompt has both `context_path`
(positive prompt embedding) and `context_null_path` (shared negative-prompt
embedding for CFG).

Single unified, idempotent entry point — replaces the legacy trio
(`preprocess_wan_data.py`, `preprocess_wan_embeddings.py`,
`preprocess_wan_embeddings_fromlist.py`).

Inputs accepted:
- Plain `.txt`: one prompt per line.
- `.json`: list of dicts, each with at least a `caption` field; existing
  `context_path` / `context_null_path` are reused as-is.

Outputs (under `--output_dir`):
- `context_<i>.npy`              umT5-XXL embedding of each prompt
- `context_null.npy`             umT5-XXL embedding of the negative prompt (shared)
- `processed_wan_prompt.json`    per-row `{caption, context_path, context_null_path}`

Idempotency:
- If `context_null.npy` exists, the negative-prompt encode is skipped.
- Per-row: if both `context_path` AND `context_null_path` are present and the
  files exist, the row is skipped (no T5 forward).
- If every row is already complete and `context_null.npy` exists, T5 is never
  loaded.

Usage:
    python teleboost/datasets/preprocessing/wan.py \\
        --input prompts.txt \\
        --output_dir data/processed/ \\
        --wan_model_path /path/to/Wan2.1-T2V-1.3B
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# Wan official Chinese negative prompt (matches `wan/configs/shared_config.py`).
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def _require_wan_runtime() -> None:
    """Require the separately installed, top-level upstream Wan runtime."""
    if importlib.util.find_spec("wan") is None:
        raise RuntimeError("Wan runtime is not importable as the top-level 'wan' package. Install the upstream Wan runtime or add its installation root to PYTHONPATH before running this command; the TeleBoost core wheel does not bundle it.")


def _load_input(path: Path) -> list[dict]:
    """Read txt or JSON; return list of dicts each with at least `caption`."""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        rows = []
        for item in data:
            if isinstance(item, dict):
                caption = item.get("caption") or item.get("text")
                if not caption:
                    raise ValueError(f"JSON row missing caption/text: {item}")
                rows.append({**item, "caption": caption})
            else:
                rows.append({"caption": str(item)})
        return rows
    if path.suffix == ".txt":
        with open(path, encoding="utf-8") as f:
            return [{"caption": line.strip()} for line in f if line.strip()]
    raise ValueError(f"Unsupported input extension: {path.suffix} (expected .txt or .json)")


def _row_complete(row: dict) -> bool:
    cp, np_ = row.get("context_path"), row.get("context_null_path")
    return bool(cp and np_ and os.path.isfile(cp) and os.path.isfile(np_))


def _build_t5(wan_model_path: Path, device: torch.device):
    """Lazy: only called when something actually needs encoding."""
    import torch

    # Apply teleboost runtime patches BEFORE importing wan — wan/modules/model.py
    # consumes `verl.utils.ulysses.{gate_with_cp_grad_reduce, modulate_with_cp_grad_reduce}`,
    # which teleboost.patches.ulysses injects. Without this import,
    # `from wan...` raises ImportError.
    from teleboost import apply_runtime_patches

    apply_runtime_patches()

    _require_wan_runtime()
    t2v_1_3B = importlib.import_module("wan.configs").t2v_1_3B
    T5EncoderModel = importlib.import_module("wan.modules.t5").T5EncoderModel

    print("Loading T5 encoder (one-shot)...")
    return T5EncoderModel(
        text_len=t2v_1_3B.text_len,
        dtype=torch.float32,
        device=device,
        checkpoint_path=str(wan_model_path / t2v_1_3B.t5_checkpoint),
        tokenizer_path=str(wan_model_path / t2v_1_3B.t5_tokenizer),
    )


def _encode(text_encoder: Any, prompt: str, device: torch.device) -> Any:
    import torch

    with torch.no_grad():
        out = text_encoder([prompt], device=device)
    arr = out[0] if isinstance(out, list) else out
    return arr.detach().cpu().numpy()


def prepare(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from tqdm import tqdm

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_input(input_path)
    print(f"Loaded {len(rows)} prompts from {input_path}")

    null_path = output_dir / "context_null.npy"
    text_encoder = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass 1: figure out what (if anything) needs T5.  A valid custom
    # context_null_path belongs to that row and must not be overwritten by a
    # newly generated global embedding.
    rows_needing_encode = [i for i, r in enumerate(rows) if not (r.get("context_path") and os.path.isfile(r["context_path"]))]
    rows_needing_null = [i for i, r in enumerate(rows) if not (r.get("context_null_path") and os.path.isfile(r["context_null_path"]))]
    null_needs_encode = bool(rows_needing_null) and not null_path.is_file()

    if rows_needing_encode or null_needs_encode:
        text_encoder = _build_t5(Path(args.wan_model_path), device)

    # Encode the shared negative prompt only when at least one row needs it.
    if null_needs_encode:
        print(f"Encoding negative prompt -> {null_path}")
        np.save(str(null_path), _encode(text_encoder, args.negative_prompt, device))
    elif rows_needing_null:
        print(f"Reusing existing {null_path}")
    else:
        print("All rows already have a valid context_null_path — preserving them.")

    # Encode any positive prompts that don't already have a saved embedding.
    if rows_needing_encode:
        print(f"Encoding {len(rows_needing_encode)}/{len(rows)} positive prompts...")
        for i in tqdm(rows_needing_encode, desc="encoding"):
            arr = _encode(text_encoder, rows[i]["caption"], device)
            ctx_path = output_dir / f"context_{i:06d}.npy"
            np.save(str(ctx_path), arr)
            rows[i]["context_path"] = str(ctx_path)
            if device.type == "cuda" and (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
    else:
        print("All positive prompts already have context_path — skipping T5 forward.")

    # Fill only missing/invalid values; preserve valid per-row embeddings.
    for i in rows_needing_null:
        rows[i]["context_null_path"] = str(null_path)

    out_json = output_dir / "processed_wan_prompt.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    skipped = len(rows) - len(rows_needing_encode)
    print(f"\nDone: wrote {out_json}")
    print(f"  positive embeddings encoded: {len(rows_needing_encode)} (skipped {skipped} already-encoded)")
    print(f"  context_null.npy: {null_path}")


def main():
    p = argparse.ArgumentParser(
        prog="teleboost-prepare-wan-data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="prompts.txt (one per line) or prompts.json (list of {caption, ...})")
    p.add_argument("--output_dir", required=True, help="where to write context_*.npy + processed_wan_prompt.json")
    p.add_argument("--wan_model_path", required=True, help="Wan checkpoint dir (for the T5 encoder)")
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, help="negative prompt for CFG")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
