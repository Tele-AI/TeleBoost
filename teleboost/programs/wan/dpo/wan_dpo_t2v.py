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
"""Smoke-test override of wan_dpo_i2v.py: 1.3B T2V + 10-pair real-video DPO.

Runs the production training graph end-to-end against a tiny 10-pair
DPO CSV built from /path/to/.../paired_videos. Uses 1.3B random-init DiT
(via ParallelWanTeletronModel) so no DiT checkpoint is required; VAE / T5
load from the downloaded Wan2.1-T2V-1.3B; CLIP image-encoder is dropped
(t2v doesn't need it).

Override entry: pass `--config-path config.wan_dpo_t2v.config`.
"""

import copy
import os

from .wan_dpo_i2v import config as _base

config = copy.deepcopy(_base)

# === 10-pair real-data CSV (kling vs hailuo per paired_videos dir) ===
_PAIRS_CSV = os.environ.get(
    "DPO_SMOKE_CSV",
    "/path/to/dpo_smoke/pairs_10.csv",
)
config["dataset"]["dataset_base_path"] = ""
config["dataset"]["dataset_metadata_path"] = _PAIRS_CSV
config["dataset"]["data_path_list"] = [_PAIRS_CSV]
# CSV columns are positive_video_path / negative_video_path, while the raw
# and encoded DPO branch names remain the canonical chosen / rejected.
config["dataset"]["chosen_path_key"] = "positive_video_path"
config["dataset"]["rejected_path_key"] = "negative_video_path"
config["dataset"]["dataset_repeat"] = 1
# 10-pair smoke at lower res to keep encoder + 2-iter forward fast
config["dataset"]["height"] = 480
config["dataset"]["width"] = 832
config["dataset"]["num_frames"] = 49

# Eval inherits the placeholder /path/to/... CSV from the canonical config and
# would crash build_train_valid_test_datasets — point it at the same 10-pair
# CSV so eval runs cheaply (we set --eval-interval=100000 so it never fires).
config["eval"]["data_path_list"] = [_PAIRS_CSV]

# === Shrink DiT 14B -> 1.3B T2V ===
config["model_config"]["dit"]["config"].update(
    dict(
        has_image_input=False,
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        num_heads=12,
        num_layers=30,
    )
)
config["model_config"]["dit"]["train"]["extra_inputs"] = []  # t2v: no input_image

# === Point encoders at the downloaded 1.3B weights ===
_W = os.environ.get("WAN13B_DIR", "/path/to/wan_ckpt/Wan2.1-T2V-1.3B")
config["model_config"]["encoder"]["vae"]["path"] = f"{_W}/Wan2.1_VAE.pth"
config["model_config"]["encoder"]["text_encoder"]["path"] = f"{_W}/models_t5_umt5-xxl-enc-bf16.pth"
config["model_config"]["encoder"]["text_encoder"]["tokenizer_path"] = f"{_W}/google/umt5-xxl"

# === T2V: drop CLIP image_encoder + img_emb_y schema ===
config["model_config"]["encoder"].pop("image_encoder", None)
config["model_config"]["encoder"]["encoder_schema"] = ["context", "latents"]
