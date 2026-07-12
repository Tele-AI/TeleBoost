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
"""Wan-1.3B T2V DiT-arch config (yaml ``config_path`` default).

Mirrors dims from the upstream HF config.json at the canonical
``Wan2.1-T2V-1.3B`` release (dim=1536, ffn_dim=8960, num_heads=12,
num_layers=30, in/out_dim=16, text_dim=4096 = T5-xl-encoder,
freq_dim=256, eps=1e-6, patch_size=(1, 2, 2), no image conditioning
for T2V).

The pre-converted teletron-format checkpoint is the consumer:
``teleboost-convert-wan-to-teletron`` renames Wan-native attention/
embedding keys to TeleBoost's teletron schema (``.q→.query`` /
``.k→.key`` / ``patch_embedding→patch_emb`` / ...). That's why
``type="ParallelWanTeletronModel"`` below — the model class expects
teletron keys.
"""

from teleboost.engines.teletron.config import Config

_WAN_1_3B_T2V_DIT = Config(
    {
        # Registered class in teleboost.models.build.MODEL_REGISTRY matching the
        # naming convention of the pre-converted teletron ckpt (Wan → teletron
        # rename applied by teleboost-convert-wan-to-teletron default mode).
        "type": "ParallelWanTeletronModel",
        # Mirrors HF config.json at
        # /path/to/wan_ckpt/Wan2.1-T2V-1.3B/config.json
        # plus the constants the model expects but HF leaves implicit:
        # patch_size, has_image_input/pos_emb, text_dim=4096 (T5-xl encoder).
        "config": Config(
            {
                "dim": 1536,
                "in_dim": 16,
                "ffn_dim": 8960,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1.0e-6,
                "patch_size": (1, 2, 2),
                "num_heads": 12,
                "num_layers": 30,
                "has_image_input": False,
                "has_image_pos_emb": False,
            }
        ),
        # DPO training knobs read by the Wan-DPO loss:
        #   beta — sigmoid temperature on (loss_reject - loss_chosen).
        #          0.1 is the Wan-DPO default.
        "train": Config(
            {
                "dpo": Config(
                    {
                        "enable": True,
                        "beta": 0.1,
                    }
                ),
            }
        ),
    }
)


# Resolved by ``set_config()`` via ``teletron_args.config_path``.
config = Config(
    {
        "model_config": Config(
            {
                "dit": _WAN_1_3B_T2V_DIT,
                # Encoder block is read by Trainer.__init__ at line 91; Phase
                # 2.2.c doesn't exercise that path (no dataloader yet), but the
                # Trainer code uses a getattr-chain that would crash if
                # the encoder key were absent — define defensively.
                "encoder": Config({"type": "noop"}),
                # Diffusion training knobs read by Wan-DPO timestep sampling:
                #   {min,max}_timestep_boundary — fraction of
                #   FlowMatchScheduler.num_train_timesteps to sample from.
                #   [0.0, 1.0] covers the full scheduler range.
                "training": Config(
                    {
                        "diffusion": Config(
                            {
                                "min_timestep_boundary": 0.0,
                                "max_timestep_boundary": 1.0,
                            }
                        ),
                    }
                ),
            }
        ),
        "eval": Config({"eval_time_steps": None}),
    }
)
