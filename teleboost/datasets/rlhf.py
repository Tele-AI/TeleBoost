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

import random
import string

import torch

from teleboost.engines.teletron import get_args


class FakeDataset:
    def __init__(self, *args, **kwargs) -> None:
        self.args = get_args()
        self.dst_num_frames = self.args.num_frames
        self.dst_size = tuple(self.args.video_resolution)

    def __len__(self):
        return 10000000

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        random_data = {}
        random_data["struct_prompt"] = "".join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["short_prompt"] = "".join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["dense_prompt"] = "".join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["images"] = torch.randn((self.dst_num_frames, 3, self.dst_size[1], self.dst_size[0]))
        random_data["raw_first_image"] = torch.randn((1, 3, self.dst_size[1], self.dst_size[0]))
        return random_data


class FakeDPODataset:
    """Pre-encoded random data shaped for DPO training (chosen + rejected pair).

    Yields the post-encoder dict schema (text encoder + VAE + CLIP already
    applied), so the DPO consumer ranks can consume it directly without going
    through the encoder chain. Use this for DPO smoke tests or when you don't
    have a real DPO dataset on disk.

    All shapes match the production fixture (Wan 14B I2V) by default;
    pass ``num_frames`` / ``video_resolution`` via ``args`` (megatron
    arguments) to scale.

    Schema (per __getitem__):
        {
            "context":  Tensor[S_text=512, D_text=4096],   # T5 embedding
            "chosen":   {
                "latents":          Tensor[C=16, T, H, W],     # VAE-encoded video
                "img_clip_feature": Tensor[N_clip=257, 1280],  # CLIP image feature
                "img_emb_y":        Tensor[C=20, T, H, W],     # reference frame latent
            },
            "rejected": {  # same keys; same shapes (T == chosen by default)
                "latents":          Tensor[16, T, H, W],
                "img_clip_feature": Tensor[257, 1280],
                "img_emb_y":        Tensor[20, T, H, W],
            },
        }
    """

    def __init__(self, *args, **kwargs):
        self.args = get_args()
        # 4× temporal compression by VAE → latent T = ceil(num_frames / 4)
        self.lat_T = max(1, (self.args.num_frames + 3) // 4)
        # 8× spatial compression by VAE → (H/8, W/8)
        # video_resolution is (W, H) in pixels
        W_px, H_px = self.args.video_resolution
        self.lat_H = max(1, H_px // 8)
        self.lat_W = max(1, W_px // 8)

    def __len__(self):
        return 10_000_000

    def __getitem__(self, idx: int) -> dict:
        T, H, W = self.lat_T, self.lat_H, self.lat_W

        def _branch():
            return {
                "latents": torch.randn(16, T, H, W, dtype=torch.bfloat16),
                "img_clip_feature": torch.randn(257, 1280, dtype=torch.bfloat16),
                "img_emb_y": torch.randn(20, T, H, W, dtype=torch.bfloat16),
            }

        return {
            "context": torch.randn(512, 4096, dtype=torch.bfloat16),
            "chosen": _branch(),
            "rejected": _branch(),
        }
