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
import torch
from torch import nn


class Wan22DualModel(nn.Module):
    def __init__(self, low_noise_model: nn.Module, high_noise_model: nn.Module, boundary: float = 0.9):
        super().__init__()
        self.low_noise_model = low_noise_model
        self.high_noise_model = high_noise_model
        self.boundary = boundary
        self.config = getattr(low_noise_model, "config", None)
        self._no_split_modules = getattr(low_noise_model, "_no_split_modules", None)

    def _normalize_timestep(self, t):
        if t is None:
            return None
        if torch.is_tensor(t):
            t_val = t.detach().flatten()[0].float().item()
        else:
            t_val = float(t)
        if t_val > 1.0:
            t_val = t_val / 1000.0
        return t_val

    def _select_model(self, t):
        t_val = self._normalize_timestep(t)
        if t_val is None:
            return self.low_noise_model
        if t_val >= self.boundary:
            return self.high_noise_model
        return self.low_noise_model

    def forward(self, *args, **kwargs):
        t = kwargs.get("t")
        if t is None and len(args) > 1:
            t = args[1]
        model = self._select_model(t)
        return model(*args, **kwargs)

    def can_generate(self) -> bool:
        """Required by upstream verl FSDPCheckpointManager.save_checkpoint.

        Upstream's checkpoint manager calls `unwrap_model.can_generate()` to decide whether
        to save an HF `generation_config.json` next to the weights. Wan diffusion models are
        not HF generation models, so always return False — this skips the generation_config
        save path. Pre-X3 worked around this by commenting out the call inside the in-tree
        verl fork; this is the same fix moved to the model side.
        """
        return False
