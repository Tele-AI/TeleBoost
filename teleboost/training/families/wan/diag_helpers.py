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
"""Small tensor-to-metric reduction helpers shared by Wan actor diagnostics.

Kept in a neutral module (not the actor) so both the actor and the composable
diagnostic terms can import them without a circular dependency.
"""

from __future__ import annotations

import torch


def to_metric_value(value):
    if torch.is_tensor(value):
        return value.detach().float().item()
    return float(value)


def std_or_zero(value: torch.Tensor) -> torch.Tensor:
    if value.numel() <= 1:
        return torch.zeros((), device=value.device, dtype=torch.float32)
    return value.float().std(unbiased=False)


def fraction(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return torch.zeros((), device=mask.device, dtype=torch.float32)
    return mask.float().mean()


def conditional_fraction(mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return torch.zeros((), device=mask.device, dtype=torch.float32)
    denom = condition.float().sum()
    if denom <= 0:
        return torch.zeros((), device=mask.device, dtype=torch.float32)
    return (mask & condition).float().sum() / denom


def broadcast_like_leading_dim(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return torch.broadcast_to(value, target.shape)


def timestep_bucket_label(timestep, bucket_count: int) -> str:
    if bucket_count <= 1:
        return "b00"
    if torch.is_tensor(timestep):
        t_value = timestep.detach().float().mean().item()
    else:
        t_value = float(timestep)
    if t_value > 1.0:
        t_value = t_value / 1000.0
    t_value = min(max(t_value, 0.0), 1.0)
    # Wan/Flow timesteps are high-noise near 1 and low-noise near 0.
    bucket = int((1.0 - t_value) * bucket_count)
    bucket = min(max(bucket, 0), bucket_count - 1)
    return f"b{bucket:02d}"
