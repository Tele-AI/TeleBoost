# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025 TeleAI-infra Team.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
# TeleAI modifications are licensed under Apache-2.0; see LICENSE at the root.

import torch
import torch.nn as nn
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from .mappings import divide, tele_rmsnorm_cuisine

__all__ = [
    "TeleColumnParallelLinear",
    "TeleRowParallelLinear",
    "TeleParallelRMSNorm",
]


class TeleColumnParallelLinear(ColumnParallelLinear):
    def forward(self, x):
        output, bias = super().forward(x)
        return output + bias if bias is not None else output


class TeleRowParallelLinear(RowParallelLinear):
    def forward(self, x):
        output, bias = super().forward(x)
        return output + bias if bias is not None else output


class TeleParallelRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        world_size = get_tensor_model_parallel_world_size()
        self.dim_per_partition = divide(dim, world_size)
        self.weight = nn.Parameter(torch.ones(self.dim_per_partition))

    def forward(self, x):
        return tele_rmsnorm_cuisine(x, self.weight, self.eps)


# Baseline
