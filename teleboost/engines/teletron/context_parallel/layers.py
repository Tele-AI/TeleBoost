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
import torch
from megatron.core import mpu


def gate_with_cp_grad_reduce(x, gate, residual):
    return GateWithGradReduce.apply(x, gate, residual)


class GateWithGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, residual):
        ctx.save_for_backward(gate, residual)
        return x + gate * residual

    @staticmethod
    def backward(ctx, x_grad):
        gate, residual = ctx.saved_tensors
        r_grad = x_grad * gate
        gate_grad = torch.sum((x_grad * residual), dim=1, keepdim=True)
        torch.distributed.all_reduce(gate_grad, group=mpu.get_context_parallel_group())
        return x_grad, gate_grad, r_grad


class ModulateWithCPGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale):
        ctx.save_for_backward(x, scale)
        return x * (1 + scale) + shift

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        x_grad = grad_output * (1 + scale)
        scale_grad = torch.sum((grad_output * x), dim=1, keepdim=True)
        torch.distributed.all_reduce(scale_grad, group=mpu.get_context_parallel_group())
        shift_grad = torch.sum(grad_output, dim=1, keepdim=True)
        torch.distributed.all_reduce(shift_grad, group=mpu.get_context_parallel_group())
        return x_grad, shift_grad, scale_grad


def modulate_with_cp_grad_reduce(x, shift, scale):
    return ModulateWithCPGradReduce.apply(x, shift, scale)
