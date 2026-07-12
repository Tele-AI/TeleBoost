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
import math

import torch
import torch.nn as nn
from einops import rearrange
from megatron.core import mpu

from teleboost.engines.teletron import get_args

from .layers import GateWithGradReduce, ModulateWithCPGradReduce
from .mappings import SeqAllToAll, gather_forward_split_backward, split_forward_gather_backward


class ContextParallelMixin:
    """
    Stateless CP helpers.

    pad_for_context_parallel returns ``(padded, origin_length)``; the matching
    remove_pad takes ``origin_length`` explicitly. split_input/gather_output
    follow the same contract.

    forward_attn is monkey-patched onto an attention module via
    enable_context_parallel(); the calling block must set
    ``self._cp_origin_length`` (the OUTER seq's pre-pad length, returned by
    its own split_input call) before invoking the attention chain. This lets
    forward_attn keep its (q, k, v) signature so we don't have to plumb
    origin_length through SelfAttention layers we don't own.
    """

    @staticmethod
    def cp_grad_reduce(grad):
        with torch.no_grad():
            reduced_grad = grad.contiguous()
            torch.distributed.all_reduce(reduced_grad, group=mpu.get_context_parallel_group())
        return reduced_grad

    def enable_context_parallel(self, attn_module: nn.Module, *, attention_fn):
        """Install CP attention using a model-owned dense-attention kernel.

        TeleTron owns sequence sharding, not a particular model family's
        attention implementation.  Requiring the concrete model to inject the
        kernel keeps this generic layer independent of ``teleboost.models``.
        """
        if not callable(attention_fn):
            raise TypeError("attention_fn must be callable")
        self._cp_attention_fn = attention_fn
        attn_module.forward = self.forward_attn

    @staticmethod
    def pad_for_context_parallel(tensor, dim):
        cp_size = mpu.get_context_parallel_world_size()
        origin_length = tensor.shape[dim]
        padded_length = math.ceil(origin_length / cp_size) * cp_size
        pad_size = padded_length - origin_length
        if pad_size <= 0:
            return tensor, origin_length
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * dim + 1)] = pad_size
        return torch.nn.functional.pad(tensor, pad), origin_length

    @staticmethod
    def remove_pad_for_context_parallel(tensor, dim, origin_length):
        return tensor.narrow(dim, 0, origin_length)

    def split_input(self, x, dim):
        cp_group = mpu.get_context_parallel_group()
        x, origin_length = self.pad_for_context_parallel(x, dim)
        x = split_forward_gather_backward(x, cp_group, dim=dim, grad_scale="none")
        return x, origin_length

    def gather_output(self, output, dim, origin_length):
        cp_group = mpu.get_context_parallel_group()
        output = gather_forward_split_backward(output, cp_group, dim=dim, grad_scale="none")
        return self.remove_pad_for_context_parallel(output, dim, origin_length)

    def forward_attn(self, q, k, v):
        # The block that owns this attention monkey-patched our forward_attn here
        # and is responsible for setting _cp_origin_length on itself before
        # invoking attention. Reading missing => programmer error, fail loud.
        origin_length = self._cp_origin_length
        cp_group = mpu.get_context_parallel_group()
        args = get_args()
        num_heads = args.num_attention_heads // mpu.get_tensor_model_parallel_world_size()

        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)

        if mpu.get_context_parallel_world_size() > 1:
            q = SeqAllToAll.apply(cp_group, q, 2, 1)
            k = SeqAllToAll.apply(cp_group, k, 2, 1)
            v = SeqAllToAll.apply(cp_group, v, 2, 1)
            q, k, v = (self.remove_pad_for_context_parallel(t, 1, origin_length) for t in (q, k, v))

        try:
            attention_fn = self._cp_attention_fn
        except AttributeError as exc:
            raise RuntimeError("ContextParallelMixin requires a model-owned attention_fn; call enable_context_parallel(..., attention_fn=...) first") from exc
        x = attention_fn(q, k, v).transpose(1, 2).contiguous()

        if mpu.get_context_parallel_world_size() > 1:
            x, _ = self.pad_for_context_parallel(x, 2)
            x = SeqAllToAll.apply(cp_group, x, 2, 1)

        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        return x

    def gate_with_cp_grad_reduce(self, x, gate, residual):
        return GateWithGradReduce.apply(x, gate, residual)

    def modulate_with_cp_grad_reduce(self, x, shift, scale):
        return ModulateWithCPGradReduce.apply(x, shift, scale)
