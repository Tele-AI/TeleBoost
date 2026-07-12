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
from functools import partial

import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import checkpoint

from teleboost.engines.teletron.transformer.memory_manager import get_memory_manager


class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
            if not pin_memory:
                manager = get_memory_manager()
                tensor_buffer = manager.get_buffer(tensor.size(), tensor.dtype)
                tensor_buffer.copy_(tensor, non_blocking=False)
                return (tensor.device, tensor_buffer)
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            manager = get_memory_manager()
            reloaded_t = torch.empty(tensor.size(), dtype=tensor.dtype, device=device)
            reloaded_t.copy_(tensor, non_blocking=pin_memory)
            manager.return_buffer(tensor)
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


def offload(forward_func):
    def wrapped_forward(self, *args, **kwargs):
        with save_on_cpu():
            return forward_func(self, *args, **kwargs)

    return wrapped_forward


class TransformerGeneralMixin:
    """
    A mixin class providing advanced memory optimizations.
    It enables activation recomputation and offloading by wrapping modules, avoiding direct method modification.
    """

    def enable_activation_optimizations(self, blocks: nn.ModuleList, enable_checkpointing: bool = True, enable_offloading: bool = False):
        """
        Unified entry point for enabling activation optimizations.

        Args:
            blocks (nn.ModuleList): ModuleList containing all Transformer layers.
            enable_checkpointing (bool): whether to enable activation recomputation.
            enable_offloading (bool): whether to enable activation offloading.
        """
        # Fetch detailed parameters from the config
        from teleboost.engines.teletron import get_args

        args = get_args()

        # Checkpointing-related config
        recompute_method = getattr(args, "recompute_method", "block")
        recompute_num_layers = getattr(args, "recompute_num_layers", 0) if enable_checkpointing else 0

        for i in range(len(blocks)):
            module_to_wrap = blocks[i]
            should_checkpoint_this_layer = False
            if enable_checkpointing and recompute_num_layers > 0:
                if recompute_method == "block":
                    if i < recompute_num_layers:
                        should_checkpoint_this_layer = True
                elif recompute_method == "uniform":
                    should_checkpoint_this_layer = True
                else:
                    raise ValueError(f"Invalid activation recompute method {recompute_method}.")
            if should_checkpoint_this_layer:
                if enable_offloading:
                    module_to_wrap.forward = partial(checkpoint, module_to_wrap.forward, use_reentrant=False)
                    module_to_wrap.forward = offload(module_to_wrap.forward)

                else:
                    module_to_wrap.forward = partial(checkpoint, module_to_wrap.forward, use_reentrant=False)

            if module_to_wrap is not blocks[i]:
                blocks[i] = module_to_wrap

    def enable_activation_checkpointing(self, blocks):
        """Convenience: checkpoint-only activation optimization."""
        self.enable_activation_optimizations(blocks, enable_checkpointing=True, enable_offloading=False)

    def enable_activation_offload(self, blocks):
        """Convenience: checkpoint + CPU offload of activations."""
        self.enable_activation_optimizations(blocks, enable_checkpointing=True, enable_offloading=True)

    def set_input_tensor(self, x):
        return None
