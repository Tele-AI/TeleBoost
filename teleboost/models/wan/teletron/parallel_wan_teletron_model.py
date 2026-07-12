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
from typing import Optional

import torch
import torch.nn as nn

from teleboost.engines.teletron import get_args, set_config
from teleboost.engines.teletron.context_parallel import ContextParallelMixin
from teleboost.engines.teletron.tensor_parallel import TensorParallelMixin
from teleboost.engines.teletron.transformer import TransformerGeneralMixin

from teleboost.models.wan.attention.backend import wan_dense_attention
from .wan_teletron_model import DiTBlock, WanTeletronModel, sinusoidal_embedding_1d


class ParallelWanTeletronDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def __init__(
        self,
        config,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        **kwargs,
    ):
        DiTBlock.__init__(self, has_image_input, dim, num_heads, ffn_dim, eps, **kwargs)
        # from ContextParallelMixin
        self.enable_context_parallel(
            self.self_attn.attn,
            attention_fn=wan_dense_attention,
        )

        # from Tensor ParallelMixin
        self.enable_ffn_tensor_parallel(self.ffn, config)
        self.enable_self_attn_tensor_parallel(self.self_attn, config)
        self.enable_cross_attn_tensor_parallel(self.cross_attn, config)

    def forward(self, x, context, t_mod, freqs, cp_origin_length):
        # forward_attn reads this off self to know the OUTER seq pre-pad length.
        self._cp_origin_length = cp_origin_length

        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        modulation = modulation + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        normed_x1 = self.norm1(x)
        modulated_x1 = self.modulate_with_cp_grad_reduce(normed_x1, shift_msa, scale_msa)
        attn_output = self.self_attn(modulated_x1, freqs)
        gated_x1 = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)

        normed_x3 = self.norm3(gated_x1)
        cross_attn_output = self.cross_attn(normed_x3, context)

        x = gated_x1 + cross_attn_output

        normed_x2 = self.norm2(x)
        modulated_x2 = self.modulate_with_cp_grad_reduce(normed_x2, shift_mlp, scale_mlp)
        ffn_output = self.ffn(modulated_x2)
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, ffn_output)

        return x


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, device="cuda"):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class ParallelWanTeletronModel(ContextParallelMixin, TensorParallelMixin, TransformerGeneralMixin, WanTeletronModel):
    def __init__(self, config):
        dit_model_config = set_config().get("model_config", None).get("dit", None).get("config", None)
        WanTeletronModel.__init__(self, **dit_model_config)
        self.config = config

        self.blocks = nn.ModuleList([ParallelWanTeletronDitBlock(self.config, self.has_image_input, self.dim, self.num_heads, self.ffn_dim, self.eps) for _ in range(self.num_layers)])

        args = get_args()

        self.enable_activation_optimizations(self.blocks, enable_checkpointing=True, enable_offloading=args.activation_offload)
        # from ContextParallelMixin
        self.register_cp_grad_reduce_hook()

    def register_cp_grad_reduce_hook(self):
        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook

        for name, param in self.named_parameters():
            if name.startswith("patch_emb") or name.startswith("time") or name.startswith("head") or "modulation" in name:
                continue

            param.register_hook(self.cp_grad_reduce)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cn_images=None,
        **kwargs,
    ):
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep)
        t = self.time_emb(t_emb)
        t_mod = self.time_proj(t)
        t_mod = t_mod.unflatten(1, (6, self.dim))

        context_emb = self.text_emb(context)

        if y is not None:
            # has image input
            x = torch.cat([x, y], dim=1)

        if self.has_image_input:
            if clip_feature is not None:
                clip_embedding = self.img_emb(clip_feature)
                context_emb = torch.cat([clip_embedding, context_emb], dim=1)

        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)

        x, (f, h, w) = self.patchify(x)

        head_dim = self.dim // self.num_heads
        freq_f = precompute_freqs_cis(head_dim - 2 * (head_dim // 3), f).view(f, 1, 1, -1).expand(f, h, w, -1)
        freq_h = precompute_freqs_cis(head_dim // 3, h).view(1, h, 1, -1).expand(f, h, w, -1)
        freq_w = precompute_freqs_cis(head_dim // 3, w).view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([freq_f, freq_h, freq_w], dim=-1)
        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        x, x_origin_length = self.split_input(x, dim=1)
        freqs, _ = self.split_input(freqs, dim=0)
        for block in self.blocks:
            x = block(x, context_emb, t_mod, freqs, cp_origin_length=x_origin_length)
        x = self.gather_output(x, dim=1, origin_length=x_origin_length)

        # Now x is in full shape, just do regular forward
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def sharded_state_dict(self):
        return self.state_dict()
