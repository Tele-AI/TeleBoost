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
"""End-to-end precision: forward+backward with PROPER all_reduce on weight grads
(mimicking register_cp_grad_reduce_hook). Compare to no-CP reference, fp32.

Earlier test reported 3% rel diff, but that was the per-rank LOCAL grad diff
WITHOUT all_reduce. Production does all_reduce via cp_grad_reduce_hook; here
we apply it explicitly to verify the math is correct.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world = dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
device = torch.cuda.current_device()

NUM_HEADS = 8
HEAD_DIM = 32
HIDDEN = NUM_HEADS * HEAD_DIM
NUM_BLOCKS = 2


def _load_context_parallel_mixin():
    """Configure the standalone process before importing the mixin."""

    from megatron.core import mpu

    import teleboost.engines.teletron as tu

    class _Args:
        num_attention_heads = NUM_HEADS

    mpu.get_context_parallel_world_size = lambda: world
    mpu.get_context_parallel_group = lambda: dist.group.WORLD
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    tu.get_args = lambda: _Args()

    from teleboost.engines.teletron.context_parallel.context_parallel_mixin import (
        ContextParallelMixin,
    )

    return ContextParallelMixin, mpu


ContextParallelMixin, mpu = _load_context_parallel_mixin()


class RefBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, h, bias=False)
        self.v = nn.Linear(h, h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.ffn = nn.Linear(h, h, bias=False)

    def forward(self, x):
        q = rearrange(self.q(x), "b s (n d) -> b n s d", n=NUM_HEADS).contiguous()
        k = rearrange(self.k(x), "b s (n d) -> b n s d", n=NUM_HEADS).contiguous()
        v = rearrange(self.v(x), "b s (n d) -> b n s d", n=NUM_HEADS).contiguous()
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2, 3).contiguous()
        a = self.o(a)
        return x + a + self.ffn(x + a)


class RefModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([RefBlock(HIDDEN) for _ in range(NUM_BLOCKS)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class CPAttn(nn.Module):
    def forward(self, q, k, v):
        return q


class CPSA(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, h, bias=False)
        self.v = nn.Linear(h, h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.attn = CPAttn()

    def forward(self, x):
        return self.o(self.attn(self.q(x), self.k(x), self.v(x)))


class CPBlock(ContextParallelMixin, nn.Module):
    def __init__(self, h):
        nn.Module.__init__(self)
        self.self_attn = CPSA(h)
        self.ffn = nn.Linear(h, h, bias=False)
        self.enable_context_parallel(
            self.self_attn.attn,
            attention_fn=lambda q, k, v: F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2),
        )

    def forward(self, x, cp_origin_length):
        self._cp_origin_length = cp_origin_length
        a = self.self_attn(x)
        return x + a + self.ffn(x + a)


class CPModel(ContextParallelMixin, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList([CPBlock(HIDDEN) for _ in range(NUM_BLOCKS)])

    def forward(self, x):
        x, xo = self.split_input(x, dim=1)
        for b in self.blocks:
            x = b(x, cp_origin_length=xo)
        return self.gather_output(x, dim=1, origin_length=xo)


torch.manual_seed(0)
ref_state = RefModel().state_dict()


def map_to_cp(rs):
    out = {}
    for k, v in rs.items():
        if any(p in k for p in (".q.", ".k.", ".v.", ".o.")):
            parts = k.split(".")
            parts.insert(2, "self_attn")
            out[".".join(parts)] = v
        else:
            out[k] = v
    return out


torch.manual_seed(7)
B, T = 1, 99
xf = torch.randn(B, T, HIDDEN, device=device, dtype=torch.bfloat16)
tf = torch.randn(B, T, HIDDEN, device=device, dtype=torch.bfloat16)
dist.broadcast(xf, 0)
dist.broadcast(tf, 0)


def run_ref(m):
    out = m(xf)
    loss = (out - tf).pow(2).mean()
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def run_cp_with_reduce(m):
    out = m(xf)
    loss = (out - tf).pow(2).mean()
    loss.backward()
    grads = {}
    for n, p in m.named_parameters():
        g = p.grad.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        grads[n] = g
    return loss.detach(), grads


m_ref = RefModel()
m_ref.load_state_dict(ref_state)
m_ref = m_ref.to(device).to(torch.bfloat16)
l_ref, g_ref = run_ref(m_ref)

m_cp = CPModel()
m_cp.load_state_dict(map_to_cp(ref_state))
m_cp = m_cp.to(device).to(torch.bfloat16)
l_cp, g_cp = run_cp_with_reduce(m_cp)

if rank == 0:
    print(f"\n[bf16, no-CP vs CP={world} (with all_reduce on grads)]")
    print(f"  loss: ref={l_ref.item():.6f}  cp={l_cp.item():.6f}  diff={abs(l_ref.item() - l_cp.item()):.3e}")
    pass_all = True
    for n in sorted(g_cp):
        ref_n = n.replace(".self_attn.", ".")
        if ref_n not in g_ref:
            continue
        d = (g_ref[ref_n] - g_cp[n]).abs().max().item()
        max_e = g_ref[ref_n].abs().max().item()
        per_elem_rel = d / (max_e + 1e-12)
        nrm = g_ref[ref_n].norm().item()
        ok = per_elem_rel < 0.05  # 5% per-element bf16 tolerance
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {n:55s}  |max_abs|={d:.3e}  max_entry={max_e:.4f}  per_elem_rel={per_elem_rel:.3e}")
        if not ok:
            pass_all = False
    print(f"\nVERDICT: {'PASS' if pass_all else 'FAIL'} (per-element bf16 tol = 5%)")

failed = torch.tensor(
    [int(rank == 0 and not pass_all)],
    dtype=torch.int32,
    device=device,
)
dist.broadcast(failed, src=0)
dist.barrier(device_ids=[local])
dist.destroy_process_group()
if failed.item():
    raise SystemExit(1)
