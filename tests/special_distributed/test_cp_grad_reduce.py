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
"""Regression test for the wan modulation grad path under Ulysses SP.

Locks in the fix from commit b0cecf7a: modulation params must NOT be
double-reduced. Pre-fix, sp=N>1 yielded 0.5*N*G_full (off by 2x at sp=4,
4x at sp=8); post-fix yields G_full bit-exact in fp32.

Run on 1/2/4/8 GPUs:
    torchrun --nproc_per_node=N tests/special_distributed/test_cp_grad_reduce.py

Or with bf16:
    DTYPE=bf16 torchrun --nproc_per_node=4 tests/...
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn

# Apply TeleBoost patches before importing the symbols they inject into
# ``verl.utils.ulysses``.  Reversing this order fails during module import on
# the pinned upstream verl release, before the patch can run.
from teleboost import apply_runtime_patches


def _load_ulysses_symbols():
    apply_runtime_patches()

    from verl.utils.ulysses import (
        gate_with_cp_grad_reduce,
        modulate_with_cp_grad_reduce,
        register_cp_grad_reduce_hook,
        set_ulysses_sequence_parallel_group,
    )

    return (
        gate_with_cp_grad_reduce,
        modulate_with_cp_grad_reduce,
        register_cp_grad_reduce_hook,
        set_ulysses_sequence_parallel_group,
    )


(
    gate_with_cp_grad_reduce,
    modulate_with_cp_grad_reduce,
    register_cp_grad_reduce_hook,
    set_ulysses_sequence_parallel_group,
) = _load_ulysses_symbols()

DTYPE = torch.bfloat16 if os.environ.get("DTYPE") == "bf16" else torch.float32
H, S, SEED = 256, 64, 42


class _MiniBlock(nn.Module):
    """Mirrors the relevant subset of wan/modules/model.py:WanAttentionBlock:
    a `modulation` Parameter consumed via modulate/gate_with_cp_grad_reduce."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict({"0": nn.Module()})
        self.blocks["0"].modulation = nn.Parameter(torch.randn(1, 6, H, dtype=DTYPE))
        self.blocks["0"].weight_other = nn.Parameter(torch.randn(H, H, dtype=DTYPE))

    def forward(self, x):
        e = self.blocks["0"].modulation.chunk(6, dim=1)
        x = modulate_with_cp_grad_reduce(x, e[0], e[1])
        x = gate_with_cp_grad_reduce(x, e[2], x)
        return x @ self.blocks["0"].weight_other


def _make_block(device):
    torch.manual_seed(SEED)
    return _MiniBlock().to(device)


def _make_inputs():
    torch.manual_seed(SEED + 1)
    return torch.randn(1, S, H, dtype=DTYPE), torch.randn(1, S, H, dtype=DTYPE)


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    set_ulysses_sequence_parallel_group(dist.group.WORLD)

    # Single-GPU analytical reference (rank 0 only): no dist hooks, no slicing.
    if rank == 0:
        m_ref = _make_block(device)
        xf, gf = _make_inputs()
        xf, gf = xf.to(device), gf.to(device)
        e = m_ref.blocks["0"].modulation.chunk(6, dim=1)
        y = xf * (1 + e[1]) + e[0]
        y = y + e[2] * y
        y = y @ m_ref.blocks["0"].weight_other
        (y * gf).sum().backward()
        g_ref = m_ref.blocks["0"].modulation.grad.float().norm().item()
    else:
        g_ref = 0.0

    # SP path: real verl forward + register_cp_grad_reduce_hook (post-fix).
    m_sp = _make_block(device)
    xf, gf = _make_inputs()
    register_cp_grad_reduce_hook(m_sp)
    sl = S // world
    slc = slice(rank * sl, (rank + 1) * sl)
    x_loc = xf[:, slc].to(device).contiguous()
    go_loc = gf[:, slc].to(device).contiguous()
    y = m_sp(x_loc)
    (y * go_loc).sum().backward()

    if rank == 0:
        g_sp = m_sp.blocks["0"].modulation.grad.float().norm().item()
        rel_err = abs(g_sp - g_ref) / max(g_ref, 1e-12)
        # fp32 must be bit-exact; bf16 has ~1e-3 reduce-rounding floor.
        tol = 1e-6 if DTYPE == torch.float32 else 5e-3
        ok = rel_err < tol
        print(f"DTYPE={'bf16' if DTYPE == torch.bfloat16 else 'fp32'}  sp={world}  g_sp={g_sp:.6f}  g_ref={g_ref:.6f}  rel_err={rel_err:.3e}  {'PASS' if ok else 'FAIL'}")
        assert ok, f"modulation grad rel_err={rel_err:.3e} exceeds tol={tol:.0e}"

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
