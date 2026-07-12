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
"""TeleBoost's Ulysses sequence-parallel layer, applied onto ``verl.utils.ulysses``.

One target module, one file: the implementations (Sections 0a/0b)
AND their delivery. Sections:

1. **Model wrapping** — input-slicing / head-gather wrappers installed on the
   Wan transformer at build time (``apply_wan_ulysses_patches``, called per
   model by the recipes worker).
2. **API re-injection** — the pre-X3 in-tree verl fork exposed Wan-specific
   helpers on ``verl.utils.ulysses``; pip-installed upstream lacks them, so
   ``apply_api()`` injects TeleBoost's implementations for existing import
   sites.
3. **Grad-reduce fix** — upstream's ``register_cp_grad_reduce_hook`` double-
   reduces wan modulation grads; ``apply_cp_grad_fix()`` installs the fixed
   hook.

Sections 2-3 are startup-installed (``teleboost.patches.apply``); section 1 is
runtime, model-scoped.
"""

from __future__ import annotations
from typing import Any, Optional
import torch
import torch.distributed as dist
from torch import Tensor


# =========================================================================
# Section 0a: diffusion SP helpers (implementation — the pre-X3 fork's
# Wan-shape ulysses layer: pad/target module state, slice/gather ops)
# =========================================================================

"""Diffusion-aware Ulysses sequence parallel helpers.

Pre-X3 lived in the in-tree verl/utils/ulysses.py fork. After X3 dropped that fork,
upstream verl 0.4.0's ulysses.py only ships the LM-shape helpers (`gather_seq_scatter_heads`,
`slice_input_tensor`, etc.). The Wan diffusion path needs:

  - module-level state for the current sequence pad/target sizes (set during input
    slicing inside the patched block.forward, read during head-gather inside
    Head.forward — same iteration of the same model);
  - a `DiffusionGather` autograd Function that's like upstream `Gather` but keeps
    the local batch-dim shape so split/cat round-trips work for image latents;
  - `split_forward_gather_backward` / `gather_forward_split_backward` round-trippers
    used by the Wan block.forward / Head.forward monkey-patches in
    `teleboost.patches.ulysses`.

Importing from `verl.utils.ulysses` for the upstream-supplied helpers
(`get_ulysses_sequence_parallel_group`, `_pad_tensor`, `_unpad_tensor`,
`all_gather_tensor`).
"""


def _verl_ulysses():
    """Lazy accessor for upstream helpers (module level must stay verl-free)."""
    import verl.utils.ulysses as _u

    return _u


# ----- module-level state ----------------------------------------------------
# Target / pad sizes are set by the input-slicing wrapper at the start of each
# transformer block.forward and read by the head-gather wrapper at the end of
# Head.forward (same iteration, same model). Storing them as module-level
# globals matches the pre-X3 fork's contract; callers don't pass them through.
_TARGET_SIZE: Optional[int] = None
_PAD_SIZE: Optional[int] = None


def set_target_len(target_size: int) -> None:
    global _TARGET_SIZE
    _TARGET_SIZE = target_size


def get_target_len() -> Optional[int]:
    return _TARGET_SIZE


def set_pad_size(pad_size: int) -> None:
    global _PAD_SIZE
    _PAD_SIZE = pad_size


def get_pad_size() -> Optional[int]:
    return _PAD_SIZE


# ----- autograd Functions ----------------------------------------------------
class DiffusionGather(torch.autograd.Function):
    """All-gather along `gather_dim`, splitting the batch dim back out on the way in.

    Variant of upstream `verl.utils.ulysses.Gather` for image-latent shape conventions:
    the gathered output has the same leading batch size as the local input
    (concat happens along `gather_dim`, not along dim=0).
    """

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler

        ctx.sp_world_size = dist.get_world_size(group=group)
        ctx.sp_rank = dist.get_rank(group=group)

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        ctx.part_size = local_shape[gather_dim]

        output = _verl_ulysses().all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Forward: all-gather across `process_group` along `dim`. Backward: split + grad scale."""

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """Forward: split along `dim`, keep this rank's chunk. Backward: gather + grad scale."""

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.split_sizes = split_sizes
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None


# ----- low-level split/gather (support unaligned shapes) ---------------------
def _split(
    input_: torch.Tensor,
    pg: dist.ProcessGroup,
    dim: int = -1,
    split_sizes: Optional[list[int]] = None,
) -> torch.Tensor:
    assert split_sizes is None or isinstance(split_sizes, list)

    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size
        # Distribute remainder to first `remainder` ranks (matches upstream LM split).
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)
    rank = dist.get_rank(pg)
    return tensor_list[rank].contiguous()


def _gather(
    input_: torch.Tensor,
    pg: dist.ProcessGroup,
    dim: int = -1,
    gather_sizes: Optional[list[int]] = None,
) -> torch.Tensor:
    assert gather_sizes is None or isinstance(gather_sizes, list)

    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()

    if gather_sizes:
        tensor_shape_base = input_.size()
        tensor_list = []
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

    assert input_.device.type == "cuda"
    dist.all_gather(tensor_list, input_, group=pg)
    return torch.cat(tensor_list, dim=dim).contiguous()


# ----- public round-trip helpers --------------------------------------------
def split_forward_gather_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    split_sizes: Optional[list[int]] = None,
    grad_scale: str = "down",
) -> torch.Tensor:
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[list[int]] = None,
    grad_scale: str = "up",
) -> torch.Tensor:
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)


# ----- diffusion-shape gather/slice -----------------------------------------
def diffusion_gather_outpus_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: Optional[int] = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """All-gather along `gather_dim`, then strip `padding_size` rows from `unpad_dim`.

    Used by the Wan Head.forward monkey-patch to undo the pre-block input slicing
    (shape now back to full sequence), then drop the padding the slicer added.
    """
    group = _verl_ulysses().get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = DiffusionGather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding_size must be int"
        if padding_size == 0:
            return x
        x = _verl_ulysses()._unpad_tensor(x, unpad_dim, padding_size)
    return x


def diffusion_slice_input_tensor_pad(
    x: Tensor,
    dim: int,
    padding: bool = False,
    grad_scaler: bool = True,
) -> Tensor:
    """Pad `x` on `dim` to be SP-divisible, then keep this rank's slice.

    Records the pre-slicing length and pad size into module-level globals so the
    matching head-gather wrapper can reverse the operation.
    """
    group = _verl_ulysses().get_ulysses_sequence_parallel_group()
    sp_world_size = dist.get_world_size(group)
    dim_size = x.size(dim)
    padding_size = (sp_world_size - dim_size % sp_world_size) % sp_world_size
    set_target_len(dim_size)
    set_pad_size(padding_size)
    if padding and padding_size > 0:
        x = _verl_ulysses()._pad_tensor(x, dim, padding_size)
    return split_forward_gather_backward(x, group, dim=dim, grad_scale="none")


# =========================================================================
# Section 0b: wan-specific CP autograd Functions (implementation)
# =========================================================================

"""TeleBoost autograd Functions for cp-aware modulation/gate inside wan blocks.

Upstream verl@v0.4.0 has the Ulysses sequence-parallel scaffolding but not
these wan-specific autograd Functions; the project added them. They live
here in teleboost so we can keep the upstream verl pin clean, and we patch
them onto verl.utils.ulysses at runtime via teleboost.patches.

NOTE: the backward of `ModulateWithCPGradReduce` and `GateWithGradReduce`
already SUM-allreduces the modulation/shift/scale grads inside the
sp_group; the cp-fix patch in teleboost.patches.ulysses accounts
for that by skipping modulation in `register_cp_grad_reduce_hook`.
"""


def _sp_group():
    return _verl_ulysses().get_ulysses_sequence_parallel_group()


class GateWithGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, residual):
        ctx.save_for_backward(gate, residual)
        return x + gate * residual

    @staticmethod
    def backward(ctx, x_grad):
        gate, residual = ctx.saved_tensors
        r_grad = x_grad * gate
        gate_grad = torch.sum(x_grad * residual, dim=1, keepdim=True)
        torch.distributed.all_reduce(gate_grad, group=_sp_group())
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
        scale_grad = torch.sum(grad_output * x, dim=1, keepdim=True)
        torch.distributed.all_reduce(scale_grad, group=_sp_group())
        shift_grad = torch.sum(grad_output, dim=1, keepdim=True)
        torch.distributed.all_reduce(shift_grad, group=_sp_group())
        return x_grad, shift_grad, scale_grad


def gate_with_cp_grad_reduce(x, gate, residual):
    return GateWithGradReduce.apply(x, gate, residual)


def modulate_with_cp_grad_reduce(x, shift, scale):
    return ModulateWithCPGradReduce.apply(x, shift, scale)


# =========================================================================
# Section 1: SP model wrapping (runtime, per-model)
# =========================================================================


# NOTE: verl imports stay INSIDE function bodies throughout this module —
# Dependency-light CPU utilities can import ``teleboost`` on verl-less
# checkouts, so module level must not touch verl.


def ulysses_self_flash_attn_forward(
    self,
    x: torch.Tensor,
    seq_lens,
    grid_sizes,
    freqs,  # will become mandatory in v4.46
    **kwargs,
):
    from verl.utils.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
        get_ulysses_sequence_parallel_world_size,
        validate_ulysses_config,
    )

    from verl.utils.ulysses import get_target_len
    from wan.modules.attention import attention
    from wan.modules.model import rope_apply

    # bsz, q_len, _ = x.size()  # q_len = seq_length / sp_size
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    f, h, w = grid_sizes[0, :]
    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        target = get_target_len()
        q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, unpadded_dim_size=target)
        k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, unpadded_dim_size=target)
        v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, unpadded_dim_size=target)

        q.size(1)  # full_q_len = seq_length
    else:
        pass

    attn_output = attention(q=rope_apply(q, grid_sizes, freqs), k=rope_apply(k, grid_sizes, freqs), v=v, k_lens=seq_lens, window_size=self.window_size)

    # attn_output = attn_output.transpose(1, 2).flatten(2, 3).contiguous()
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.flatten(2).contiguous()

    attn_output = self.o(attn_output)
    return attn_output


# ---------------------------------------------------------------------------
# Wan-specific Ulysses input-slicing / head-gather monkey patches.
# Pre-X3 lived inside the in-tree verl `apply_monkey_patch` for `model_type=="t2v"`.
# After X3 dropped that fork, we apply them directly to the Wan model from the
# recipe's `_build_model_optimizer` when `ulysses_sequence_parallel_size > 1`.
# ---------------------------------------------------------------------------
def patch_diffusion_for_ulysses_input_slicing(model) -> None:
    """Wrap `model.blocks[0].forward` to slice the input `x` along seq-dim across SP ranks.

    The pre-X3 contract: only the first block's forward is wrapped; subsequent
    blocks see the already-sliced tensor (the model passes `x` through the chain).
    The wrapper also stashes the original sequence length / pad size into module
    state so the matching head-gather wrapper can undo the operation.
    """
    from verl.utils.ulysses import (
        diffusion_slice_input_tensor_pad,
        get_ulysses_sequence_parallel_world_size,
    )

    def _wrap(original_forward):
        def wrapped(*args, **kwargs):
            x = kwargs.get("x")
            if x is not None and get_ulysses_sequence_parallel_world_size() > 1:
                kwargs["x"] = diffusion_slice_input_tensor_pad(x, dim=1, padding=True)
            return original_forward(*args, **kwargs)

        return wrapped

    try:
        model.blocks[0].forward = _wrap(model.blocks[0].forward)
        print(f"[teleboost] Patched {type(model).__name__}.blocks[0].forward for Ulysses SP input slicing.")
    except Exception as e:
        # This patch is only applied under SP>1, where a missing input slice
        # means silently-wrong training — never continue without it.
        raise RuntimeError(f"Failed to patch {type(model).__name__} for Ulysses SP input slicing (required for SP>1 correctness): {e}") from e


def patch_diffusion_for_ulysses_head_gather(module_class) -> None:
    """Wrap `Head.forward` to all-gather the seq dim back, undoing the input slicing.

    Patched at the class level so it applies to every Head instance the model creates.
    """
    from verl.utils.ulysses import (
        diffusion_gather_outpus_and_unpad,
        get_pad_size,
        get_ulysses_sequence_parallel_world_size,
    )

    def _wrap(original_forward):
        def wrapped(self, *args, **kwargs):
            x = kwargs.get("x")
            if x is not None and get_ulysses_sequence_parallel_world_size() > 1:
                pad_size = get_pad_size() or 0
                kwargs["x"] = diffusion_gather_outpus_and_unpad(x, gather_dim=1, unpad_dim=1, padding_size=pad_size)
            return original_forward(self, *args, **kwargs)

        return wrapped

    try:
        # Idempotency sentinel (P0 fix): the wan2.2 dual-model path calls
        # apply_wan_ulysses_patches twice (low + high model) and both share the
        # SAME class object `Head`, so a naive re-wrap stacks _wrap(_wrap(fwd)) and
        # the seq-dim all-gather runs TWICE under SP>1 (silent-wrong / shape-crash).
        # The old "benign because SP==1 short-circuits" reasoning was wrong: this
        # path is only entered when SP>1 (caller guards on it), so the short-circuit
        # never applies. Stamp the wrapper and skip if already wrapped.
        if getattr(module_class.forward, "_tb_head_wrapped", False):
            return
        wrapped = _wrap(module_class.forward)
        wrapped._tb_head_wrapped = True
        module_class.forward = wrapped
        print(f"[teleboost] Patched {module_class.__name__}.forward for Ulysses SP head gather.")
    except Exception as e:
        # Only entered under SP>1: without the gather the outputs stay sliced
        # and training is silently wrong — fail loudly instead.
        raise RuntimeError(f"Failed to patch {module_class.__name__} for Ulysses SP head gather (required for SP>1 correctness): {e}") from e


def apply_wan_ulysses_patches(model) -> None:
    """Install all three Wan-specific Ulysses patches on a (low or high) WanModel instance.

    Idempotent at the class level: `Head.forward` is guarded by a `_tb_head_wrapped`
    sentinel (see `patch_diffusion_for_ulysses_head_gather`), and
    `WanSelfAttention.forward` is a plain *fixed-function* reassignment (not a
    wrap-of-current), so re-running on the second (high) model does not stack
    wrappers. This matters because the wan2.2 dual-model path calls this twice on
    the shared classes.
    """

    from wan.modules.model import Head, WanSelfAttention

    patch_diffusion_for_ulysses_input_slicing(model)
    patch_diffusion_for_ulysses_head_gather(Head)
    WanSelfAttention.forward = ulysses_self_flash_attn_forward


# =========================================================================
# Section 2: pre-X3 ulysses API re-injection (startup install)
# =========================================================================

"""Inject diffusion-aware Ulysses helpers into `verl.utils.ulysses`.

Pre-X3's in-tree verl/utils/ulysses.py exposed Wan-specific helpers
(`set/get_target_len`, `set/get_pad_size`, `diffusion_gather_outpus_and_unpad`,
`diffusion_slice_input_tensor_pad`, etc.). After X3 dropped that fork, callers
that still write `from verl.utils.ulysses import get_target_len` (e.g.
`teleboost.patches.ulysses.ulysses_self_flash_attn_forward`) need those
symbols to exist on the upstream namespace.

Mirror the pre-X3 surface by attribute-injecting from
Section 0a onto `verl.utils.ulysses`. The functions
themselves live in teleboost; this patch is just the "make `import` work" shim.
"""


def apply_api() -> None:
    import verl.utils.ulysses as _u

    # Implementations live in Section 0a of this module.
    for name, value in [
        ("DiffusionGather", DiffusionGather),
        ("diffusion_gather_outpus_and_unpad", diffusion_gather_outpus_and_unpad),
        ("diffusion_slice_input_tensor_pad", diffusion_slice_input_tensor_pad),
        ("gather_forward_split_backward", gather_forward_split_backward),
        ("get_pad_size", get_pad_size),
        ("get_target_len", get_target_len),
        ("set_pad_size", set_pad_size),
        ("set_target_len", set_target_len),
        ("split_forward_gather_backward", split_forward_gather_backward),
    ]:
        if not hasattr(_u, name):
            setattr(_u, name, value)


# =========================================================================
# Section 3: modulation grad double-reduce fix (startup install)
# =========================================================================

"""Fix the modulation grad double-reduce bug under Ulysses sequence parallel.

Upstream `register_cp_grad_reduce_hook` matches every parameter with "blocks"
in its name and does a SUM all-reduce of the grad. For wan-style transformer
blocks, the modulation parameter (shift/scale/gate) has its grad already
SUM-allreduced inside `ModulateWithCPGradReduce.backward` /
`GateWithGradReduce.backward`. Letting the hook fire on it duplicates the
reduce, producing `0.5 * sp_size * G_full` instead of `G_full` once the
historical `mul_(0.5)` post-backward compensation is removed.

Fix: skip the modulation params in the hook. Mathematically equivalent to
"do not double-reduce". Verified bit-exact in fp32 and within the bf16 reduce
floor (~1.5e-4 at sp=8) by `tests/special_distributed/test_cp_grad_reduce.py`.
"""


def apply_cp_grad_fix() -> None:
    """Inject TeleBoost cp-aware autograd Functions + fixed register hook into
    verl.utils.ulysses. Wan transformer blocks then call them via
    `from verl.utils.ulysses import gate_with_cp_grad_reduce` etc.
    """
    import torch
    import torch.distributed as dist
    import verl.utils.ulysses as _u

    # 1. Inject the cp-aware autograd Function entry points (upstream-missing).
    # Implementations live in Section 0b of this module.
    for name, value in [
        ("GateWithGradReduce", GateWithGradReduce),
        ("ModulateWithCPGradReduce", ModulateWithCPGradReduce),
        ("gate_with_cp_grad_reduce", gate_with_cp_grad_reduce),
        ("modulate_with_cp_grad_reduce", modulate_with_cp_grad_reduce),
    ]:
        if not hasattr(_u, name):
            setattr(_u, name, value)

    # 2. Replace register_cp_grad_reduce_hook to skip modulation params (root-cause
    #    fix for the cp grad double-reduce bug).
    def register_cp_grad_reduce_hook(model):
        def _cp_grad_reduce(grad):
            with torch.no_grad():
                dist.all_reduce(
                    grad,
                    op=dist.ReduceOp.SUM,
                    group=_u.get_ulysses_sequence_parallel_group(),
                )
                return grad

        for name, param in model.named_parameters():
            # modulation params are already SUM-allreduced inside Modulate/Gate
            # WithCPGradReduce.backward; skipping here avoids double-reduce.
            if "blocks" in name and "modulation" not in name.lower():
                param.register_hook(_cp_grad_reduce)

    _u.register_cp_grad_reduce_hook = register_cp_grad_reduce_hook
