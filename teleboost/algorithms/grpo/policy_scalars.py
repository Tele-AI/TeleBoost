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
"""Policy-scalar safety boundary for diffusion-RL loss.

Patch A of the common diffusion-RL layer.  Every quantity that enters the
GRPO loss as a *per-sample* term — log-prob, old log-prob, advantage, the
transition std used as a weight — must be reduced to a clean ``[B]`` (or a
shape that matches the advantage) *before* it touches the loss.  This module
owns that reduction + the shape assertions, so no training script can
reintroduce the ``[B,1,1,1,1] * [B] -> [B,1,1,1,B]`` cross-sample broadcast
bug (the P0 class).

Two distinct reductions, because conflating them is itself a bug source:

* :func:`reduce_log_density` — for log-prob-like quantities that are a
  density over the latent dims.  The reduction (``mean`` vs ``sum`` over
  non-batch dims) is *load-bearing*: ``ratio = exp(new − old)`` is only
  correct if ``new`` and ``old`` used the **same** reduction.  So the
  reduction is an explicit argument and the rollout record must carry
  ``logprob_reduction`` (see :mod:`rollout_contract`) — never silently mean a
  log-prob that was summed at rollout time.

* :func:`extract_per_sample_constant` — for quantities that are *constant*
  across the latent dims (the SDE transition std: ``sigma`` is a per-sample
  scalar broadcast over the latent).  Reducing is exact, not lossy; we assert
  the non-batch dims really are uniform so a future change that makes them
  vary can't silently get collapsed.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "reduce_log_density",
    "extract_per_sample_constant",
    "assert_same_shape",
    "assert_per_sample_shape",
    "assert_broadcasts_into",
]


def reduce_log_density(
    x: torch.Tensor,
    *,
    reduction: str = "mean",
    name: str = "log_density",
) -> torch.Tensor:
    """Reduce a log-density tensor over all non-batch dims to ``[B]``.

    ``reduction`` MUST match the reduction used when the paired old log-prob
    was recorded at rollout time (carried as ``logprob_reduction`` in the
    rollout record) — otherwise the importance ratio is silently wrong.

    Already-``[B]`` input is returned unchanged (idempotent).
    """
    if x.ndim == 0:
        raise ValueError(f"{name}: expected at least a batch dim, got scalar")
    if x.ndim == 1:
        return x
    dims = tuple(range(1, x.ndim))
    if reduction == "mean":
        return x.mean(dim=dims)
    if reduction == "sum":
        return x.sum(dim=dims)
    raise ValueError(f"{name}: unknown reduction {reduction!r}; use mean|sum")


def extract_per_sample_constant(
    x: torch.Tensor,
    *,
    name: str = "per_sample_constant",
    rtol: float = 1e-4,
    atol: float = 1e-6,
    check_uniform: bool = True,
) -> torch.Tensor:
    """Collapse a per-sample-constant latent-shaped tensor to ``[B]``.

    For the SDE transition std (``[B,1,1,1,1]`` or broadcast to
    ``[B,C,T,H,W]``) the value is identical across every non-batch element by
    construction, so taking element 0 is exact.  With ``check_uniform`` we
    assert that invariant holds (``max−min`` per sample ≈ 0) so a future
    spatially-varying std can't be silently truncated to one element.
    """
    if x.ndim == 1:
        return x
    bsz = x.shape[0]
    flat = x.reshape(bsz, -1)
    if check_uniform and flat.shape[1] > 1:
        spread = (flat.max(dim=1).values - flat.min(dim=1).values).abs()
        scale = flat.abs().max(dim=1).values.clamp_min(atol)
        if not bool((spread <= atol + rtol * scale).all()):
            worst = float((spread / scale).max())
            raise ValueError(f"{name}: tensor is not per-sample-constant across non-batch dims (worst rel spread {worst:.3g}); it is NOT safe to collapse to [B]. Use reduce_log_density or handle the dense shape explicitly.")
    return flat[:, 0]


def assert_same_shape(*tensors: torch.Tensor, names: Optional[list] = None) -> None:
    """Assert every tensor shares the same shape (the anti-broadcast invariant)."""
    shapes = [tuple(t.shape) for t in tensors]
    if len(set(shapes)) != 1:
        labels = names or [f"arg{i}" for i in range(len(tensors))]
        detail = ", ".join(f"{n}={s}" for n, s in zip(labels, shapes, strict=False))
        raise AssertionError(f"policy tensors must share one shape: {detail}")


def assert_broadcasts_into(source: torch.Tensor, target: torch.Tensor, *, names=("source", "target")) -> None:
    """Assert ``source`` broadcasts INTO ``target`` without expanding its shape.

    The real anti-cross-broadcast invariant for ``advantage * ratio``: a
    per-sample scalar / ``[1]`` advantage legitimately broadcasts to a ``[N]``
    ratio (one sample's advantage weighting all its per-step ratios), and a
    dense advantage matches a dense ratio — both give a result the SAME shape as
    ``ratio``. The P0 bug is the *expanding* broadcast, e.g. std ``[B,1,1,1,1]``
    × adv ``[B]`` → ``[B,1,1,1,B]`` (a new trailing batch dim that cross-mixes
    samples). So the check is ``broadcast(source, target) == target.shape``,
    NOT ``source.shape == target.shape`` (which wrongly rejects ``[1]×[N]``).
    """
    try:
        bshape = tuple(torch.broadcast_shapes(source.shape, target.shape))
    except RuntimeError as e:
        raise AssertionError(f"{names[0]}{tuple(source.shape)} is not broadcast-compatible with {names[1]}{tuple(target.shape)}: {e}")
    if bshape != tuple(target.shape):
        raise AssertionError(f"{names[0]}{tuple(source.shape)} must broadcast INTO {names[1]}{tuple(target.shape)}, but the result {bshape} expands beyond it — this is the cross-sample broadcast bug (e.g. [B,1,1,1,1]×[B]).")


def assert_per_sample_shape(bsz: int, *tensors: torch.Tensor, names: Optional[list] = None) -> None:
    """Strict ``[B]`` boundary: every tensor must be exactly ``(bsz,)``.

    Use on the scalar (non-VIPO) policy path. The dense pixel-weighted path
    keeps its own ``advantage.shape == ratio.shape`` invariant instead.
    """
    labels = names or [f"arg{i}" for i in range(len(tensors))]
    for n, t in zip(labels, tensors, strict=False):
        if tuple(t.shape) != (bsz,):
            raise AssertionError(f"policy scalar {n!r} must be [B]=({bsz},) at the loss boundary, got {tuple(t.shape)}")
