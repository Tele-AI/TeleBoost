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
"""Single source of truth for the clipped GRPO policy loss.

Patch A of the common diffusion-RL layer.  No training script may hand-write
``-advantage * ratio`` / ``std * advantage * ratio`` anymore; they call
:func:`grpo_policy_loss`.  This is what guarantees, across any current or
future video model, that the loss boundary is consistent (shapes asserted,
nan/inf guarded) and that per-sample weighting can't silently cross-broadcast.

Bit-identity contract (for the Wan cutover): with ``timestep_weight=None`` and
``beta=0`` the returned loss is exactly

    torch.mean(torch.maximum(-adv * ratio, -adv * clamp(ratio, 1±clip)))

i.e. byte-for-byte the previous inline ``dp_actor`` computation — same ops,
same order.  ``timestep_weight`` (normalized noise weight) and the KL term are
opt-in extensions used by later patches; they do nothing when unset.

Reweight discipline: ``timestep_weight`` is the *normalized* policy weight
(``mean≈1``), NOT the raw transition std.  GRPO-Guard's dt-invariant
``grad_reweight`` stays a separate post-hoc scale applied by the caller (it is
not moved in Patch A).
"""

from __future__ import annotations

from typing import Optional

import torch

from teleboost.algorithms.grpo.policy_scalars import (
    assert_broadcasts_into,
    assert_same_shape,
    reduce_log_density,
)

__all__ = ["grpo_policy_loss"]


def grpo_policy_loss(
    *,
    advantage: torch.Tensor,
    clip_range: float,
    ratio: Optional[torch.Tensor] = None,
    log_prob: Optional[torch.Tensor] = None,
    old_log_prob: Optional[torch.Tensor] = None,
    timestep_weight: Optional[torch.Tensor] = None,
    kl: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    guard_finite: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Clipped GRPO policy loss + metrics.

    Args:
        advantage: per-sample (or dense, matching ``ratio``) advantage.
        clip_range: PPO clip ε.
        ratio: importance ratio. If ``None`` it is ``exp(log_prob -
            old_log_prob)`` — pass it explicitly when it carries a RatioNorm
            adjustment (GRPO-Guard) that isn't just ``exp(Δlogp)``.
        log_prob, old_log_prob: needed when ``ratio`` is ``None`` and for the
            ``approx_kl`` metric.
        timestep_weight: optional normalized per-sample noise weight (mean≈1).
            Multiplied into the per-element loss before the mean. ``None`` ⇒
            unweighted (Wan-identical).
        kl: optional per-sample KL ``[B]``; added as ``beta * kl.mean()``.
        beta: KL coefficient. ``0`` ⇒ no KL term (Wan-identical).

    Returns:
        ``(loss, metrics)``. ``loss`` is the policy loss (+ ``beta*kl`` term).
    """
    if ratio is None:
        if log_prob is None or old_log_prob is None:
            raise ValueError("grpo_policy_loss: pass ratio, or both log_prob and old_log_prob")
        assert_same_shape(log_prob, old_log_prob, names=["log_prob", "old_log_prob"])
        ratio = torch.exp(log_prob - old_log_prob)

    # The load-bearing invariant: advantage must broadcast INTO ratio (a [1] /
    # per-sample-scalar advantage weighting all of a sample's per-step ratios is
    # fine; the dense VIPO case matches), but must NOT expand it — that
    # expanding broadcast (std [B,1,1,1,1] × adv [B] → [B,1,1,1,B]).
    assert_broadcasts_into(advantage, ratio, names=("advantage", "ratio"))

    unclipped_loss = -advantage * ratio
    clipped_loss = -advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_elem = torch.maximum(unclipped_loss, clipped_loss)

    if timestep_weight is not None:
        assert_broadcasts_into(timestep_weight, ratio, names=("timestep_weight", "ratio"))
        per_elem = timestep_weight * per_elem

    policy_loss = torch.mean(per_elem)
    loss = policy_loss

    kl_loss = None
    if beta and beta > 0.0:
        if kl is None:
            raise ValueError("grpo_policy_loss: beta>0 requires kl")
        # KL must obey the same [B] boundary as the policy term: a latent-shaped
        # KL ([B,1,1,1,1] / [B,C,T,H,W]) silently mean-collapses otherwise (the
        # same broadcast class as the P0 std bug). Reduce to [B], then assert it
        # matches the ratio so it can't cross-broadcast against advantage.
        if kl.ndim > 1:
            kl = reduce_log_density(kl, reduction="mean", name="kl")
        assert_broadcasts_into(kl, ratio, names=("kl", "ratio"))
        kl_loss = kl.mean()
        loss = policy_loss + beta * kl_loss

    if guard_finite and not torch.isfinite(loss).all():
        raise FloatingPointError(f"grpo_policy_loss produced non-finite loss (policy_loss={float(policy_loss.detach()) if torch.isfinite(policy_loss).all() else 'nan'}, ratio[min,max]=[{float(ratio.detach().min())},{float(ratio.detach().max())}])")

    with torch.no_grad():
        clipped_mask = (ratio < (1.0 - clip_range)) | (ratio > (1.0 + clip_range))
        metrics = {
            "ratio_mean": ratio.mean().item(),
            "clipfrac": clipped_mask.float().mean().item(),
            "policy_loss": policy_loss.detach().item(),
        }
        if log_prob is not None and old_log_prob is not None:
            # standard k3 approx-KL estimator (Schulman): E[(r-1) - log r]
            log_ratio = log_prob - old_log_prob
            metrics["approx_kl"] = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
        if kl_loss is not None:
            metrics["kl_loss"] = kl_loss.detach().item()

    return loss, metrics
