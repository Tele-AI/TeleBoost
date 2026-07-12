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
"""GRPO-Guard: RatioNorm + grad-reweight policy-loss adjustments.

Paper: "GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching
via Regulated Clipping", arXiv 2510.22319 (Jing Wang, Jiajun Liang, et al.).

The paper's contribution is a pair of orthogonal corrections to the
PPO-style update used in flow-matching / diffusion RL (DanceGRPO,
Flow-GRPO, etc.):

* **RatioNorm (Eq. 8)** rewrites the importance-sampling ratio so that
  the bias introduced by per-step Δμ (the difference between the new and
  old policy means inside the Gaussian rollout transition) is
  *explicitly accounted for*::

      log r̂_t(θ) = σ_t · √Δt · ( log r_t(θ) + ‖Δμ_θ‖² / (2 σ_t² Δt) )
                 = − Δμ_θ · ε                    (where ε ∼ N(0, I))

  Compared to the naive ``r_t = exp(log p_new − log p_old)``, this
  changes both the inside (additive bias term) and the outside (a
  ``σ_t · √Δt`` outer scale).

* **Grad-reweight (Eq. 12)** rescales the policy loss by ``δ = β/Δt``
  so the gradient magnitude is dt-invariant.  The paper presents two
  shapes for ``β``:

      flow_grpo  (Flow-GRPO):  β = 1 + η²/2 (constant, absorbed)  ⇒  δ = 1/Δt
      dancegrpo  (DanceGRPO):  β = 1 + η²(1−t)/(2t)
                              ⇒  δ = (1 + η²(1−t)/(2t)) / Δt

  Section 4.3 of the paper treats RatioNorm and grad-reweight as
  **independent** ablation levers (Mean-revised / RatioNorm /
  GRPO-Guard-combined), so this module exposes them as two separate
  helpers with separate flags.

Adding a new δ form: define a function with the same signature as
``_delta_flow_grpo`` and add it to ``GRAD_REWEIGHT_FORMS``.  No
dispatcher code change required.

This module owns *only* the math.  Wiring (config reading, batch
plumbing, metric reporting) lives in ``dp_actor.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_TensorOrFloat = torch.Tensor | float


# ---------------------------------------------------------------------------
# RatioNorm (paper Eq. 8)
# ---------------------------------------------------------------------------


def compute_ratio_norm_bias(
    prev_sample_mean_new: torch.Tensor,
    prev_sample_mean_old: torch.Tensor,
    sqrt_dt: _TensorOrFloat,
    std_dev_t: _TensorOrFloat,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, _TensorOrFloat, _TensorOrFloat]:
    """Compute the RatioNorm additive bias term and outer scale.

    Implements paper Eq. 8 in the form actually used by the actor::

        Δμ                = μ_new(x_t, t) − μ_old(x_t, t)
        ‖Δμ‖²             = mean over non-batch dims (per-sample scalar)
        sigma_t           = std_dev_t / (sqrt_dt + eps)
        scale             = sqrt_dt · sigma_t          (≡ std_dev_t)
        ratio_mean_bias   = ‖Δμ‖² / (2 · scale² + eps)

    Returns:
        ratio_mean_bias: shape (batch,) — the additive bias added to
            ``log r_t(θ)`` *inside* the exp (paper Eq. 8 LHS).
        scale:           the outer ``σ_t · √Δt`` multiplier applied
            after the additive bias (paper Eq. 8 RHS outer factor).
        sqrt_dt_scalar:  scalar reduction of ``sqrt_dt`` for downstream
            reuse (e.g. in ``compute_grad_reweight_delta``).

    The function is logic-preserving with the inline implementation in
    ``dp_actor.update_policy`` from commit 912f0e32 — same operations,
    same reduction order, same eps placement.
    """
    diff_squared = (prev_sample_mean_new - prev_sample_mean_old).pow(2)
    if diff_squared.ndim > 1:
        ratio_mean_bias = diff_squared.flatten(start_dim=1).mean(dim=1)
    else:
        ratio_mean_bias = diff_squared.mean()
    ratio_mean_bias = ratio_mean_bias.flatten()

    sqrt_dt_scalar = sqrt_dt.mean() if isinstance(sqrt_dt, torch.Tensor) and sqrt_dt.ndim > 0 else sqrt_dt
    std_dev_t_scalar = std_dev_t.mean() if isinstance(std_dev_t, torch.Tensor) and std_dev_t.ndim > 0 else std_dev_t

    sigma_t = std_dev_t_scalar / (sqrt_dt_scalar + eps)
    scale = sqrt_dt_scalar * sigma_t
    ratio_mean_bias = ratio_mean_bias / (2 * (scale**2 + eps))

    return ratio_mean_bias, scale, sqrt_dt_scalar


# ---------------------------------------------------------------------------
# Grad-reweight δ (paper Eq. 12)
# ---------------------------------------------------------------------------


def _delta_flow_grpo(
    t: _TensorOrFloat,
    dt: _TensorOrFloat,
    eta: float,
    eps: float,
) -> _TensorOrFloat:
    """Flow-GRPO grad-reweight: ``δ = 1/Δt`` (β ≈ const)."""
    return 1.0 / (dt + eps)


def _delta_dancegrpo(
    t: _TensorOrFloat,
    dt: _TensorOrFloat,
    eta: float,
    eps: float,
) -> _TensorOrFloat:
    """DanceGRPO grad-reweight: ``δ = (1 + η²(1−t)/(2t)) / Δt``.

    The factor ``(1−t)/(2t)`` blows up as t → 0; ``eps`` doubles as the
    floor for ``t`` so the early-noise end of the schedule cannot
    produce ``inf`` δ.
    """
    if isinstance(t, torch.Tensor):
        t_safe = torch.clamp(t, min=eps)
    else:
        t_safe = max(float(t), eps)
    beta = 1.0 + (eta**2) * (1.0 - t_safe) / (2.0 * t_safe)
    return beta / (dt + eps)


# Dict registry of grad-reweight δ shapes.  Adding a new form: define a
# function with the (t, dt, eta, eps) -> δ signature and register here.
GRAD_REWEIGHT_FORMS: dict[str, Callable[..., _TensorOrFloat]] = {
    "flow_grpo": _delta_flow_grpo,
    "dancegrpo": _delta_dancegrpo,
}


def compute_grad_reweight_delta(
    form: str,
    t: _TensorOrFloat,
    dt: _TensorOrFloat,
    eta: float,
    eps: float = 1e-6,
) -> _TensorOrFloat:
    """Return the policy-loss reweight scalar δ for the chosen form.

    Args:
        form: one of ``GRAD_REWEIGHT_FORMS`` keys (currently
            ``"flow_grpo"`` and ``"dancegrpo"``).
        t:    SDE time of the current denoise step (in [0, 1] for the
            flow-matching schedule).  May be a scalar or a tensor that
            the form-specific helper will reduce; ``compute_grad_reweight_delta``
            does **not** reduce ``t`` itself, so callers should pass an
            already-reduced scalar when they want δ to be batch-uniform.
        dt:   the step's Δt (= sqrt_dt²).  Same scalar/tensor flexibility
            as ``t``.
        eta:  the SDE noise scale (== ``actor.eta`` in the trainer
            config).
        eps:  numerical guard.  Used for both ``1/dt`` and the t-floor in
            the dancegrpo form.

    Raises:
        ValueError: if ``form`` is not in ``GRAD_REWEIGHT_FORMS``.
    """
    if form not in GRAD_REWEIGHT_FORMS:
        raise ValueError(f"unknown grad_reweight_form={form!r}; valid forms: {sorted(GRAD_REWEIGHT_FORMS.keys())}. See arxiv 2510.22319 §3.2.3.")
    return GRAD_REWEIGHT_FORMS[form](t, dt, eta, eps)


__all__ = [
    "compute_ratio_norm_bias",
    "compute_grad_reweight_delta",
    "GRAD_REWEIGHT_FORMS",
]
