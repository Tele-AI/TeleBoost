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
"""SDE noise-schedule registry: DanceGRPO vs Flow-GRPO σ_t conventions.

Both DanceGRPO (arXiv 2505.07818) and Flow-GRPO (arXiv 2505.05470) recast
flow-matching sampling as an SDE solver so policy gradients have a
well-defined Gaussian transition kernel.  They differ in the σ_t
*formula* and consequently in the SDE-correction term that gets added to
the deterministic mean update:

    DanceGRPO:  σ_t = η                    (constant)
    Flow-GRPO:  σ_t = η · √( t / (1 − t) ) (t-dependent)

GRPO-Guard (arXiv 2510.22319 §3.2) formalises this distinction at the
*loss* level (see ``algorithms/grpo_guard.py`` for the matching
``GRAD_REWEIGHT_FORMS`` registry).  This module is the matching
distinction at the *SDE-step* level — same set of keys
(``"dancegrpo"`` / ``"flow_grpo"``), so the two registries compose.

Each registered function returns ``(prev_sample_mean, std_dev_t,
sqrt_dt)`` where:

* ``prev_sample_mean`` — deterministic part of the next sample,
  including the form-specific Girsanov / score-correction term.
  Adding ``randn * std_dev_t`` to this gives the full SDE update.
  Pure ODE Euler is reached via ``eta=0.0``: both forms then
  degenerate cleanly to ``latents + dsigma · model_output`` (the score
  correction and noise std go to zero together).
* ``std_dev_t`` — Gaussian std used for *both* noise injection and the
  log-probability density's variance.  Self-consistent by construction.
* ``sqrt_dt`` — ``√(σ − σ_next) = √Δt``.  Returned alongside ``std_dev_t``
  so downstream GRPO-Guard can recover ``σ_t = std_dev_t / sqrt_dt``
  without re-doing the schedule arithmetic.

Adding a new form: define a function with the
``(model_output, latents, eta, sigma, sigma_next,
pred_original_sample)`` signature returning the triple above, and add
it to ``SIGMA_FORMS``.  No dispatcher edit required.

References
----------
* Flow-GRPO upstream:
  ``yifan123/flow_grpo`` ``flow_grpo/diffusers_patch/sd3_sde_with_logprob.py``
  (the ``'sde'`` branch).  The mean-update coefficients
  ``(1 + std² /(2σ)·dt)`` on ``sample`` and
  ``(1 + std²(1−σ)/(2σ))·dt`` on ``model_output`` come from there.
* DanceGRPO: this module's existing in-code form (matches ByteDance's
  ``Dance-GRPO`` ``teleboost/programs/wan/diffusion_rollout.py``
  ``wan_step``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_StepReturn = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def compute_noise_scale_g(
    t: torch.Tensor,
    t_next: torch.Tensor,
    sigma_form: str,
) -> torch.Tensor:
    """Model-agnostic σ_t *shape* ``g(t)`` — the scalar noise-scale factor.

    ONE authoritative implementation shared by the Wan full step and reusable
    rectified-flow code, so g(t) cannot drift between call sites:

    * ``dancegrpo`` → ``g(t) = 1`` (constant σ_t = η).
    * ``flow_grpo`` → ``g(t) = √( t / (1 − t) )``.

    ``t = 1`` (first denoise step of the Wan/SD3/rectified-flow schedule) is a
    pole of the ``1 − t`` denominator. Root-fix: substitute ONLY the denominator
    with ``1 − t_next`` at ``t ≥ 1`` (numerator keeps the original ``t``, which
    is safe there). NO clamp — the root-fix already makes the denominator
    positive on any valid schedule, so a non-positive ``1 − t`` means a malformed
    schedule and is raised, not silently masked.
    """
    if sigma_form == "dancegrpo":
        return torch.ones_like(t)
    if sigma_form == "flow_grpo":
        one_minus_t = 1.0 - torch.where(t >= 1.0, t_next, t)
        if torch.any(one_minus_t <= 0):
            raise ValueError(f"flow_grpo g(t): denominator 1-t <= 0 after the t>=1 root-fix (t={t}, t_next={t_next}). Malformed schedule — need t<1, or t>=1 with t_next<1. Fix the schedule rather than clamp.")
        return torch.sqrt(t / one_minus_t)
    raise ValueError(f"unknown sigma_form {sigma_form!r}; use dancegrpo|flow_grpo")


def _dancegrpo_sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    pred_original_sample: torch.Tensor,
) -> _StepReturn:
    """DanceGRPO σ_t = η constant form.

    Matches the existing ``wan_step`` implementation byte-for-byte: same
    operations, same order, so existing rollouts stay identical when
    ``sigma_form="dancegrpo"`` (the default).

    The score-correction term is always applied — pre-refactor the
    ``sde_solver=False`` branch existed but was never reachable in
    production (all five call sites passed ``sde_solver=True``).  When
    GRPO sampling needs a pure ODE step the actual mechanism is
    ``eta=0.0`` (e.g. outside the SDE window in
    ``diffusion_rollout.py``), which makes the score correction vanish
    via ``η² = 0`` *and* zeros the Gaussian noise std — the two halves
    of the SDE go to zero together, cleanly degenerating to ODE Euler.
    """
    dsigma = sigma_next - sigma  # negative (sigma decreases)
    delta_t = sigma - sigma_next  # positive

    score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2)
    log_term = -0.5 * (eta**2) * score_estimate
    prev_sample_mean = latents + dsigma * model_output + log_term * dsigma

    sqrt_dt = torch.sqrt(delta_t)
    std_dev_t = eta * sqrt_dt
    return prev_sample_mean, std_dev_t, sqrt_dt


def _flow_grpo_sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    pred_original_sample: torch.Tensor,
) -> _StepReturn:
    """Flow-GRPO σ_t = η·√(t/(1−t)) form (paper arXiv 2505.05470 §3.2).

    Translated from ``yifan123/flow_grpo`` upstream
    ``sd3_sde_with_logprob.py`` ``'sde'`` branch:

        std_dev_t        = √(σ/(1−σ)) · noise_level
        prev_sample_mean = sample · (1 + std² / (2σ) · dt)
                         + model_output · (1 + std² · (1−σ)/(2σ)) · dt

    Note ``dt = σ_next − σ < 0``, so the sign convention matches
    upstream verbatim.  ``noise_level`` corresponds to our ``eta``.

    The effective Gaussian kernel std (for both noise injection and the
    log-prob density) is ``std_dev_t · √(−dt) = σ_t · √Δt``, so we
    return ``σ_t · √Δt`` as ``std_dev_t`` to keep the caller's contract
    identical to the DanceGRPO branch.

    The ``pred_original_sample`` argument is unused — Flow-GRPO's score
    correction is already folded into the linear coefficients of
    ``sample`` and ``model_output`` above.  It is kept in the signature
    so the registry exposes a uniform call pattern.

    Like the DanceGRPO branch, this function always applies the SDE
    correction; ``eta=0.0`` is the supported way to fall back to pure
    ODE (it zeros ``σ_t`` so all the ``std²`` terms in the linear
    coefficients vanish, leaving ``latents + dt · model_output``).

    σ=1 edge case: the formula has a pole at σ=1 because ``1−σ`` is in
    the denominator.  Flow-GRPO upstream's
    ``sd3_sde_with_logprob.py`` substitutes ``sigma_max`` (the next
    schedule value) when ``sigma == 1`` to keep ``std_dev_t`` finite.
    We mirror that: when ``sigma == 1`` we use ``sigma_next`` (which is
    the schedule's max-after-σ=1 value) as the denominator's offset.
    Without this, the first denoise step (``sigma == 1`` for Wan and
    SD3 schedulers) produces ``inf`` σ_t → NaN log-probs.
    """
    del pred_original_sample  # unused in Flow-GRPO; correction is in coeffs

    dt = sigma_next - sigma  # negative
    delta_t = sigma - sigma_next  # positive

    # σ_t shape g(σ) is the shared authority (compute_noise_scale_g), which owns
    # the σ=1 pole handling (substitute σ_next in the 1-σ denominator). The mean
    # coefficients below keep the original ``σ`` — at σ=1 those are safe
    # (numerator σ/…=1, /(2σ)=1/2, and the (1-σ) correction vanishes).
    sigma_t = compute_noise_scale_g(sigma, sigma_next, "flow_grpo") * eta

    prev_sample_mean = latents * (1.0 + (sigma_t**2) / (2.0 * sigma) * dt) + model_output * (1.0 + (sigma_t**2) * (1.0 - sigma) / (2.0 * sigma)) * dt

    sqrt_dt = torch.sqrt(delta_t)
    std_dev_t = sigma_t * sqrt_dt
    return prev_sample_mean, std_dev_t, sqrt_dt


SIGMA_FORMS: dict[str, Callable[..., _StepReturn]] = {
    "dancegrpo": _dancegrpo_sde_step,
    "flow_grpo": _flow_grpo_sde_step,
}


def compute_sde_step(
    form: str,
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    pred_original_sample: torch.Tensor,
) -> _StepReturn:
    """Dispatch one SDE step to the form-specific implementation.

    Args:
        form: one of ``SIGMA_FORMS`` keys (``"dancegrpo"``, ``"flow_grpo"``).
        model_output: flow-matching network's prediction (velocity).
        latents: current-timestep latent ``x_t``.
        eta: SDE noise scale (``actor.eta`` in the trainer config).  Set
            to ``0.0`` to fall back to pure ODE Euler in either form
            (both forms degenerate cleanly: σ_t → 0 zeros the score
            correction *and* the Gaussian noise std).
        sigma: current step's σ (= ``sigmas[index]``).
        sigma_next: next step's σ (= ``sigmas[index + 1]``); satisfies
            ``sigma_next < sigma`` along the sampling trajectory.
        pred_original_sample: ``latents − σ · model_output`` precomputed
            by the caller (used by the DanceGRPO form's score estimate;
            ignored by Flow-GRPO).

    Returns:
        Tuple ``(prev_sample_mean, std_dev_t, sqrt_dt)``.

    Raises:
        ValueError: if ``form`` is not registered in ``SIGMA_FORMS``.
    """
    if form not in SIGMA_FORMS:
        raise ValueError(f"unknown sigma_form={form!r}; valid forms: {sorted(SIGMA_FORMS.keys())}. See arxiv 2505.07818 (DanceGRPO) and 2505.05470 (Flow-GRPO).")
    return SIGMA_FORMS[form](model_output, latents, eta, sigma, sigma_next, pred_original_sample)


__all__ = [
    "SIGMA_FORMS",
    "compute_noise_scale_g",
    "compute_sde_step",
]
