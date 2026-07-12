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
"""Normalized noise-aware timestep weighting for diffusion-RL.

TempFlow-GRPO's noise-aware policy weighting (paper arXiv 2508.04324, Eq. 8):

    J_policy = (1/G) Σ_i (1/T) Σ_t  Norm(σ_t·√Δt) · min(r·Â, clip(r)·Â)

i.e. each per-timestep clipped GRPO term is scaled by ``Norm(σ_t·√Δt)``, where
``σ_t = a·√(t/(1-t))`` (paper Eq. 5) is the SDE noise level and ``Norm(·)``
normalizes the weight to **mean 1** across the T timesteps (so it reweights
*across* timesteps without rescaling the loss magnitude). The paper's
motivation (Fig. 4): the SDE noise level σ_t·√Δt is large at high-noise early
steps and ~0 at low-noise late steps, and standard GRPO's scale term is
*inversely* proportional to it — so late refinement steps wrongly dominate the
gradient. Weighting by Norm(σ_t·√Δt) realigns optimization pressure with each
step's actual exploration capacity.

Paper-faithfulness: ``σ_t = a·√(t/(1-t))`` is exactly our ``flow_grpo`` σ_t
form (``eta·√(σ/(1-σ))``), so ``compute_schedule_noise_std(..., sigma_form=
"flow_grpo")`` returns the paper's raw weight ``σ_t·√Δt``. With a *constant*-σ_t
form (``dancegrpo``, σ_t=η) the weight degenerates to ``Norm(√Δt)`` and is NOT
the paper's noise-aware weighting — ``compute_timestep_weights`` warns in that
case.

We differ from the official TempFlow scripts only in the global scale: they
multiply by a hand-tuned per-model constant, we normalize to **mean 1** (same
Eq. 8 per-step shape ∝ σ_t·√Δt, the global scale absorbed by the LR).

This module owns the math:

* :func:`compute_schedule_noise_std` — the per-timestep transition std
  ``σ_t·√Δt`` for the whole schedule, computed via the registered
  :data:`SIGMA_FORMS` SDE step (so the weight and the rollout/recompute use one
  σ_t convention — no second formula scattered in a rollout worker).
* :func:`compute_timestep_weights` — turn that into a ``[T]`` policy weight with
  ``mean ≈ 1`` (so it reweights *across* timesteps without rescaling the loss
  magnitude). Modes: ``none`` / ``tempflow_noise_norm``.
* :func:`resolve_timestep_weights` — the single driver entry point: turn a
  batch ``[B,T+1]`` (or ``[T+1]``) sigma schedule directly into the ``[T]``
  mean-1 weight vector (or ``None`` when disabled), asserting the schedule is
  shared across the batch. ``update_policy`` calls this once per micro-batch.
* :func:`summarize_noise_weights` — raw + normalized min/max/mean for the
  startup log, so "solver correction" and "reweight in effect" stay separable
  when reading experiments.

The result plugs into :func:`grpo_loss.grpo_policy_loss` via its
``timestep_weight`` arg (the caller passes ``weights[k]`` for the *global*
denoise step ``k`` trained at this loop position). The raw transition std stays
the SDE density's std — it is NEVER used directly as the loss weight.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from teleboost.algorithms.grpo.sigma_schedule import SIGMA_FORMS, compute_sde_step

__all__ = [
    "REWEIGHT_MODES",
    "compute_schedule_noise_std",
    "compute_timestep_weights",
    "resolve_timestep_weights",
    "summarize_noise_weights",
]

# TempFlow Eq. 8 noise-aware weighting only. GRPO-Guard's dt-invariant
# grad-reweight is a *different* mechanism and lives in grpo_guard.py — it is
# deliberately NOT offered here so this module is unambiguously TempFlow.
REWEIGHT_MODES = ("none", "tempflow_noise_norm")


def compute_schedule_noise_std(sigmas: torch.Tensor, *, eta: float, sigma_form: str) -> torch.Tensor:
    """Per-timestep transition std ``σ_t·√Δt`` for a sigma schedule → ``[T]``.

    ``sigmas`` is the ``[T+1]`` schedule (descending, e.g. 1→0); the returned
    ``[T]`` are the stds for steps ``0..T-1``. Computed through the registered
    SDE step so it matches the actual rollout/recompute transition.
    """
    if sigma_form not in SIGMA_FORMS:
        raise ValueError(f"sigma_form={sigma_form!r} not registered; valid: {sorted(SIGMA_FORMS.keys())}")
    sigmas = sigmas.detach().float()
    if sigmas.ndim != 1 or sigmas.shape[0] < 2:
        raise ValueError(f"sigmas must be a 1-D schedule of length >=2, got {tuple(sigmas.shape)}")
    zero = torch.zeros((), dtype=torch.float32)
    stds = []
    for i in range(sigmas.shape[0] - 1):
        _, std_dev_t, _ = compute_sde_step(
            form=sigma_form,
            model_output=zero,
            latents=zero,
            eta=eta,
            sigma=sigmas[i],
            sigma_next=sigmas[i + 1],
            pred_original_sample=zero,
        )
        stds.append(std_dev_t.reshape(()).float())
    return torch.stack(stds)


def compute_timestep_weights(
    noise_std: torch.Tensor,
    *,
    mode: str,
    sigma_form: Optional[str] = None,
    eps: float = 1e-8,
    assert_mean_one: bool = True,
) -> torch.Tensor:
    """Normalized per-timestep policy weight ``[T]`` (mean ≈ 1).

    * ``none`` — all ones (unweighted; Wan-identical).
    * ``tempflow_noise_norm`` — ``Norm(σ_t·√Δt)`` mean-1 (TempFlow Eq. 8).
      Paper-faithful only when ``noise_std`` came from the ``flow_grpo`` σ_t
      form; pass ``sigma_form`` so a constant-σ_t (``dancegrpo``) misuse warns.
    """
    noise_std = noise_std.float()
    if mode == "none":
        return torch.ones_like(noise_std)
    if mode == "tempflow_noise_norm":
        if sigma_form is not None and sigma_form != "flow_grpo":
            warnings.warn(
                f"tempflow_noise_norm with sigma_form={sigma_form!r}: TempFlow Eq. 8 assumes σ_t=a·√(t/(1-t)) (the 'flow_grpo' form). With a constant-σ_t form the weight degenerates to Norm(√Δt) and is NOT the paper's noise-aware weighting. Use sigma_form='flow_grpo' for paper fidelity.",
                RuntimeWarning,
                stacklevel=2,
            )
        w = noise_std / (noise_std.mean() + eps)
    else:
        raise ValueError(f"unknown reweight mode {mode!r}; use one of {REWEIGHT_MODES}")

    if assert_mean_one:
        m = float(w.mean())
        if abs(m - 1.0) > 1e-4:
            raise AssertionError(f"normalized timestep weights must have mean≈1, got {m}")
    return w


def resolve_timestep_weights(
    sigma_schedule: torch.Tensor,
    *,
    mode: str,
    eta: float,
    sigma_form: str,
) -> Optional[torch.Tensor]:
    """Driver entry point: ``[B,T+1]``/``[T+1]`` schedule → ``[T]`` weights or ``None``.

    ``None`` when ``mode == "none"`` (unweighted / Wan-identical). Otherwise the
    mean-1 weight vector for the schedule's transitions. The batch is asserted to
    share one denoise schedule (TempFlow weighting assumes a single schedule),
    so we derive the ``[T]`` vector from row 0.
    """
    if mode == "none":
        return None
    if sigma_schedule.dim() == 2 and sigma_schedule.shape[0] > 1:
        max_dev = (sigma_schedule - sigma_schedule[:1]).abs().max()
        if float(max_dev) > 1e-6:
            raise AssertionError(f"sigma_schedule differs across the batch (max dev {float(max_dev):.2e}); TempFlow noise weighting assumes one shared denoise schedule.")
    sigmas = sigma_schedule[0] if sigma_schedule.dim() == 2 else sigma_schedule
    noise_std = compute_schedule_noise_std(sigmas, eta=eta, sigma_form=sigma_form)
    return compute_timestep_weights(noise_std, mode=mode, sigma_form=sigma_form)


def summarize_noise_weights(noise_std: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    """Raw + normalized weight stats for the startup log."""
    ns = noise_std.float()
    w = weights.float()
    return {
        "raw_noise_weight/min": float(ns.min()),
        "raw_noise_weight/max": float(ns.max()),
        "raw_noise_weight/mean": float(ns.mean()),
        "normalized_weight/min": float(w.min()),
        "normalized_weight/max": float(w.max()),
        "normalized_weight/mean": float(w.mean()),
    }
