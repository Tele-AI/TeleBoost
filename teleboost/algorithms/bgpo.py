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
"""BGPO — TeleAI's Bayesian-prior-driven GRPO refinement (arxiv 2511.18919).

**Naming note.** The paper names the method **BPGO** (*Bayesian
Prior-Guided Optimization*). This repository keeps the existing
``BGPO`` identifier for backward compatibility with config keys
``algorithm.bgpo.*``, env vars ``BGPO_*``, and ``BGPOMixin``. Use
**BPGO** in external citations; use **BGPO** when referring to this
repo's code/config surface.

This module follows the **reference implementation** (the authors'
released code); the paper's published equations describe it at a higher
level. Two branches share the ``algorithm.bgpo.enable`` flag:

* **CRT** (``use_rerange``) — Contrastive Reward Transformation. Rewards
  are rearranged around the prior before the advantage is computed::

      R̃ = [a·(R − R_prior) + 𝟙{R > R_prior}] / (1 + exp(−R/τ)) · R

* **RAS** (``adaptive_weight_method=bayes``) — Reliability-Adaptive
  Scaling. The scalar advantage is scaled by ``w = 1 + α·(w_bayes − 1)``,
  where ``w_bayes`` is a Bayesian posterior reliability gain — the
  probability the group posterior beats the prior under a Normal posterior
  with ``prior_var`` shrinkage.

When ``enable=false`` the mixin is a no-op and the trainer follows the
baseline GRPO path bit-for-bit.

The trainer-side hooks live in ``teleboost/programs/wan/bgpo.py`` so the trainer can
inherit it and keep ``self.config`` / ``self.global_steps`` shared with
the rest of the training loop. Pure helpers are module-level functions
so they can be unit-tested without spinning up a full trainer.

"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no trainer state)
# ---------------------------------------------------------------------------


def binary_rerange_group_rewards(
    group_rewards: torch.Tensor,
    prior: float,
    a: float,
    temperature: float,
    exp_clamp: float = 30.0,
) -> torch.Tensor:
    """CRT reward rearrangement, ``binary`` prior-threshold variant.

    Reference-implementation form (``rerange_method="binary"``):

    .. math::

        \\tilde{R} = \\frac{a\\,(R - R_{prior}) + \\mathbb{1}\\{R > R_{prior}\\}}
                          {1 + \\exp(-R / \\tau)} \\cdot R

    Args:
        group_rewards: raw rewards within a single rollout group.
        prior: ``R_prior`` for this prompt.
        a: slope on the ``(R − R_prior)`` term (config ``rerange_a``).
        temperature: ``τ`` in the sigmoid gate (config ``rerange_temperature``).
        exp_clamp: numerical-safety clamp on the ``exp`` argument.
    """
    flag = group_rewards - prior
    positive_sign = torch.clamp(torch.sign(flag), min=0.0)
    numerator = a * flag + positive_sign
    exponent = torch.clamp(-group_rewards / max(temperature, 1e-8), min=-exp_clamp, max=exp_clamp)
    denom = 1.0 + torch.exp(exponent)
    coef = numerator / denom
    return coef * group_rewards


def bayes_reliability_weight(
    group_rewards: torch.Tensor,
    prior: float,
    prior_var: float = 1.0,
    weight_range: tuple = (0.5, 1.5),
) -> float:
    """RAS Bayesian reliability gain (reference implementation, ``bayes``).

    Treats the group rewards as evidence updating a Normal prior of
    variance ``prior_var``, then measures how strongly the group's
    posterior beats the prior. Returns the *centered* weight ``w − 1`` so
    the caller forms ``w = 1 + α·(w − 1)`` (see ``_apply_bgpo_on_advantages``
    in ``teleboost/programs/wan/bgpo.py``). The base weight ``w`` lies in
    ``weight_range`` and rises with ``P(R_prior < posterior mean)``.

    Args:
        group_rewards: rewards in a single rollout group, shape ``(rollout_n,)``.
        prior: ``R_prior`` for the prompt.
        prior_var: variance of the Normal prior (shrinkage strength).
        weight_range: ``(w_min, w_max)`` bounds on the base weight.
    """
    n = int(group_rewards.shape[0])
    if n <= 1:
        sample_var = torch.tensor(1.0, device=group_rewards.device, dtype=group_rewards.dtype)
    else:
        sample_var = group_rewards.var(unbiased=True) + 1e-8
    w_min, w_max = float(weight_range[0]), float(weight_range[1])
    sample_mean = group_rewards.mean()
    posterior_var = 1.0 / (n / sample_var + 1.0 / prior_var)
    posterior_mean = posterior_var * (n * sample_mean / sample_var + prior / prior_var)
    posterior_std = torch.sqrt(posterior_var)
    z = (prior - posterior_mean) / posterior_std
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=group_rewards.device, dtype=group_rewards.dtype))
    prob_better = 1.0 - 0.5 * (1 + torch.erf(z / sqrt_2))
    weight = w_min + (w_max - w_min) * prob_better
    return float(weight.item() - 1.0)


__all__ = [
    "binary_rerange_group_rewards",
    "bayes_reliability_weight",
]
