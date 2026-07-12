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
"""TeleBoost shared algorithm layer.

This package holds what every run shares, regardless of which algorithm
vertical is selected:

* :mod:`grpo_advantage` / :mod:`grpo_loss` — the baseline GRPO math.
* :mod:`sigma_schedule` — the SDE step-form registry (the sampling
  regime itself, DanceGRPO vs Flow-GRPO forms).
* :mod:`grpo_guard` — a cross-cutting loss-level regularizer,
  composable with any driver-phase algorithm; its
  ``GRAD_REWEIGHT_FORMS`` registry mirrors ``sigma_schedule``'s key set.
(Multi-reward pieces live by kind: driver orchestration in
:mod:`teleboost.training.rewarding`, pure advantage-weight math in
:mod:`teleboost.training.rewarding.joint_advantage_weights`.)

Per-algorithm math lives in this package; program-specific trainer wiring
lives under ``teleboost.programs``.  The shared layer names no optional
algorithm; see ``recipes/README.md`` for the layering contract and
``teleboost/algorithms/README.md`` for the per-algorithm map.
"""

from teleboost.algorithms.grpo.advantage import (
    per_prompt_zscore_advantage,
)
from teleboost.algorithms.grpo_guard import (
    GRAD_REWEIGHT_FORMS,
    compute_grad_reweight_delta,
    compute_ratio_norm_bias,
)
from teleboost.algorithms.grpo.sigma_schedule import (
    SIGMA_FORMS,
    compute_noise_scale_g,
    compute_sde_step,
)

__all__ = [
    # GRPO advantage (paper arxiv 2402.03300 + 2505.07818)
    "per_prompt_zscore_advantage",
    # GRPO-Guard (paper arxiv 2510.22319) — cross-cutting loss regularizer
    "compute_ratio_norm_bias",
    "compute_grad_reweight_delta",
    "GRAD_REWEIGHT_FORMS",
    # SDE σ_t schedule registry (DanceGRPO 2505.07818 vs Flow-GRPO 2505.05470)
    "SIGMA_FORMS",
    "compute_noise_scale_g",
    "compute_sde_step",
]
