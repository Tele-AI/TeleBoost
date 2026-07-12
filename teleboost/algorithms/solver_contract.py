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
"""SolverContract — runtime self-consistency guard for diffusion-RL solvers.

Patch B of the common diffusion-RL layer.  The single most dangerous
open-source footgun is a config where the rollout draws its trajectory with
one solver but the policy log-prob is recomputed with a *different* transition
— e.g. a UniPC multistep rollout paired with a single-step Euler-SDE log-prob.
That silently invalidates the GRPO importance ratio: the stored ``old_log_prob``
is not the density of the trajectory under the rollout policy, so ``exp(new −
old)`` is not a real importance weight.  The run *looks* like it trains; the
gradient is wrong.

This contract makes that a startup hard-fail instead of a silent error:

* ``sigma_form`` must be a registered :data:`SIGMA_FORMS` key.
* ``rollout_transition`` must equal ``recompute_transition`` — unless the
  caller *explicitly* opts into a mismatch via ``allow_transition_mismatch``
  (for diagnostic / known-unsupported experiments), which is then surfaced.
* ``eval_transition`` MAY differ (eval is usually a pure ODE sampler) — eval is
  not the policy distribution, so it never feeds the recompute log-prob.  It is
  recorded for clarity but not required to match.
* ``logprob_reduction`` must be a valid reduction (the new/old log-prob must use
  the same one — see :mod:`policy_scalars`).
"""

from __future__ import annotations

from dataclasses import dataclass

from teleboost.algorithms.rollout_contract import VALID_LOGPROB_REDUCTIONS
from teleboost.algorithms.grpo.sigma_schedule import SIGMA_FORMS

__all__ = ["SolverContract"]


@dataclass(frozen=True)
class SolverContract:
    """Declares + validates the solver used across rollout / recompute / eval.

    ``name`` is a human label (e.g. ``"euler_flow"``).  ``rollout_transition``
    and ``recompute_transition`` name the *actual* one-step update used to draw
    the trajectory and to recompute the policy log-prob; they MUST match so the
    GRPO ratio is the density under the rollout policy.
    """

    name: str
    sigma_form: str
    rollout_transition: str
    recompute_transition: str
    eval_transition: str
    logprob_reduction: str = "mean"
    allow_transition_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.sigma_form not in SIGMA_FORMS:
            raise ValueError(f"SolverContract({self.name!r}): sigma_form={self.sigma_form!r} is not registered; valid forms: {sorted(SIGMA_FORMS.keys())}.")
        if self.logprob_reduction not in VALID_LOGPROB_REDUCTIONS:
            raise ValueError(f"SolverContract({self.name!r}): logprob_reduction={self.logprob_reduction!r} invalid; use one of {VALID_LOGPROB_REDUCTIONS}.")
        if self.rollout_transition != self.recompute_transition and not self.allow_transition_mismatch:
            raise ValueError(
                f"SolverContract({self.name!r}): rollout_transition="
                f"{self.rollout_transition!r} != recompute_transition="
                f"{self.recompute_transition!r}. The GRPO importance ratio is only "
                f"valid when the policy log-prob is recomputed with the SAME "
                f"transition that drew the trajectory (e.g. a UniPC multistep "
                f"rollout paired with an Euler-SDE log-prob is INVALID). Fix the "
                f"config to use one solver for both, or set "
                f"allow_transition_mismatch=True for an explicit diagnostic run."
            )

    @property
    def transition_consistent(self) -> bool:
        return self.rollout_transition == self.recompute_transition

    def assert_matches_record(self, *, solver_id: str, sigma_form: str, logprob_reduction: str) -> None:
        """Hard-fail if a rollout record's solver metadata diverges from this contract."""
        mism = []
        if solver_id != self.name:
            mism.append(f"solver_id {solver_id!r} != {self.name!r}")
        if sigma_form != self.sigma_form:
            mism.append(f"sigma_form {sigma_form!r} != {self.sigma_form!r}")
        if logprob_reduction != self.logprob_reduction:
            mism.append(f"logprob_reduction {logprob_reduction!r} != {self.logprob_reduction!r}")
        if mism:
            raise ValueError(f"rollout record violates SolverContract({self.name!r}): " + "; ".join(mism) + ". UniPC-rollout / Euler-logprob style mixing is invalid for the GRPO ratio.")
