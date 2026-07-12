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
"""Rollout-record contract for diffusion-RL (Patch A).

Every per-sample / per-timestep rollout record that later feeds the policy
recompute + GRPO loss must carry enough metadata to make the recompute
*self-consistent* — so the loss boundary never has to guess how the stored
log-prob was produced.  This module is the single declaration of those
required fields + a validator; each model's adapter populates them.

The four fields beyond the obvious tensors are the ones whose absence caused
the bugs we found in standalone forks:

* ``logprob_reduction`` — "mean" | "sum" over non-batch dims. The recompute
  MUST reduce ``new_log_prob`` the same way (see
  :func:`policy_scalars.reduce_log_density`) or the ratio is silently wrong.
* ``solver_id`` — which SolverContract produced the trajectory; the recompute
  + eval must use the same one (no UniPC-rollout / Euler-logprob mixing).
* ``sigma_form`` — the SIGMA_FORMS key, so the SDE step is reproduced exactly.
* ``policy_forward_kind`` — MUST be a training forward (e.g. the model's SFT
  forward); the inference/generate path must never feed the training log-prob
  (its in-place / KV-cache ops also break autograd).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "REQUIRED_ROLLOUT_FIELDS",
    "VALID_LOGPROB_REDUCTIONS",
    "VALID_POLICY_FORWARD_KINDS",
    "validate_rollout_record",
    "validate_solver_contract",
]

# Per-sample tensors + the metadata that makes the recompute self-consistent.
REQUIRED_ROLLOUT_FIELDS = (
    "prompt_id",
    "sample_id",
    "timestep_index",
    "timestep_value",
    "latent_t",
    "latent_next",
    "old_log_prob",
    "reward",
    "advantage",
    # self-consistency metadata
    "logprob_reduction",
    "solver_id",
    "sigma_form",
    "policy_forward_kind",
)

VALID_LOGPROB_REDUCTIONS = ("mean", "sum", "channel_sum_dense")
# Only training forwards may produce the policy log-prob. Inference samplers
# (in-place packed-sequence writes + KV cache) are forbidden — they break
# autograd and aren't the distribution we optimize.
VALID_POLICY_FORWARD_KINDS = ("training_forward", "sft_forward")


def validate_rollout_record(record: Mapping[str, Any]) -> None:
    """Raise if a rollout record is missing fields or has invalid metadata."""
    missing = [f for f in REQUIRED_ROLLOUT_FIELDS if f not in record]
    if missing:
        raise ValueError(f"rollout record missing required fields: {missing}")

    red = record["logprob_reduction"]
    if red not in VALID_LOGPROB_REDUCTIONS:
        raise ValueError(f"logprob_reduction={red!r} invalid; use one of {VALID_LOGPROB_REDUCTIONS}")

    kind = record["policy_forward_kind"]
    if kind not in VALID_POLICY_FORWARD_KINDS:
        raise ValueError(f"policy_forward_kind={kind!r} invalid; the policy log-prob must come from a TRAINING forward, one of {VALID_POLICY_FORWARD_KINDS} — never an inference/generate sampler.")

    # sigma_form must be a registered SDE-step form (lazy import keeps this
    # schema module free of the torch-heavy sigma_schedule at load time).
    from teleboost.algorithms.grpo.sigma_schedule import SIGMA_FORMS

    if record["sigma_form"] not in SIGMA_FORMS:
        raise ValueError(f"sigma_form={record['sigma_form']!r} is not a registered SDE-step form; valid forms: {sorted(SIGMA_FORMS.keys())}.")


def validate_solver_contract(record: Mapping[str, Any], expected_contract: Any) -> None:
    """Hard-fail if a rollout record's solver metadata diverges from the contract.

    ``expected_contract`` is a :class:`~teleboost.algorithms.solver_contract.
    SolverContract` (duck-typed here to avoid an import cycle). Catches the
    UniPC-rollout / Euler-logprob class of mismatch at the data boundary.
    """
    validate_rollout_record(record)
    expected_contract.assert_matches_record(
        solver_id=record["solver_id"],
        sigma_form=record["sigma_form"],
        logprob_reduction=record["logprob_reduction"],
    )
