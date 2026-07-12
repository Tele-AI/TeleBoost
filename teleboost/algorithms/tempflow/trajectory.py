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
"""Trajectory-branching process-reward ledger (TempFlow-GRPO).

TempFlow-GRPO (paper arXiv 2508.04324) assigns a *process* reward to each
denoise timestep without a step-level reward model, via **trajectory
branching** (Def. 1): run the ODE deterministically to a branch timestep ``k``,
inject one SDE step there (Eq. 5), then run ODE to ``x_0`` and score the final
image — so the terminal reward becomes the process reward for step ``k``.
Theorem 1 (Credit Localization): because all stochasticity is concentrated at
``k``, the reward variance is attributable to that branch, so the group-relative
advantage (Eq. 3) is computed *per branch point*.

This module owns the **ledger** — the explicit bookkeeping that the standalone
forks got wrong by aligning rewards to timesteps implicitly via tensor dims:

* every reward record carries its ``branch_timestep_index`` explicitly;
* the advantage group boundary is computed + returned (loggable / auditable),
  never inferred from a reshape.

The actual ODE-SDE-ODE rollout (the model-side branch generation) is the
adapter's job; here we only plan the branch points and turn branched rewards
into per-(timestep, sample) advantages with explicit group keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "BRANCH_POINT_MODES",
    "REWARD_TARGETS",
    "ADVANTAGE_GROUPS",
    "BranchPlan",
    "BranchRecord",
    "BranchSample",
    "select_branch_points",
    "select_trainable_branch_points",
    "compute_branched_advantages",
    "compute_branched_advantage",
]

BRANCH_POINT_MODES = ("all", "early_k", "sampled_k")
REWARD_TARGETS = ("final_image", "middle_frame", "video_reward")
# TempFlow Eq. 3 normalizes the advantage over the G samples of ONE prompt c
# (here, per branch timestep). That is the only correct group, so it is the
# only one we offer — cross-prompt "seed"/"global" groupings are not Eq. 3.
ADVANTAGE_GROUPS = ("prompt",)


@dataclass(frozen=True)
class BranchPlan:
    """Explicit trajectory-branching configuration (no implicit defaults hidden
    in a rollout worker)."""

    branch_points: str = "all"
    exploration_k: int = 1  # SDE branches drawn per (group, branch point)
    branch_transition: str = "sde"  # fixed by Def. 1
    tail_transition: str = "ode"  # fixed by Def. 1
    reward_target: str = "final_image"
    advantage_group: str = "prompt"
    early_k: Optional[int] = None  # for branch_points="early_k"
    num_sampled: Optional[int] = None  # for branch_points="sampled_k"

    def __post_init__(self) -> None:
        if self.branch_points not in BRANCH_POINT_MODES:
            raise ValueError(f"branch_points={self.branch_points!r} invalid; use {BRANCH_POINT_MODES}")
        if self.reward_target not in REWARD_TARGETS:
            raise ValueError(f"reward_target={self.reward_target!r} invalid; use {REWARD_TARGETS}")
        if self.advantage_group not in ADVANTAGE_GROUPS:
            raise ValueError(f"advantage_group={self.advantage_group!r} invalid; use {ADVANTAGE_GROUPS}")
        if self.branch_transition != "sde" or self.tail_transition != "ode":
            raise ValueError("Def. 1 fixes branch_transition='sde' and tail_transition='ode'")
        if self.exploration_k < 1:
            raise ValueError("exploration_k must be >= 1")
        if self.branch_points == "early_k" and not self.early_k:
            raise ValueError("branch_points='early_k' requires early_k")
        if self.branch_points == "sampled_k" and not self.num_sampled:
            raise ValueError("branch_points='sampled_k' requires num_sampled")


def select_branch_points(
    num_timesteps: int,
    plan: BranchPlan,
    *,
    sampled_indices: Optional[Sequence[int]] = None,
) -> list[int]:
    """Return the denoise timestep indices to branch at.

    ``all`` → every step; ``early_k`` → the first ``early_k`` (high-noise)
    steps, where exploration capacity is largest; ``sampled_k`` → an explicit
    ``sampled_indices`` (caller-provided, to keep this deterministic) of size
    ``num_sampled``.
    """
    if num_timesteps < 1:
        raise ValueError("num_timesteps must be >= 1")
    if plan.branch_points == "all":
        return list(range(num_timesteps))
    if plan.branch_points == "early_k":
        return list(range(min(plan.early_k, num_timesteps)))
    # sampled_k
    if sampled_indices is None:
        raise ValueError("branch_points='sampled_k' needs sampled_indices (deterministic selection)")
    idx = sorted(set(int(i) for i in sampled_indices))
    if any(i < 0 or i >= num_timesteps for i in idx):
        raise ValueError(f"sampled_indices out of range [0,{num_timesteps})")
    if len(idx) != plan.num_sampled:
        raise ValueError(f"expected {plan.num_sampled} sampled indices, got {len(idx)}")
    return idx


def select_trainable_branch_points(
    sampling_steps: int,
    plan: BranchPlan,
    *,
    sampled_indices: Optional[Sequence[int]] = None,
) -> list[int]:
    """Branch points restricted to the TRAINABLE steps.

    The actor drops the final ``σ → 0`` transition (peaked log-prob / 0-std →
    NaN), so only steps ``0 .. sampling_steps-2`` are ever trained. A branch at
    the dropped step would have no usable policy-gradient target, so cap branch
    points at ``trainable_steps = sampling_steps - 1`` (i.e. ``k ≤
    sampling_steps - 2``). The deterministic ODE tail still runs through the
    dropped step to reach a decodable ``x_0``.
    """
    trainable_steps = sampling_steps - 1
    if trainable_steps < 1:
        raise ValueError(f"sampling_steps={sampling_steps} leaves no trainable step")
    points = select_branch_points(trainable_steps, plan, sampled_indices=sampled_indices)
    # defensive: select_branch_points already bounds to [0, trainable_steps)
    assert all(0 <= k <= sampling_steps - 2 for k in points), points
    return points


@dataclass
class BranchRecord:
    """One branched rollout outcome. ``branch_timestep_index`` is explicit — the
    reward knows which step it credits, never inferred from a tensor axis."""

    prompt_id: int
    branch_timestep_index: int
    sample_id: int
    reward: float
    advantage: Optional[float] = None


@dataclass
class BranchSample:
    """Full rollout→trainer hand-off record for ONE trajectory branch.

    Every field the actor/trainer needs is carried EXPLICITLY so M3 aligns by
    name, never by reshaping a stacked tensor axis. Identity fields locate the
    branch; the training-tensor fields describe the single SDE transition at
    ``branch_timestep_index`` (the only trained step of this branch); the
    scoring fields carry the branch's final video for the reward path —
    ``reward``/``advantage`` are placeholders the trainer/M3 fill (the reward
    model is NOT called inside the rollout).

    Tensor fields are typed ``Any`` to keep this module torch-free (it is also
    imported by the pure-Python ledger). ``sample_id`` is the index WITHIN the
    ``(prompt_id, branch_timestep_index)`` group — not a global DataProto row.
    ``branch_row_id`` is DEBUG-ONLY: the rollout assigns it worker-locally so it
    collides across sharded workers — DataProto alignment is by post-concat row
    position, and grouping is by the GLOBAL ``prompt_id`` (see
    compute_branched_advantage).
    """

    # --- identity / grouping (explicit; never inferred from a tensor axis) ---
    prompt_id: int
    branch_timestep_index: int  # global denoise step k that was branched
    sample_id: int  # index within the (prompt_id, k) group of SDE branches
    timestep_value: int  # int(sigma[k]*1000), the Wan model timestep — don't re-derive

    # --- the trained SDE transition at step k (actor recompute + PPO ratio) ---
    old_log_prob: Any  # rollout log-prob of the SDE step at k
    latent_k: Any  # x_k (input to the actor's new-logprob recompute)
    next_latent_k: Any  # x_{k+1}, the realized SDE sample
    prev_sample_mean: Any = None  # SDE step mean at k (needed when RatioNorm is on)
    sigma_schedule: Any = None  # [T+1] schedule (recompute σ_t; align global step)
    timestep_indices: Any = None  # global-step index/map, consistent with the M1 weight index

    # --- scoring (rollout emits the branch's final video; reward filled later) ---
    video_frames: Any = None  # branch final decoded video for the reward path
    final_latent: Any = None  # x_0 (if the rollout returns latents, not frames)
    reward: Optional[float] = None  # placeholder — trainer/reward path fills
    advantage: Optional[float] = None  # placeholder — M3 (compute_branched_advantages) fills
    branch_row_id: Optional[int] = None  # optional DataProto row alignment (NOT sample_id)

    def to_record(self) -> BranchRecord:
        """Project to the ledger record used by ``compute_branched_advantages``."""
        return BranchRecord(
            prompt_id=self.prompt_id,
            branch_timestep_index=self.branch_timestep_index,
            sample_id=self.sample_id,
            reward=0.0 if self.reward is None else float(self.reward),
            advantage=self.advantage,
        )


def compute_branched_advantages(
    records: Sequence[BranchRecord],
    plan: BranchPlan,
    *,
    min_group_std: float = 0.0,
    eps: float = 1e-8,
) -> tuple[list[BranchRecord], dict]:
    """Group-relative advantage (Eq. 3) per branch point.

    The group is the ``G`` samples of one prompt at a given branch timestep —
    ``(prompt_id, branch_timestep_index)`` — so credit stays localized to the
    branch (Theorem 1). Returns ``(records_with_advantage, group_boundaries)``
    where ``group_boundaries`` maps each explicit group key to the member sample
    ids used for its mean/std, so the grouping is auditable, not implicit. A
    singleton group gets advantage 0 (std undefined).

    ``std`` is the population std (``/n``) of the group — the experiment's
    deliberate choice, kept as-is. (It differs from baseline GRPO's Bessel
    ``/(n-1)`` by a constant rescale that the LR absorbs; not "fixed" to match,
    since population z-score is a valid — arguably more natural — convention for a
    fixed N-sample group.) ``min_group_std`` mirrors the baseline floor: a group
    whose reward std is below it carries no usable ranking signal (the z-score
    would amplify float noise into ±O(1) random-sign advantages — the freeze
    mechanism), so its advantages are zeroed. Default 0.0 = off.
    """
    groups: dict = {}
    for rec in records:
        groups.setdefault((rec.prompt_id, rec.branch_timestep_index), []).append(rec)

    boundaries: dict = {}
    out: list[BranchRecord] = []
    for key, members in groups.items():
        rewards = [m.reward for m in members]
        n = len(rewards)
        mean = sum(rewards) / n
        if n > 1:
            # population std (/n) — the experiment's deliberate choice; differs
            # from baseline GRPO's Bessel /(n-1) only by a constant rescale.
            var = sum((r - mean) ** 2 for r in rewards) / n
            std = var**0.5
        else:
            std = 0.0
        frozen = std <= eps or (min_group_std > 0.0 and std < min_group_std)
        for m in members:
            adv = 0.0 if frozen else (m.reward - mean) / (std + eps)
            out.append(
                BranchRecord(
                    prompt_id=m.prompt_id,
                    branch_timestep_index=m.branch_timestep_index,
                    sample_id=m.sample_id,
                    reward=m.reward,
                    advantage=adv,
                )
            )
        boundaries[key] = {
            "sample_ids": [m.sample_id for m in members],
            "n": n,
            "reward_mean": mean,
            "reward_std": std,
        }
    return out, boundaries


# --- trainer-side advantage (M3a) ---------------------------------------------
# ``torch`` is imported lazily inside the function so the ledger above stays
# torch-free (it is unit-tested as pure Python, like bgpo's helpers). The work
# lives in a module function so it is testable without a trainer instance; the
# The thin trainer hook lives in teleboost.programs.wan.tempflow.


def compute_branched_advantage(data, *, min_group_std: float = 0.0):
    """TempFlow trajectory-branching advantage (Eq. 3 / Theorem 1).

    Each DataProto row is ONE branch; group by ``(prompt_id,
    branch_timestep_index)`` — ``prompt_id`` is the driver-stamped GLOBAL id —
    z-score within the group, and scatter back BY POST-CONCAT ROW POSITION
    (``adv[i]`` for row ``i``), NOT by ``branch_row_id`` (the rollout assigns it
    worker-locally so it collides across the sharded workers). Reuses
    :func:`compute_branched_advantages` so the grouping math is single-sourced.
    Runs ONLY when branching is enabled; the normal GRPO advantage path is
    untouched.
    """
    import torch

    ntb = data.non_tensor_batch
    rewards = data.batch["rewards"].reshape(-1)
    n = int(rewards.shape[0])
    prompt_id = ntb["prompt_id"]  # globally unique per prompt instance
    k = ntb["branch_timestep_index"]
    # Scatter target is the current row position ``i``: after the sharded rollout
    # DataProtos are concatenated (and possibly reordered/balanced),
    # rewards[i]/prompt_id[i]/k[i] all belong to the same branch, so ``adv[i]`` is
    # that row's advantage — a clean permutation by construction, immune to any
    # upstream reordering. We pass ``i`` as the ledger sample_id so the scored
    # record carries its own row position back. ``branch_row_id`` is deliberately
    # NOT used (worker-local → collides after concat); grouping keys on the
    # driver-stamped global ``prompt_id``.
    records = [BranchRecord(int(prompt_id[i]), int(k[i]), i, float(rewards[i])) for i in range(n)]
    scored, _boundaries = compute_branched_advantages(records, BranchPlan(advantage_group="prompt"), min_group_std=min_group_std)
    adv = torch.zeros(n, dtype=torch.float32)
    for r in scored:
        adv[r.sample_id] = float(r.advantage)
    data.batch["advantages"] = adv
    return data
