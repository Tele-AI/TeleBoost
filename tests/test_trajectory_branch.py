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
import math

import pytest

from teleboost.algorithms.tempflow.trajectory import (
    BranchPlan,
    BranchRecord,
    BranchSample,
    compute_branched_advantages,
    select_branch_points,
    select_trainable_branch_points,
)


def test_branch_plan_validation():
    BranchPlan()  # defaults ok
    with pytest.raises(ValueError, match="branch_points"):
        BranchPlan(branch_points="every")
    with pytest.raises(ValueError, match="reward_target"):
        BranchPlan(reward_target="audio")
    with pytest.raises(ValueError, match="advantage_group"):
        BranchPlan(advantage_group="batch")
    with pytest.raises(ValueError, match="Def. 1"):
        BranchPlan(branch_transition="ode")
    with pytest.raises(ValueError, match="early_k"):
        BranchPlan(branch_points="early_k")
    with pytest.raises(ValueError, match="num_sampled"):
        BranchPlan(branch_points="sampled_k")


def test_select_branch_points_all_and_early_k():
    assert select_branch_points(5, BranchPlan(branch_points="all")) == [0, 1, 2, 3, 4]
    assert select_branch_points(5, BranchPlan(branch_points="early_k", early_k=2)) == [0, 1]
    # early_k larger than T clamps
    assert select_branch_points(3, BranchPlan(branch_points="early_k", early_k=9)) == [0, 1, 2]


def test_select_branch_points_sampled_requires_explicit_indices():
    plan = BranchPlan(branch_points="sampled_k", num_sampled=2)
    assert select_branch_points(8, plan, sampled_indices=[1, 5]) == [1, 5]
    with pytest.raises(ValueError, match="sampled_indices"):
        select_branch_points(8, plan)  # no indices -> deterministic-only
    with pytest.raises(ValueError, match="out of range"):
        select_branch_points(8, plan, sampled_indices=[1, 99])


def test_branched_advantages_per_prompt_group_zscore():
    plan = BranchPlan(advantage_group="prompt")
    recs = [
        BranchRecord(prompt_id=0, branch_timestep_index=3, sample_id=0, reward=1.0),
        BranchRecord(prompt_id=0, branch_timestep_index=3, sample_id=1, reward=2.0),
        BranchRecord(prompt_id=0, branch_timestep_index=3, sample_id=2, reward=3.0),
    ]
    out, boundaries = compute_branched_advantages(recs, plan)
    by_sample = {r.sample_id: r.advantage for r in out}
    # population std for [1,2,3]: var=(1+0+1)/3=2/3 -> std=sqrt(2/3)
    std = math.sqrt(2.0 / 3.0)
    assert by_sample[0] == pytest.approx(-1.0 / std, rel=1e-4)
    assert by_sample[1] == pytest.approx(0.0, abs=1e-6)
    assert by_sample[2] == pytest.approx(1.0 / std, rel=1e-4)
    # group boundary is explicit + keyed by (prompt, timestep)
    assert (0, 3) in boundaries
    assert boundaries[(0, 3)]["sample_ids"] == [0, 1, 2]
    assert boundaries[(0, 3)]["reward_mean"] == pytest.approx(2.0)


def test_branched_advantage_min_group_std_freeze():
    """A group whose reward std is below min_group_std is frozen to advantage 0 —
    the baseline GRPO freeze floor, carried onto the branch path so a low-variance
    group can't be re-inflated into random-sign advantages."""
    plan = BranchPlan(advantage_group="prompt")
    recs = [
        BranchRecord(prompt_id=0, branch_timestep_index=3, sample_id=0, reward=1.000),
        BranchRecord(prompt_id=0, branch_timestep_index=3, sample_id=1, reward=1.001),
    ]
    # std ~ 7e-4; a floor above it freezes both advantages to 0.
    out, _ = compute_branched_advantages(recs, plan, min_group_std=0.01)
    assert all(r.advantage == 0.0 for r in out)
    # without the floor (default 0.0) the tiny spread still yields +/- advantage.
    out2, _ = compute_branched_advantages(recs, plan)
    assert any(r.advantage != 0.0 for r in out2)


def test_branched_advantages_groups_are_separated_by_timestep_and_prompt():
    # Two branch points and two prompts -> 4 independent groups; advantages must
    # NOT leak across them (credit localization).
    plan = BranchPlan(advantage_group="prompt")
    recs = [
        BranchRecord(0, 1, 0, reward=0.0),
        BranchRecord(0, 1, 1, reward=10.0),  # group (0,1): huge spread
        BranchRecord(0, 5, 2, reward=5.0),
        BranchRecord(0, 5, 3, reward=5.0),  # group (0,5): zero spread -> adv 0
    ]
    out, boundaries = compute_branched_advantages(recs, plan)
    by_sample = {r.sample_id: r.advantage for r in out}
    assert by_sample[2] == 0.0 and by_sample[3] == 0.0  # std=0 group
    assert by_sample[0] != 0.0 and by_sample[1] != 0.0
    assert set(boundaries.keys()) == {(0, 1), (0, 5)}


def test_non_prompt_advantage_group_rejected():
    # Only the TempFlow paper-faithful per-prompt group is allowed.
    with pytest.raises(ValueError, match="advantage_group"):
        BranchPlan(advantage_group="global")


def test_branch_sample_carries_explicit_schema_and_projects_to_record():
    """BranchSample must carry every M3 field by NAME (no reshape-guessing), and
    project cleanly to the ledger BranchRecord."""
    s = BranchSample(
        prompt_id=2,
        branch_timestep_index=5,
        sample_id=1,  # index WITHIN the (prompt 2, k=5) group
        timestep_value=312,  # int(sigma_k*1000), carried, not re-derived
        old_log_prob="lp",  # tensors are Any here (torch-free module)
        latent_k="xk",
        next_latent_k="xk1",
        prev_sample_mean="mean",
        sigma_schedule="sched",
        timestep_indices="idx",
        video_frames="vid",
    )
    # placeholders default to None — reward path / M3 fill them
    assert s.reward is None and s.advantage is None and s.branch_row_id is None
    # all training fields present and explicit
    assert (s.prev_sample_mean, s.timestep_value, s.video_frames) == ("mean", 312, "vid")
    rec = s.to_record()
    assert isinstance(rec, BranchRecord)
    assert (rec.prompt_id, rec.branch_timestep_index, rec.sample_id) == (2, 5, 1)
    assert rec.reward == 0.0  # None reward → 0.0 placeholder for the ledger


def test_trainable_branch_points_never_include_last_step():
    """select_trainable_branch_points must cap at k ≤ sampling_steps-2 (the final
    σ→0 step is dropped by the actor, so it is not a valid branch target)."""
    sampling_steps = 16  # trainable steps = 15 (k in 0..14)
    pts = select_trainable_branch_points(sampling_steps, BranchPlan(branch_points="all"))
    assert pts == list(range(15))
    assert max(pts) == sampling_steps - 2  # 14, never 15

    early = select_trainable_branch_points(sampling_steps, BranchPlan(branch_points="early_k", early_k=2))
    assert early == [0, 1]


def test_trainable_branch_points_tiny_schedule_guard():
    with pytest.raises(ValueError):
        select_trainable_branch_points(1, BranchPlan(branch_points="all"))
