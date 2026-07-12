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
"""Spec for compute_branched_advantage (trainer side, M3a).

Per-(prompt, branch step) group z-score, scattered back BY ROW POSITION (adv[i]
is row i's advantage). The grouping key is the driver-stamped GLOBAL ``prompt_id``
— NOT ``branch_row_id``, which the rollout assigns worker-locally and therefore
collides across the sharded GPU workers. Uses a tiny stub DataProto (the function
only reads data.batch["rewards"], data.non_tensor_batch[...], and sets
data.batch["advantages"]).
"""

import numpy as np
import torch

from teleboost.algorithms.tempflow.trajectory import compute_branched_advantage


class _Stub:
    def __init__(self, rewards, prompt_id, k, sample_id, branch_row_id):
        self.batch = {"rewards": torch.tensor(rewards, dtype=torch.float32)}
        self.non_tensor_batch = {
            "prompt_id": np.array(prompt_id),
            "branch_timestep_index": np.array(k),
            "sample_id": np.array(sample_id),
            "branch_row_id": np.array(branch_row_id),
        }


def _zscore(vals):
    m = sum(vals) / len(vals)
    # population std (/n) — matches compute_branched_advantages.
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    s = var**0.5
    return [0.0 if s <= 1e-8 else (v - m) / (s + 1e-8) for v in vals]


def test_group_zscore_per_prompt_and_step():
    # prompt 0 @k=3 : rewards 0.2/0.5/0.8 ; prompt 1 @k=3 : 0.1/0.1/0.7
    rewards = [0.2, 0.5, 0.8, 0.1, 0.1, 0.7]
    pid = [0, 0, 0, 1, 1, 1]
    k = [3, 3, 3, 3, 3, 3]
    sid = [0, 1, 2, 0, 1, 2]
    row = [0, 1, 2, 3, 4, 5]
    out = compute_branched_advantage(_Stub(rewards, pid, k, sid, row))
    adv = out.batch["advantages"].tolist()
    exp = _zscore(rewards[:3]) + _zscore(rewards[3:])
    assert all(abs(a - e) < 1e-4 for a, e in zip(adv, exp, strict=False)), (adv, exp)


def test_same_prompt_different_step_are_separate_groups():
    # prompt 0 at k=2 and k=5 must NOT be pooled
    rewards = [0.2, 0.8, 0.3, 0.9]
    pid = [0, 0, 0, 0]
    k = [2, 2, 5, 5]
    sid = [0, 1, 0, 1]
    row = [0, 1, 2, 3]
    adv = compute_branched_advantage(_Stub(rewards, pid, k, sid, row)).batch["advantages"].tolist()
    # each k-group is its own 2-sample z-score: [-1,1] then [-1,1]
    assert adv[0] < 0 < adv[1] and adv[2] < 0 < adv[3]
    assert abs(adv[0] + adv[1]) < 1e-4 and abs(adv[2] + adv[3]) < 1e-4


def test_scatter_is_by_row_position_and_ignores_local_branch_row_id():
    # Rows from TWO prompt groups are interleaved, AND branch_row_id collides
    # (all 0) exactly like the worker-local ids do after the sharded rollout is
    # concatenated. The advantage must group by global prompt_id and land at the
    # row's POSITION — proving branch_row_id is not used as the scatter target.
    rewards = [0.8, 0.1, 0.2, 0.7]  # rows: p0(0.8), p1(0.1), p0(0.2), p1(0.7)
    pid = [0, 1, 0, 1]
    k = [4, 4, 4, 4]
    sid = [0, 0, 1, 1]
    row = [0, 0, 0, 0]  # colliding worker-local ids — MUST be ignored
    adv = compute_branched_advantage(_Stub(rewards, pid, k, sid, row)).batch["advantages"].tolist()
    z0 = _zscore([0.8, 0.2])  # prompt 0 group
    z1 = _zscore([0.1, 0.7])  # prompt 1 group
    assert abs(adv[0] - z0[0]) < 1e-4 and abs(adv[2] - z0[1]) < 1e-4
    assert abs(adv[1] - z1[0]) < 1e-4 and abs(adv[3] - z1[1]) < 1e-4


def test_singleton_group_zero_advantage():
    adv = compute_branched_advantage(_Stub([0.5], [0], [3], [0], [0])).batch["advantages"].tolist()
    assert adv == [0.0]
