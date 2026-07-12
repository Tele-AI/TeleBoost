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
import torch

from teleboost.training.families.wan.actor import DiffusionDataParallelPPOActor


def test_record_mismatch_metrics_matches_hand_computed_values():
    metrics = {}
    log_prob_delta = torch.tensor([0.0, math.log(2.0), math.log(0.5)])
    ratio = torch.exp(log_prob_delta)
    clipped_mask = torch.tensor([False, True, True])
    timestep = torch.tensor([900, 900, 900])

    DiffusionDataParallelPPOActor._record_mismatch_metrics(
        metrics,
        prefix="mismatch/grpo",
        log_prob_delta=log_prob_delta,
        ratio=ratio,
        clipped_mask=clipped_mask,
        timestep=timestep,
    )

    weights = ratio
    expected_ess = weights.sum().square() / weights.square().sum() / weights.numel()

    assert metrics["mismatch/grpo/logprob_diff_mean"] == [torch.mean(log_prob_delta).item()]
    assert metrics["mismatch/grpo/logprob_diff_abs_mean"] == [torch.mean(log_prob_delta.abs()).item()]
    assert metrics["mismatch/grpo/ratio_mean"] == [torch.mean(ratio).item()]
    assert metrics["mismatch/grpo/chi2"] == [torch.mean((ratio - 1.0).square()).item()]
    assert metrics["mismatch/grpo/ess"] == [expected_ess.item()]
    # clipped_frac is a float32 tensor mean (0.66666669), so compare with a
    # tolerance rather than the float64 literal 2/3.
    assert metrics["mismatch/grpo/clipped_frac"][0] == pytest.approx(2.0 / 3.0)
    assert metrics["mismatch/grpo/timestep"] == [900.0]
