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
"""Unit tests for the pure functions extracted into ``teleboost/algorithms/``.

These cover the refactor's contract surface: ``compute_joint_advantage_weights``,
``binary_rerange_group_rewards``, and ``bayes_reliability_weight``. They run
CPU-only and need ``torch`` but not a GPU.

Run from the repo root:

    pytest tests/test_teleboost_algorithms.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from teleboost.algorithms.bgpo import bayes_reliability_weight, binary_rerange_group_rewards
from teleboost.training.rewarding.joint_advantage_weights import compute_joint_advantage_weights

# ---------------------------------------------------------------------------
# compute_joint_advantage_weights
# ---------------------------------------------------------------------------


class TestComputeJointAdvantageWeights:
    def test_empty_input_returns_zeros(self):
        out = compute_joint_advantage_weights(torch.empty(0, 3))
        assert out.shape == (0, 3)
        assert out.numel() == 0

    def test_single_sample_straddling_zero(self):
        # One sample, advantages span [-2, 1, 3]. min=-2 (idx 0), max=3 (idx 2).
        # t = -(-2) / (3 - (-2)) = 0.4. So c[0] = 0.6, c[2] = 0.4, c[1] = 0.
        adv = torch.tensor([[-2.0, 1.0, 3.0]])
        weights = compute_joint_advantage_weights(adv)
        assert weights.shape == (1, 3)
        assert torch.allclose(weights[0], torch.tensor([0.6, 0.0, 0.4]), atol=1e-6)

    def test_all_positive_picks_argmin(self):
        # All advantages > 0 -> single-mass at the smallest one.
        adv = torch.tensor([[1.0, 2.0, 3.0]])
        weights = compute_joint_advantage_weights(adv)
        assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0]))

    def test_all_negative_picks_argmax(self):
        adv = torch.tensor([[-3.0, -1.0, -2.0]])
        weights = compute_joint_advantage_weights(adv)
        # argmax = idx 1 (value -1.0)
        assert torch.allclose(weights[0], torch.tensor([0.0, 1.0, 0.0]))

    def test_uniform_when_max_equals_min(self):
        # All-zero advantages: min == max == 0, near-zero -> uniform 1/n.
        adv = torch.zeros(1, 4)
        weights = compute_joint_advantage_weights(adv)
        assert torch.allclose(weights[0], torch.full((4,), 0.25))

    def test_per_row_independence(self):
        # Mix three independent rows; weights should be computed per-row.
        adv = torch.tensor(
            [
                [-1.0, 1.0],  # straddle
                [2.0, 5.0],  # all positive
                [-4.0, -1.0],  # all negative
            ]
        )
        weights = compute_joint_advantage_weights(adv)
        # Row 0: t = 1/2 -> [0.5, 0.5]
        assert torch.allclose(weights[0], torch.tensor([0.5, 0.5]))
        # Row 1: argmin = 0 -> [1, 0]
        assert torch.allclose(weights[1], torch.tensor([1.0, 0.0]))
        # Row 2: argmax = 1 -> [0, 1]
        assert torch.allclose(weights[2], torch.tensor([0.0, 1.0]))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            compute_joint_advantage_weights(torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# binary_rerange_group_rewards (BGPO CRT branch — binary prior-threshold)
# ---------------------------------------------------------------------------
#   R̃ = [a·(R − R_prior) + 𝟙{R > R_prior}] / (1 + exp(−R/τ)) · R


class TestBinaryRerangeGroupRewards:
    def test_above_prior_matches_formula(self):
        # R=1.0, prior=0.5, a=50, τ=5 → numerator = 50·0.5 + 1 = 26,
        # denom = 1 + exp(-1.0/5), output = 26/denom · 1.0
        out = binary_rerange_group_rewards(torch.tensor([1.0]), prior=0.5, a=50.0, temperature=5.0)
        expected = (50 * 0.5 + 1.0) / (1.0 + math.exp(-1.0 / 5.0)) * 1.0
        assert torch.allclose(out, torch.tensor([expected]), rtol=1e-5)

    def test_below_prior_is_negative(self):
        # R=0.1, prior=0.5, a=50 → flag=-0.4, sign=0, numerator=-20
        out = binary_rerange_group_rewards(torch.tensor([0.1]), prior=0.5, a=50.0, temperature=5.0)
        expected = (50 * -0.4 + 0.0) / (1.0 + math.exp(-0.1 / 5.0)) * 0.1
        assert torch.allclose(out, torch.tensor([expected]), rtol=1e-5)
        assert out.item() < 0.0

    def test_at_prior_numerator_zero(self):
        # R == prior → flag=0, clamp(sign(0),min=0)=0 → numerator=0 → output=0
        out = binary_rerange_group_rewards(torch.tensor([0.5]), prior=0.5, a=50.0, temperature=5.0)
        assert out.item() == pytest.approx(0.0, abs=1e-7)

    def test_clamps_extreme_reward(self):
        out = binary_rerange_group_rewards(torch.tensor([1e6]), prior=0.0, a=1.0, temperature=5.0)
        assert torch.isfinite(out).all()

    def test_preserves_dtype_and_device(self):
        r = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        out = binary_rerange_group_rewards(r, prior=0.5, a=10.0, temperature=5.0)
        assert out.dtype == r.dtype
        assert out.device == r.device


# ---------------------------------------------------------------------------
# bayes_reliability_weight (BGPO RAS branch — Bayesian posterior gain)
# ---------------------------------------------------------------------------
#   Returns centered w−1; base w ∈ [w_min, w_max] via P(prior < posterior mean).
#   The trainer's ``advantage *= clamp(1 + α·w_centered, ...)`` reconstructs w.


class TestBayesReliabilityWeight:
    def test_above_prior_is_positive(self):
        # group posterior >> prior → prob_better high → base w>1 → centered>0
        w = bayes_reliability_weight(torch.tensor([0.8, 0.9, 1.0]), prior=0.0)
        assert w > 0.0

    def test_below_prior_is_negative(self):
        w = bayes_reliability_weight(torch.tensor([-0.8, -0.9, -1.0]), prior=0.0)
        assert w < 0.0

    def test_centered_stays_in_weight_range(self):
        # base w ∈ [w_min, w_max] → centered ∈ [w_min−1, w_max−1]
        for rewards in (torch.tensor([0.9, 1.0, 1.1]), torch.tensor([-0.9, -1.0, -1.1])):
            w = bayes_reliability_weight(rewards, prior=0.0, weight_range=(0.5, 1.5))
            assert -0.5 <= w <= 0.5

    def test_explicit_posterior_formula(self):
        # rewards mean 5, unbiased var 1; prior 0, prior_var 1:
        # posterior_var = 1/(3/1 + 1/1) = 0.25, posterior_mean = 0.25·(3·5) = 3.75,
        # z = (0 − 3.75)/0.5 = −7.5 → prob_better ≈ 1 → base w ≈ 1.5 → centered ≈ 0.5
        w = bayes_reliability_weight(torch.tensor([4.0, 5.0, 6.0]), prior=0.0, prior_var=1.0, weight_range=(0.5, 1.5))
        assert w == pytest.approx(0.5, abs=0.02)

    def test_custom_weight_range(self):
        w = bayes_reliability_weight(torch.tensor([4.0, 5.0, 6.0]), prior=0.0, weight_range=(0.8, 1.2))
        assert w == pytest.approx(0.2, abs=0.02)

    def test_single_sample_no_crash(self):
        # n<=1 → sample_var defaults to 1.0, no division blow-up
        w = bayes_reliability_weight(torch.tensor([0.5]), prior=0.0)
        assert math.isfinite(w)
