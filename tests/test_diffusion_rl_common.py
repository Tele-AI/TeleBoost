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
import pytest
import torch

from teleboost.algorithms.grpo.loss import grpo_policy_loss
from teleboost.algorithms.grpo.policy_scalars import (
    assert_per_sample_shape,
    assert_same_shape,
    extract_per_sample_constant,
    reduce_log_density,
)

# ----------------------------- grpo_policy_loss --------------------------------


def _wan_inline_policy_loss(advantages, ratio, clip_range):
    """The exact pre-refactor dp_actor inline computation."""
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.mean(torch.maximum(unclipped_loss, clipped_loss))


def test_grpo_policy_loss_is_bit_identical_to_wan_inline():
    torch.manual_seed(0)
    for _ in range(5):
        adv = torch.randn(8)
        ratio = torch.exp(0.01 * torch.randn(8))  # near 1
        clip = 0.2
        ref = _wan_inline_policy_loss(adv, ratio, clip)
        got, _ = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=clip)
        # byte-for-byte: same ops, same order
        assert torch.equal(got, ref), f"diff={(got - ref).abs().item()}"


def test_grpo_policy_loss_ratio_from_logprobs_matches():
    torch.manual_seed(1)
    log_prob = torch.randn(6)
    old = torch.randn(6)
    adv = torch.randn(6)
    ratio = torch.exp(log_prob - old)
    ref = _wan_inline_policy_loss(adv, ratio, 0.2)
    got, m = grpo_policy_loss(advantage=adv, log_prob=log_prob, old_log_prob=old, clip_range=0.2)
    assert torch.equal(got, ref)
    assert "approx_kl" in m and "clipfrac" in m


def test_grpo_policy_loss_rejects_cross_broadcast_shapes():
    # The P0 bug: an EXPANDING broadcast (result bigger than ratio) cross-mixes
    # samples and must be rejected. [4] × [4,1,1,1,1] -> [4,1,1,1,4] expands.
    adv = torch.randn(4)
    ratio = torch.randn(4, 1, 1, 1, 1)
    with pytest.raises(AssertionError, match="expands beyond|broadcast INTO"):
        grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2)


def test_grpo_policy_loss_allows_per_sample_scalar_advantage_broadcast():
    # Real Wan micro-batch=1 case: advantage [1] weights all of one sample's
    # per-step ratios [N]. This is a broadcast INTO ratio (result == ratio
    # shape), legitimate, and bit-identical to the old `-adv * ratio`.
    adv = torch.tensor([0.7])
    ratio = torch.exp(0.01 * torch.randn(16))
    got, _ = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2)
    ref = _wan_inline_policy_loss(adv, ratio, 0.2)
    assert torch.equal(got, ref)


def test_grpo_policy_loss_timestep_weight_none_is_unweighted():
    adv = torch.randn(5)
    ratio = torch.exp(0.01 * torch.randn(5))
    base, _ = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2)
    w = torch.full((5,), 1.0)
    weighted, _ = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2, timestep_weight=w)
    assert torch.allclose(base, weighted)
    w2 = torch.full((5,), 2.0)
    weighted2, _ = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2, timestep_weight=w2)
    assert torch.allclose(weighted2, 2.0 * base)


def test_grpo_policy_loss_kl_term_added_when_beta_positive():
    adv = torch.randn(4)
    ratio = torch.exp(0.01 * torch.randn(4))
    kl = torch.full((4,), 0.5)
    base, _ = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2)
    withkl, m = grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2, kl=kl, beta=0.1)
    assert torch.allclose(withkl, base + 0.1 * 0.5)
    assert m["kl_loss"] == pytest.approx(0.5)


def test_grpo_policy_loss_nan_guard():
    adv = torch.tensor([1.0, float("nan")])
    ratio = torch.tensor([1.0, 1.0])
    with pytest.raises(FloatingPointError):
        grpo_policy_loss(advantage=adv, ratio=ratio, clip_range=0.2)


# ----------------------------- policy_scalars ----------------------------------


def test_reduce_log_density_mean_and_sum():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.allclose(reduce_log_density(x, reduction="mean"), x.mean(dim=(1, 2)))
    assert torch.allclose(reduce_log_density(x, reduction="sum"), x.sum(dim=(1, 2)))
    # idempotent on [B]
    b = torch.randn(5)
    assert torch.equal(reduce_log_density(b), b)


def test_extract_per_sample_constant_exact_for_broadcast_std():
    # std as [B,1,1,1,1] broadcast to a latent shape -> per-sample constant
    std_b = torch.tensor([0.3, 0.7, 1.1])
    dense = std_b.view(3, 1, 1, 1, 1).expand(3, 4, 2, 5, 5)
    out = extract_per_sample_constant(dense.contiguous())
    assert torch.allclose(out, std_b)


def test_extract_per_sample_constant_rejects_nonuniform():
    bad = torch.randn(3, 4, 4)  # genuinely varies across non-batch dims
    with pytest.raises(ValueError, match="not per-sample-constant"):
        extract_per_sample_constant(bad)


def test_assert_per_sample_shape_and_same_shape():
    b = torch.randn(4)
    assert_per_sample_shape(4, b, b)
    with pytest.raises(AssertionError):
        assert_per_sample_shape(4, torch.randn(4, 1))
    assert_same_shape(b, b.clone())
    with pytest.raises(AssertionError):
        assert_same_shape(b, torch.randn(4, 1))


# ---------------------------- rollout_contract ---------------------------------


def _good_record():
    return {
        "prompt_id": 0,
        "sample_id": 0,
        "timestep_index": 3,
        "timestep_value": 0.6,
        "latent_t": None,
        "latent_next": None,
        "old_log_prob": 0.0,
        "reward": 1.0,
        "advantage": 0.5,
        "logprob_reduction": "mean",
        "solver_id": "euler_flow",
        "sigma_form": "flow_grpo",
        "policy_forward_kind": "sft_forward",
    }


def test_validate_rollout_record_accepts_good_and_rejects_bad():
    from teleboost.algorithms.rollout_contract import validate_rollout_record

    validate_rollout_record(_good_record())  # ok

    missing = _good_record()
    del missing["solver_id"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_rollout_record(missing)

    bad_reduction = _good_record()
    bad_reduction["logprob_reduction"] = "rms"
    with pytest.raises(ValueError, match="logprob_reduction"):
        validate_rollout_record(bad_reduction)

    # the load-bearing one: inference forward forbidden as policy log-prob source
    inference = _good_record()
    inference["policy_forward_kind"] = "generate_image"
    with pytest.raises(ValueError, match="TRAINING forward"):
        validate_rollout_record(inference)
