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
"""Paper-equation level tests for ``algorithms/grpo_guard.py``.

Paper: arxiv 2510.22319 (GRPO-Guard).

These tests pin the exact numerical formulas from the paper:

* :func:`compute_ratio_norm_bias` — Eq. 8 (RatioNorm bias + outer scale)
* :func:`compute_grad_reweight_delta` — Eq. 12 (grad-reweight δ for both forms)

If a future refactor changes the math, these tests fail before any
end-to-end run is needed.  Run with:

    pytest tests/test_grpo_guard.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from teleboost.algorithms.grpo_guard import (
    GRAD_REWEIGHT_FORMS,
    compute_grad_reweight_delta,
    compute_ratio_norm_bias,
)

# ---------------------------------------------------------------------------
# RatioNorm (Eq. 8)
# ---------------------------------------------------------------------------


def test_ratio_norm_bias_zero_diff_is_zero():
    """Δμ_new == Δμ_old → bias = 0 (sanity)."""
    x = torch.randn(4, 3, 8, 8)
    bias, scale, sqrt_dt_scalar = compute_ratio_norm_bias(
        x,
        x.clone(),
        sqrt_dt=torch.tensor(0.1),
        std_dev_t=torch.tensor(0.05),
        eps=1e-6,
    )
    assert bias.shape == (4,)
    assert torch.allclose(bias, torch.zeros(4), atol=1e-9)


def test_ratio_norm_bias_matches_inline_formula():
    """compute_ratio_norm_bias should match the pre-refactor inline formula
    used in dp_actor.py prior to commit 912f0e32 byte-for-byte.

    Inline formula (the legacy code path):
        diff_squared       = (new - old).pow(2)
        ratio_mean_bias    = diff_squared.flatten(start_dim=1).mean(dim=1)
        sqrt_dt_scalar     = sqrt_dt.mean()  if ndim>0 else sqrt_dt
        std_dev_t_scalar   = std_dev_t.mean() if ndim>0 else std_dev_t
        sigma_t            = std_dev_t_scalar / (sqrt_dt_scalar + eps)
        scale              = sqrt_dt_scalar * sigma_t
        ratio_mean_bias    = ratio_mean_bias / (2 * (scale**2 + eps))
    """
    torch.manual_seed(0)
    new_mu = torch.randn(2, 3, 4, 4)
    old_mu = torch.randn(2, 3, 4, 4)
    sqrt_dt = torch.tensor(0.2)
    std_dev_t = torch.tensor(0.1)
    eps = 1e-6

    # reference (inline)
    diff_squared = (new_mu - old_mu).pow(2)
    ref_bias = diff_squared.flatten(start_dim=1).mean(dim=1).flatten()
    sqrt_dt_scalar = sqrt_dt
    std_dev_t_scalar = std_dev_t
    sigma_t = std_dev_t_scalar / (sqrt_dt_scalar + eps)
    ref_scale = sqrt_dt_scalar * sigma_t
    ref_bias = ref_bias / (2 * (ref_scale**2 + eps))

    # under test
    bias, scale, sqrt_dt_out = compute_ratio_norm_bias(
        new_mu,
        old_mu,
        sqrt_dt,
        std_dev_t,
        eps=eps,
    )

    assert torch.allclose(bias, ref_bias, atol=0.0, rtol=0.0)  # byte-identical
    assert torch.allclose(torch.as_tensor(scale), torch.as_tensor(ref_scale), atol=0.0, rtol=0.0)
    assert torch.allclose(torch.as_tensor(sqrt_dt_out), sqrt_dt, atol=0.0, rtol=0.0)


def test_ratio_norm_bias_scalar_input():
    """1-D ndim==1 input (no batch dim) should still produce a 1-D bias."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    bias, scale, _ = compute_ratio_norm_bias(
        x,
        y,
        sqrt_dt=torch.tensor(0.1),
        std_dev_t=torch.tensor(0.05),
        eps=1e-6,
    )
    assert bias.ndim == 1


# ---------------------------------------------------------------------------
# Grad-reweight δ (Eq. 12)
# ---------------------------------------------------------------------------


def test_grad_reweight_flow_grpo_is_inv_dt():
    """δ = 1/dt for the flow_grpo form (paper says β ≈ const).

    Tolerance is float32-precision-sized: dt arrives as a float32 tensor in
    actual training (the SDE schedule lives in float32), so we compare with
    rel_tol=1e-5 rather than the float64 ideal of 1e-9.
    """
    dt = torch.tensor(0.1)
    delta = compute_grad_reweight_delta("flow_grpo", t=0.5, dt=dt, eta=0.25, eps=1e-6)
    expected = 1.0 / (0.1 + 1e-6)
    assert math.isclose(float(delta), expected, rel_tol=1e-5)


def test_grad_reweight_dancegrpo_at_t_05_eta_025():
    """Pin the dancegrpo form at a known operating point.

    β = 1 + η²·(1−t)/(2t) at t=0.5, η=0.25:
        β = 1 + 0.0625 · 0.5 / 1.0 = 1 + 0.03125 = 1.03125
    δ = β / dt = 1.03125 / dt   (modulo eps)
    """
    dt = 0.1
    eta = 0.25
    eps = 1e-6
    delta = compute_grad_reweight_delta("dancegrpo", t=0.5, dt=dt, eta=eta, eps=eps)
    expected = 1.03125 / (dt + eps)
    assert math.isclose(float(delta), expected, rel_tol=1e-9)


def test_grad_reweight_dancegrpo_at_t_one_reduces_to_inv_dt():
    """At t=1, (1−t) = 0 → β = 1 → δ = 1/dt (matches flow_grpo form)."""
    dt = 0.05
    eta = 0.3
    eps = 1e-6
    delta_dance = compute_grad_reweight_delta("dancegrpo", t=1.0, dt=dt, eta=eta, eps=eps)
    delta_flow = compute_grad_reweight_delta("flow_grpo", t=1.0, dt=dt, eta=eta, eps=eps)
    assert math.isclose(float(delta_dance), float(delta_flow), rel_tol=1e-9)


def test_grad_reweight_dancegrpo_t_floor_eps():
    """t below eps should be clamped to eps so δ is finite."""
    dt = 0.1
    eta = 0.25
    eps = 1e-3  # use a larger eps so we can verify the floor numerically
    # t below eps: should be clamped to eps
    delta_below = compute_grad_reweight_delta("dancegrpo", t=1e-9, dt=dt, eta=eta, eps=eps)
    # t exactly at eps: should produce same result
    delta_at = compute_grad_reweight_delta("dancegrpo", t=eps, dt=dt, eta=eta, eps=eps)
    # both should be finite and equal (floor active in both)
    assert math.isfinite(float(delta_below))
    assert math.isfinite(float(delta_at))
    assert math.isclose(float(delta_below), float(delta_at), rel_tol=1e-9)


def test_grad_reweight_form_invalid_raises():
    with pytest.raises(ValueError, match="unknown grad_reweight_form"):
        compute_grad_reweight_delta("not_a_real_form", t=0.5, dt=0.1, eta=0.25, eps=1e-6)


def test_grad_reweight_forms_registry_has_both():
    assert "flow_grpo" in GRAD_REWEIGHT_FORMS
    assert "dancegrpo" in GRAD_REWEIGHT_FORMS
    assert callable(GRAD_REWEIGHT_FORMS["flow_grpo"])
    assert callable(GRAD_REWEIGHT_FORMS["dancegrpo"])


def test_grad_reweight_tensor_t_input():
    """δ should accept tensor t (the actor passes t_t.float().mean() which
    is a 0-d tensor)."""
    t = torch.tensor(0.5)
    dt = torch.tensor(0.1)
    delta = compute_grad_reweight_delta("dancegrpo", t=t, dt=dt, eta=0.25, eps=1e-6)
    expected = 1.03125 / (0.1 + 1e-6)
    assert math.isclose(float(delta), expected, rel_tol=1e-6)
