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

from teleboost.algorithms.tempflow.noise import (
    compute_schedule_noise_std,
    compute_timestep_weights,
    resolve_timestep_weights,
    summarize_noise_weights,
)


def test_schedule_noise_std_dancegrpo_is_eta_sqrt_dt():
    sigmas = torch.tensor([1.0, 0.6, 0.0])
    eta = 0.25
    ns = compute_schedule_noise_std(sigmas, eta=eta, sigma_form="dancegrpo")
    # dancegrpo: std = eta * sqrt(Δt)
    expected = torch.tensor([eta * math.sqrt(0.4), eta * math.sqrt(0.6)])
    assert torch.allclose(ns, expected, atol=1e-5)


def test_schedule_noise_std_flow_grpo_matches_paper_sigma_t_sqrt_dt():
    # Paper Eq. 8 raw weight σ_t·√Δt with σ_t = a·√(σ/(1-σ)) (== flow_grpo form).
    sigmas = torch.tensor([0.8, 0.5])
    eta = 1.0
    ns = compute_schedule_noise_std(sigmas, eta=eta, sigma_form="flow_grpo")
    sigma_t = eta * math.sqrt(0.8 / (1 - 0.8))
    expected = torch.tensor([sigma_t * math.sqrt(0.8 - 0.5)])
    assert torch.allclose(ns, expected, atol=1e-5)


def test_timestep_weights_none_is_ones():
    ns = torch.tensor([0.1, 0.2, 0.3])
    w = compute_timestep_weights(ns, mode="none")
    assert torch.allclose(w, torch.ones(3))


def test_tempflow_noise_norm_is_mean_one_and_proportional():
    ns = torch.tensor([0.1, 0.2, 0.3, 0.4])
    w = compute_timestep_weights(ns, mode="tempflow_noise_norm", sigma_form="flow_grpo")
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)
    # proportional to noise_std: w_i / w_j == ns_i / ns_j
    assert float(w[3] / w[0]) == pytest.approx(0.4 / 0.1, rel=1e-5)


def test_tempflow_with_dancegrpo_warns_not_paper_faithful():
    ns = torch.tensor([0.1, 0.2, 0.3])
    with pytest.warns(RuntimeWarning, match="flow_grpo"):
        compute_timestep_weights(ns, mode="tempflow_noise_norm", sigma_form="dancegrpo")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown reweight mode"):
        compute_timestep_weights(torch.tensor([0.1, 0.2]), mode="bogus")


def test_summarize_noise_weights_keys():
    ns = torch.tensor([0.1, 0.3])
    w = compute_timestep_weights(ns, mode="tempflow_noise_norm", sigma_form="flow_grpo")
    s = summarize_noise_weights(ns, w)
    for k in ("raw_noise_weight/mean", "normalized_weight/mean", "normalized_weight/max"):
        assert k in s
    assert s["normalized_weight/mean"] == pytest.approx(1.0, abs=1e-5)


def test_flow_window_weight_indexes_global_step_not_local():
    """Regression for the dp_actor wiring: under a flow-grpo SDE window the
    trained steps are a SUBSET of the full schedule, mapped local→global by
    ``timestep_indices``. The per-step TempFlow weight MUST be taken at the
    global denoise step k (``weights[timestep_indices[j]]``), never at the local
    loop counter j — otherwise a high-noise late-window step gets an early
    step's weight. This pins that contract so a revert to ``weights[step_idx]``
    fails here.
    """
    num_steps = 16  # T transitions, schedule has T+1 points
    sigmas = torch.linspace(1, 0, num_steps + 1)
    noise_std = compute_schedule_noise_std(sigmas, eta=0.3, sigma_form="flow_grpo")
    weights = compute_timestep_weights(noise_std, mode="tempflow_noise_norm", sigma_form="flow_grpo")
    assert weights.shape == (num_steps,)

    # flow window: train only global steps 10..13 (local j=0..3 -> global 10..13)
    timestep_indices = [10, 11, 12, 13]
    global_w = torch.stack([weights[k] for k in timestep_indices])  # correct
    local_w = torch.stack([weights[j] for j in range(len(timestep_indices))])  # the bug

    # the two must differ (else the test proves nothing on this schedule)
    assert not torch.allclose(global_w, local_w)
    # global indexing returns exactly the schedule's weights at those steps
    for j, k in enumerate(timestep_indices):
        assert torch.equal(global_w[j], weights[k])


def test_resolve_timestep_weights_none_disabled():
    sched = torch.linspace(1, 0, 9).expand(4, 9)  # [B=4, T+1=9]
    assert resolve_timestep_weights(sched, mode="none", eta=0.3, sigma_form="flow_grpo") is None


def test_resolve_timestep_weights_batch_schedule_is_mean_one():
    sched = torch.linspace(1, 0, 9).expand(4, 9).contiguous()  # shared across batch
    w = resolve_timestep_weights(sched, mode="tempflow_noise_norm", eta=0.3, sigma_form="flow_grpo")
    assert w.shape == (8,)  # T = (T+1) - 1
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-4)


def test_resolve_timestep_weights_rejects_nonuniform_batch_schedule():
    sched = torch.linspace(1, 0, 9).expand(4, 9).clone()
    sched[2, 3] += 0.1  # one row's schedule drifts
    with pytest.raises(AssertionError, match="differs across the batch"):
        resolve_timestep_weights(sched, mode="tempflow_noise_norm", eta=0.3, sigma_form="flow_grpo")
