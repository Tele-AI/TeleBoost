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
"""Rollout <-> actor transition parity.

The SDE step and the Gaussian log-prob density are hand-duplicated between
``DiffusionRollout.wan_step`` (generation) and
``DiffusionDataParallelPPOActor.wan_step`` (training recompute). The GRPO
importance ratio ``exp(new - old)`` is only valid while the two stay
numerically IDENTICAL — nothing in production enforces that (the SolverContract
transition strings are hardcoded equal on both sides), so this test does:
any edit that lands on one copy but not the other fails here, loudly.

Both methods only touch ``self._sigma_form`` and a pixel-mode predicate
(``_pixel_enabled()`` on the actor, ``_pixel_enable`` on the rollout), so they
are exercised unbound with duck-typed selves — no worker construction needed.
"""

from types import SimpleNamespace

import pytest
import torch

from teleboost.training.families.wan.actor import DiffusionDataParallelPPOActor
from teleboost.training.families.wan.rollout import DiffusionRollout


def _selves(sigma_form, pixel):
    actor = SimpleNamespace(_sigma_form=sigma_form, _pixel_enabled=lambda: pixel)
    rollout = SimpleNamespace(_sigma_form=sigma_form, _pixel_enable=pixel)
    return actor, rollout


def _inputs(seed, steps=8):
    g = torch.Generator().manual_seed(seed)
    shape = (16, 3, 8, 8)
    return (
        torch.randn(shape, generator=g, dtype=torch.float64),  # model_output
        torch.randn(shape, generator=g, dtype=torch.float64),  # latents
        torch.randn(shape, generator=g, dtype=torch.float64),  # prev_sample
        torch.linspace(1.0, 0.05, steps, dtype=torch.float64),  # sigmas
    )


@pytest.mark.parametrize("sigma_form", ["dancegrpo", "flow_grpo"])
@pytest.mark.parametrize("index", [0, 3, 6])
@pytest.mark.parametrize("pixel", [False, True])
def test_grpo_recompute_matches_rollout(sigma_form, index, pixel):
    model_output, latents, prev_sample, sigmas = _inputs(seed=42 + index)
    eta = 0.3
    actor_self, rollout_self = _selves(sigma_form, pixel)

    a_prev, a_x0, a_logp, a_mean, _a_std, _a_sqdt = DiffusionDataParallelPPOActor.wan_step(
        actor_self,
        model_output,
        latents,
        eta,
        sigmas,
        index,
        prev_sample=prev_sample,
        grpo=True,
        return_stats=True,
    )
    r_prev, r_x0, r_logp, r_mean = DiffusionRollout.wan_step(
        rollout_self,
        model_output,
        latents,
        eta,
        sigmas,
        index,
        prev_sample=prev_sample,
        grpo=True,
        return_prev_sample_mean=True,
    )

    # Exact equality, not allclose: the two bodies claim to be the same math on
    # the same inputs. Any tolerance here would hide exactly the drift this
    # test exists to catch.
    assert torch.equal(a_x0, r_x0), "pred_original_sample drifted"
    assert torch.equal(a_mean, r_mean), "prev_sample_mean drifted"
    assert torch.equal(a_prev, r_prev), "prev_sample changed"
    assert torch.equal(a_logp, r_logp), "log_prob drifted (max |d|={:.3e})".format((a_logp.double() - r_logp.double()).abs().max().item())
    assert a_logp.shape == r_logp.shape, "log_prob reduction shape drifted"
