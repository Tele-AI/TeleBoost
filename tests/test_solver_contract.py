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

from teleboost.algorithms.rollout_contract import validate_solver_contract
from teleboost.algorithms.solver_contract import SolverContract


def _euler_contract(**kw):
    base = dict(
        name="euler_flow",
        sigma_form="flow_grpo",  # a registered SIGMA_FORMS key
        rollout_transition="euler_flow_sde",
        recompute_transition="euler_flow_sde",
        eval_transition="euler_flow_ode",
        logprob_reduction="mean",
    )
    base.update(kw)
    return SolverContract(**base)


def test_good_contract_passes():
    c = _euler_contract()
    assert c.transition_consistent is True
    assert c.sigma_form == "flow_grpo"


def test_bad_sigma_form_fails():
    with pytest.raises(ValueError, match="not registered|not a registered"):
        _euler_contract(sigma_form="unipc_made_up")


def test_rollout_recompute_mismatch_fails():
    # The headline footgun: UniPC rollout sigmas + Euler-SDE logprob.
    with pytest.raises(ValueError, match="recompute_transition"):
        _euler_contract(rollout_transition="unipc_multistep", recompute_transition="euler_flow_sde")


def test_transition_mismatch_allowed_only_with_explicit_flag():
    c = _euler_contract(
        rollout_transition="unipc_multistep",
        recompute_transition="euler_flow_sde",
        allow_transition_mismatch=True,
    )
    assert c.transition_consistent is False  # surfaced, not hidden


def test_eval_transition_may_be_ode_and_differ():
    # eval is a pure ODE sampler; it is NOT the policy distribution, so it is
    # allowed to differ from rollout/recompute.
    c = _euler_contract(eval_transition="euler_flow_ode")
    assert c.eval_transition == "euler_flow_ode"
    assert c.transition_consistent is True


def test_bad_logprob_reduction_fails():
    with pytest.raises(ValueError, match="logprob_reduction"):
        _euler_contract(logprob_reduction="rms")


def test_assert_matches_record_catches_divergence():
    c = _euler_contract()
    # matching record is fine
    c.assert_matches_record(solver_id="euler_flow", sigma_form="flow_grpo", logprob_reduction="mean")
    # a record claiming a different solver / sigma form must fail
    with pytest.raises(ValueError, match="violates SolverContract"):
        c.assert_matches_record(solver_id="unipc", sigma_form="flow_grpo", logprob_reduction="mean")
    with pytest.raises(ValueError, match="sigma_form"):
        c.assert_matches_record(solver_id="euler_flow", sigma_form="dancegrpo", logprob_reduction="mean")


def test_validate_solver_contract_end_to_end():
    c = _euler_contract()
    record = {
        "prompt_id": 0,
        "sample_id": 0,
        "timestep_index": 1,
        "timestep_value": 0.5,
        "latent_t": None,
        "latent_next": None,
        "old_log_prob": 0.0,
        "reward": 1.0,
        "advantage": 0.2,
        "logprob_reduction": "mean",
        "solver_id": "euler_flow",
        "sigma_form": "flow_grpo",
        "policy_forward_kind": "sft_forward",
    }
    validate_solver_contract(record, c)  # ok

    bad = dict(record)
    bad["solver_id"] = "unipc"
    with pytest.raises(ValueError, match="violates SolverContract"):
        validate_solver_contract(bad, c)
