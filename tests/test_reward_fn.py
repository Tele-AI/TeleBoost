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

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from teleboost.reward.execution.custom import load_custom_reward_fn


def test_loads_pkg_reward_fn_and_merges_kwargs():
    cfg = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": "pkg://tests.test_reward_fn",
                    "name": "example_reward",
                    "reward_kwargs": {"scale": 3.0},
                }
            }
        }
    )

    assert load_custom_reward_fn(cfg)(SimpleNamespace(value=2.0)) == 6.0


def test_reward_fn_name_defaults_to_compute_score():
    cfg = OmegaConf.create({"reward": {"custom_reward_function": {"path": "pkg://tests.test_reward_fn"}}})
    assert load_custom_reward_fn(cfg)(SimpleNamespace(value=2.0)) == 2.0


def test_missing_reward_fn_path_fails_loudly():
    cfg = OmegaConf.create({"reward": {"custom_reward_function": {"path": None, "name": "compute_score"}}})
    with pytest.raises(ValueError, match="custom_reward_function.path"):
        load_custom_reward_fn(cfg)


def compute_score(data):
    return data.value


def example_reward(data, scale=1.0):
    return data.value * scale
