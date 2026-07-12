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
"""Reward adapter contract helpers.

``reward.reward_model.adapter`` chooses how rollout data is converted into a
reward request. ``reward.reward_model.model_name`` names the served/scoring
model only; it does not select the data path.
"""

from __future__ import annotations

from typing import Any

from teleboost.config.access import select as _select


VIDEO_VLM_ADAPTER = "video_vlm"
VLLM_REWARD_ADAPTERS = frozenset({VIDEO_VLM_ADAPTER})


def reward_model_enabled(config: Any) -> bool:
    return bool(
        _select(
            config,
            "reward.reward_model.enable",
            default=False,
        )
    )


def reward_model_type(config: Any) -> str:
    return (
        str(
            _select(
                config,
                "reward.reward_model.type",
                default="",
            )
            or ""
        )
        .strip()
        .lower()
    )


def reward_model_name(config: Any) -> str:
    return (
        str(
            _select(
                config,
                "reward.reward_model.model_name",
                default="",
            )
            or ""
        )
        .strip()
        .lower()
    )


def explicit_reward_adapter(config: Any) -> str:
    return (
        str(
            _select(
                config,
                "reward.reward_model.adapter",
                default="",
            )
            or ""
        )
        .strip()
        .lower()
    )


def reward_model_adapter(config: Any) -> str:
    return explicit_reward_adapter(config)


def uses_vllm_reward_router(config: Any) -> bool:
    return reward_model_enabled(config) and reward_model_adapter(config) in VLLM_REWARD_ADAPTERS


def is_video_vlm_reward_config(config: Any) -> bool:
    return reward_model_enabled(config) and reward_model_adapter(config) == VIDEO_VLM_ADAPTER
