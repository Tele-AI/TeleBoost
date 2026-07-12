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
"""TeleBoost reward domain code.

Package initialization stays dependency-light.  Heavy reward contracts pull in
training/runtime dependencies such as TensorDict and are loaded lazily only when
the reward worker asks for them.
"""


def __getattr__(name: str):
    if name in {"BaseRewardModel", "RewardConfig"}:
        from teleboost.reward import contract

        return getattr(contract, name)
    if name == "RewardRegistry":
        from teleboost.reward.registry import RewardRegistry

        return RewardRegistry
    if name == "create_reward_model":
        return create_reward_model
    if name == "list_available_models":
        return list_available_models
    raise AttributeError(name)


def list_available_models():
    """List all available reward model names."""

    from teleboost.reward.registry import RewardRegistry

    return RewardRegistry.list_available()


def create_reward_model(name: str, config, rank: int, world_size: int):
    """Create a reward model instance by name."""

    from teleboost.reward.registry import RewardRegistry

    return RewardRegistry.create(name, config, rank, world_size)


__all__ = [
    "BaseRewardModel",
    "RewardConfig",
    "RewardRegistry",
    "create_reward_model",
    "list_available_models",
]
