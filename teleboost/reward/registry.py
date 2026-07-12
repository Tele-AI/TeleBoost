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
"""
Reward Model Registry - A registry pattern for managing reward models.

Usage:
    @RewardRegistry.register("my_reward")
    class MyRewardModel(BaseRewardModel):
        ...

    # Create instance
    model = RewardRegistry.create("my_reward", config, rank, world_size)
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teleboost.reward.contract import BaseRewardModel, RewardConfig


class RewardRegistry:
    """
    Central registry for all reward models.

    Provides a decorator-based registration mechanism and factory method
    for creating reward model instances.
    """

    _registry: dict[str, type["BaseRewardModel"]] = {}
    # Import reward implementations only when selected.  Several of these
    # have sizeable optional dependency trees (OpenCV, torchvision, HPSv2,
    # Videophy); importing every model made even the lightweight/random
    # reward path depend on all of them being installed correctly.
    _builtin_modules: dict[str, str] = {
        "aesthetic": "teleboost.reward.providers.external.aesthetic",
        "hps": "teleboost.reward.providers.external.hps",
        "raft": "teleboost.reward.providers.external.raft",
        "random": "teleboost.reward.providers.debug.random",
        "temporal_quality": "teleboost.reward.providers.debug.temporal_quality",
        "videoclip": "teleboost.reward.providers.external.videoclip",
        "videophy": "teleboost.reward.providers.external.videophy",
    }

    @classmethod
    def _load_builtin(cls, name: str) -> None:
        module_name = cls._builtin_modules.get(name)
        if module_name is None or name in cls._registry:
            return
        importlib.import_module(module_name)

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a reward model class.

        Args:
            name: Unique identifier for the reward model

        Example:
            @RewardRegistry.register("aesthetic")
            class AestheticRewardModel(BaseRewardModel):
                pass
        """

        def decorator(model_cls: type["BaseRewardModel"]) -> type["BaseRewardModel"]:
            if name in cls._registry:
                raise ValueError(f"Reward model '{name}' is already registered")
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type["BaseRewardModel"]:
        """
        Get a reward model class by name.

        Args:
            name: The registered name of the reward model

        Returns:
            The reward model class

        Raises:
            ValueError: If the name is not registered
        """
        cls._load_builtin(name)
        if name not in cls._registry:
            available = cls.list_available()
            raise ValueError(f"Unknown reward model: '{name}'. Available models: {available}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, config: "RewardConfig", rank: int, world_size: int) -> "BaseRewardModel":
        """
        Factory method to create a reward model instance.

        Args:
            name: The registered name of the reward model
            config: Configuration for the reward model
            rank: Current process rank
            world_size: Total number of processes

        Returns:
            An instance of the requested reward model
        """
        model_cls = cls.get(name)
        return model_cls(config, rank, world_size)

    @classmethod
    def list_available(cls) -> list:
        """List built-in and custom reward names without importing models."""
        return sorted(cls._builtin_modules.keys() | cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a reward model name is known without loading it."""
        return name in cls._registry or name in cls._builtin_modules
