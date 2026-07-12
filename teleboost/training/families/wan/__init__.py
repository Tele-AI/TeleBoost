# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Wan training-family adapters."""

__all__ = ["RayWanTrainer", "WanGenerationMixin"]


def __getattr__(name: str):
    if name in __all__:
        from teleboost.training.families.wan.trainer import RayWanTrainer, WanGenerationMixin

        return {"RayWanTrainer": RayWanTrainer, "WanGenerationMixin": WanGenerationMixin}[name]
    raise AttributeError(name)
