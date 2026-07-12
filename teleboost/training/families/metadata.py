# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light metadata for built-in training families."""

WAN_RUNTIME_TYPES = frozenset({"diffusion", "wan"})

WAN_CAPABILITIES = frozenset(
    {
        "algorithm.grpo",
        "generation.latent",
        "media.video",
        "parallel.context",
        "parallel.fsdp",
        "reward.video_vlm",
    }
)
__all__ = [
    "WAN_CAPABILITIES",
    "WAN_RUNTIME_TYPES",
]
