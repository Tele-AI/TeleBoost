# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Wan attention kernels and backend selection."""

from teleboost.models.wan.attention.backend import (
    available_wan_attention_backends,
    resolve_wan_attention_backend,
    wan_dense_attention,
    wan_flash_varlen_attention,
)
from teleboost.models.wan.attention.ops import wan_attention, wan_flash_attention

__all__ = [
    "available_wan_attention_backends",
    "resolve_wan_attention_backend",
    "wan_attention",
    "wan_dense_attention",
    "wan_flash_attention",
    "wan_flash_varlen_attention",
]
