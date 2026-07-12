# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""FSDP execution, sharding, and context-parallel helpers."""

from teleboost.engines.fsdp.execution import (
    ensure_fsdp_forward_dispatch,
    fsdp_managed_call,
    is_fsdp_module,
)

__all__ = [
    "ensure_fsdp_forward_dispatch",
    "fsdp_managed_call",
    "is_fsdp_module",
]
