# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit import boundary for Wan worker-side verl patches.

Runtime modules import ``PATCHES_APPLIED`` before importing verl. Keeping the
call in this tiny boundary preserves normal import ordering in the large worker
modules while leaving program/backend contracts pure.
"""

from teleboost import apply_runtime_patches

PATCHES_APPLIED = apply_runtime_patches()

__all__ = ["PATCHES_APPLIED"]
