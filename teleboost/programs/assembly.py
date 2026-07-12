# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Thin assembly helpers for first-class TeleBoost programs.

The heavy family implementations remain in ``teleboost.programs.<family>``.
This module maps a dependency-light :class:`ProgramSpec` to the backend family
that owns the actual construction policy.
"""

from __future__ import annotations

from teleboost.programs.contract import ProgramSpec


def backend_name_for_program(program: ProgramSpec) -> str:
    """Return the backend family selected by ``program``.

    Keeping this as a tiny function gives program selection one stable assembly
    seam without introducing a broad Engine/Program abstraction ahead of need.
    """

    return program.backend_name


__all__ = ["backend_name_for_program"]
