# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light registry for first-class TeleBoost programs."""

from __future__ import annotations

from teleboost.programs.builtins import (
    BUILTIN_PROGRAMS,
    ProgramNotFoundError,
    builtin_program_names,
    get_builtin_program,
)
from teleboost.programs.contract import ProgramSpec


def get_program(name: object) -> ProgramSpec:
    return get_builtin_program(name)


def program_names(*, public_only: bool = False) -> tuple[str, ...]:
    return builtin_program_names(public_only=public_only)


__all__ = [
    "BUILTIN_PROGRAMS",
    "ProgramNotFoundError",
    "ProgramSpec",
    "get_program",
    "program_names",
]
