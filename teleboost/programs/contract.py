# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contract for first-class TeleBoost programs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


_PROGRAM_NAME_PATTERN: Final = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")


class ProgramNotFoundError(LookupError):
    """No built-in program matches the requested public program name."""


def normalize_program_name(value: object, *, field_name: str = "name") -> str:
    if not isinstance(value, str):
        raise ValueError(f"Program {field_name} must be a string; got {type(value).__name__}")
    normalized = value.strip().lower()
    if not normalized or not _PROGRAM_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid program {field_name} {value!r}; use lowercase letters, digits, '.', '_' or '-'")
    return normalized


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    """Public program identity: family × algorithm × engine × runtime policy."""

    name: str
    backend_name: str
    family: str
    algorithm: str
    engine: str
    policy: str = "train"
    public: bool = True

    def __post_init__(self) -> None:
        for field_name in ("name", "backend_name", "family", "algorithm", "engine", "policy"):
            object.__setattr__(
                self,
                field_name,
                normalize_program_name(getattr(self, field_name), field_name=field_name),
            )


__all__ = [
    "ProgramNotFoundError",
    "ProgramSpec",
    "normalize_program_name",
]
