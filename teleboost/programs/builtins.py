# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Built-in ProgramSpec declaration table."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from teleboost.programs.contract import ProgramNotFoundError, ProgramSpec, normalize_program_name


def _program(
    name: str,
    *,
    backend_name: str,
    family: str,
    algorithm: str,
    engine: str,
    policy: str = "train",
    public: bool = True,
) -> ProgramSpec:
    return ProgramSpec(
        name=name,
        backend_name=backend_name,
        family=family,
        algorithm=algorithm,
        engine=engine,
        policy=policy,
        public=public,
    )


BUILTIN_PROGRAMS: Final[tuple[ProgramSpec, ...]] = (
    _program("wan.grpo.fsdp", backend_name="wan", family="wan", algorithm="grpo", engine="fsdp"),
    _program("wan.bgpo.fsdp", backend_name="wan", family="wan", algorithm="bgpo", engine="fsdp"),
    _program("wan.vipo.fsdp", backend_name="wan", family="wan", algorithm="vipo", engine="fsdp"),
    _program("wan.tempflow.fsdp", backend_name="wan", family="wan", algorithm="tempflow", engine="fsdp"),
    _program("wan.dpo.teletron", backend_name="wan", family="wan", algorithm="dpo", engine="teletron"),
)

BUILTIN_PROGRAMS_BY_NAME = MappingProxyType({program.name: program for program in BUILTIN_PROGRAMS})


def get_builtin_program(name: object) -> ProgramSpec:
    normalized = normalize_program_name(name, field_name="name")
    try:
        return BUILTIN_PROGRAMS_BY_NAME[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_PROGRAMS_BY_NAME))
        raise ProgramNotFoundError(f"Unknown TeleBoost program {normalized!r}. Known programs: {known}") from exc


def builtin_program_names(*, public_only: bool = False) -> tuple[str, ...]:
    programs = BUILTIN_PROGRAMS
    if public_only:
        programs = tuple(program for program in programs if program.public)
    return tuple(program.name for program in programs)


__all__ = [
    "BUILTIN_PROGRAMS",
    "BUILTIN_PROGRAMS_BY_NAME",
    "ProgramNotFoundError",
    "ProgramSpec",
    "builtin_program_names",
    "get_builtin_program",
]
