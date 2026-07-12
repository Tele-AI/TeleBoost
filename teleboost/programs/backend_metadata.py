# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Built-in backend metadata.

This is the single dependency-light source of truth for TeleBoost's built-in
model families.  Concrete training factories live at the training composition
root, not in the public plugin SPI.
"""

from __future__ import annotations

from types import MappingProxyType

from teleboost.programs.backend_registry import BackendRegistration, BackendRegistry
from teleboost.programs.builtins import (
    BUILTIN_PROGRAMS,
    BUILTIN_PROGRAMS_BY_NAME,
    ProgramNotFoundError,
    builtin_program_names,
    get_builtin_program,
)
from teleboost.programs.contract import ProgramSpec, normalize_program_name
from teleboost.training.families.metadata import WAN_CAPABILITIES, WAN_RUNTIME_TYPES


def _unbound_builtin_backend():
    raise RuntimeError("Built-in backend factories are installed by the training composition root before creating an in-tree backend.")


WAN_BACKEND = BackendRegistration(
    name="wan",
    factory=_unbound_builtin_backend,
    runtime_types=WAN_RUNTIME_TYPES,
    capabilities=WAN_CAPABILITIES,
    dependency_hint="install the train and wan dependency profiles and an importable top-level wan runtime",
)
BUILTIN_BACKENDS = (WAN_BACKEND,)
BUILTIN_BACKENDS_BY_NAME = MappingProxyType({registration.name: registration for registration in BUILTIN_BACKENDS})
BUILTIN_BACKENDS_BY_RUNTIME_TYPE = MappingProxyType({runtime_type: registration for registration in BUILTIN_BACKENDS for runtime_type in registration.runtime_types})


def register_builtin_backends(registry: BackendRegistry) -> BackendRegistry:
    """Install all protected built-ins into ``registry`` without constructing them."""

    for registration in BUILTIN_BACKENDS:
        registry.register_builtin(registration)
    return registry


__all__ = [
    "BUILTIN_BACKENDS",
    "BUILTIN_BACKENDS_BY_NAME",
    "BUILTIN_BACKENDS_BY_RUNTIME_TYPE",
    "BUILTIN_PROGRAMS",
    "BUILTIN_PROGRAMS_BY_NAME",
    "WAN_BACKEND",
    "WAN_CAPABILITIES",
    "WAN_RUNTIME_TYPES",
    "ProgramNotFoundError",
    "ProgramSpec",
    "builtin_program_names",
    "get_builtin_program",
    "normalize_program_name",
    "register_builtin_backends",
]
