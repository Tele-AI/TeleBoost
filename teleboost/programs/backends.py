# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Program-owned registrations for in-tree backend implementations."""

from __future__ import annotations

from teleboost.programs.backend_metadata import WAN_CAPABILITIES, WAN_RUNTIME_TYPES
from teleboost.programs.backend_registry import BackendRegistration, BackendRegistry


def _create_wan_backend():
    from teleboost.programs.wan.backend import WanBackendSpec

    return WanBackendSpec()


WAN_BACKEND = BackendRegistration(
    name="wan",
    factory=_create_wan_backend,
    runtime_types=WAN_RUNTIME_TYPES,
    capabilities=WAN_CAPABILITIES,
    dependency_hint="install the train and wan dependency profiles and an importable top-level wan runtime",
)
BUILTIN_BACKENDS = (WAN_BACKEND,)


def register_program_backends(registry: BackendRegistry) -> BackendRegistry:
    for registration in BUILTIN_BACKENDS:
        if registration.name in getattr(registry, "_by_name", {}):
            continue
        registry.register_builtin(registration)
    return registry


__all__ = [
    "BUILTIN_BACKENDS",
    "WAN_BACKEND",
    "register_program_backends",
]
