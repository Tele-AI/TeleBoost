# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable training program identities.

Package initialization is intentionally dependency-light.  Import concrete
selection helpers lazily so metadata users can read ProgramSpec declarations
without importing backend registries or training composition code.
"""


def __getattr__(name: str):
    if name in {"ProgramNotFoundError", "ProgramSpec"}:
        from teleboost.programs import contract

        return getattr(contract, name)
    if name in {
        "BACKEND_API_VERSION",
        "BACKEND_ENTRY_POINT_GROUP",
        "BackendAPIVersionError",
        "BackendCollisionError",
        "BackendFactory",
        "BackendFactoryError",
        "BackendNotFoundError",
        "BackendPluginLoadError",
        "BackendRegistration",
        "BackendRegistrationError",
        "BackendRegistry",
        "BackendRegistryError",
        "BackendSpec",
        "create",
        "get_backend_registry",
        "register_builtin",
        "register_external",
        "resolve",
        "resolve_canonical",
    }:
        from teleboost.programs import backend_api

        return getattr(backend_api, name)
    if name in {"get_program", "program_names"}:
        from teleboost.programs import registry

        return getattr(registry, name)
    if name in {"select_backend", "select_backend_registration"}:
        from teleboost.programs import selection

        return getattr(selection, name)
    raise AttributeError(name)


__all__ = [
    "BACKEND_API_VERSION",
    "BACKEND_ENTRY_POINT_GROUP",
    "BackendAPIVersionError",
    "BackendCollisionError",
    "BackendFactory",
    "BackendFactoryError",
    "BackendNotFoundError",
    "BackendPluginLoadError",
    "BackendRegistration",
    "BackendRegistrationError",
    "BackendRegistry",
    "BackendRegistryError",
    "BackendSpec",
    "ProgramNotFoundError",
    "ProgramSpec",
    "create",
    "get_backend_registry",
    "get_program",
    "program_names",
    "register_builtin",
    "register_external",
    "resolve",
    "resolve_canonical",
    "select_backend",
    "select_backend_registration",
]
