# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light structural backend API used by program assembly."""

from teleboost.programs.backend_contract import BackendFactory, BackendSpec
from teleboost.programs.backend_registry import (
    BACKEND_API_VERSION,
    BACKEND_ENTRY_POINT_GROUP,
    BackendAPIVersionError,
    BackendCollisionError,
    BackendFactoryError,
    BackendNotFoundError,
    BackendPluginLoadError,
    BackendRegistration,
    BackendRegistrationError,
    BackendRegistry,
    BackendRegistryError,
    create,
    get_backend_registry,
    register_builtin,
    register_external,
    resolve,
    resolve_canonical,
)

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
    "create",
    "get_backend_registry",
    "register_builtin",
    "register_external",
    "resolve",
    "resolve_canonical",
]
