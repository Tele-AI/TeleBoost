# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Configuration-driven program/backend selection."""

from __future__ import annotations

from typing import Any

from teleboost.programs.backend_api import (
    BackendNotFoundError,
    BackendRegistration,
    BackendRegistry,
    BackendSpec,
    get_backend_registry,
)
from teleboost.programs.backend_metadata import BUILTIN_BACKENDS_BY_RUNTIME_TYPE
from teleboost.config.access import select
from teleboost.programs.assembly import backend_name_for_program
from teleboost.programs.backends import register_program_backends
from teleboost.programs.registry import ProgramNotFoundError, get_program

_RUNTIME_SELECTOR_PATHS = (
    "actor_rollout_ref.type",
    "trainer.type",
    "data.type",
)


def _normalized_selector(value: object) -> str:
    return str(value or "").strip().lower()


def _runtime_selectors(config: Any) -> list[tuple[str, str]]:
    selectors = []
    for path in _RUNTIME_SELECTOR_PATHS:
        value = select(config, path)
        if value is not None:
            selectors.append((path, _normalized_selector(value)))
    return selectors


def _resolve_named_backend(
    backend_name: str,
    registry: BackendRegistry,
) -> BackendRegistration:
    try:
        return registry.resolve_canonical(backend_name)
    except BackendNotFoundError as exc:
        rendered = backend_name or "<missing>"
        raise ValueError(f"Unknown training backend {rendered!r} in backend.name. {exc}") from exc


def _resolve_program_backend(
    program_name: str,
    registry: BackendRegistry,
) -> BackendRegistration:
    try:
        program = get_program(program_name)
    except ProgramNotFoundError as exc:
        rendered = program_name or "<missing>"
        raise ValueError(f"Unknown training program {rendered!r} in program.name. {exc}") from exc
    try:
        backend_name = backend_name_for_program(program)
        return registry.resolve_canonical(backend_name)
    except BackendNotFoundError as exc:
        raise ValueError(f"Training program {program.name!r} selects backend {backend_name_for_program(program)!r}, but that backend is not registered. {exc}") from exc


def _resolve_builtin_from_runtime_types(
    selectors: list[tuple[str, str]],
    registry: BackendRegistry,
) -> BackendRegistration:
    if not selectors:
        raise ValueError("Unknown training backend '<missing>'. Set program.name or backend.name explicitly, or supply aligned actor_rollout_ref.type, trainer.type, and data.type for a built-in backend.")

    resolved = []
    for path, runtime_type in selectors:
        registration = BUILTIN_BACKENDS_BY_RUNTIME_TYPE.get(runtime_type)
        if registration is None:
            rendered = runtime_type or "<missing>"
            raise ValueError(f"Unknown training backend {rendered!r} in {path}. External plugins are discovered only by an exact backend.name.")
        resolved.append((path, runtime_type, registration))

    selected_path, selected_type, selected_registration = resolved[0]
    conflicts = [f"{path}={runtime_type!r}" for path, runtime_type, registration in resolved[1:] if registration.name != selected_registration.name]
    if conflicts:
        raise ValueError(f"Inconsistent training backend selectors: {selected_path}={selected_type!r}, {', '.join(conflicts)}")
    return registry.resolve_canonical(selected_registration.name)


def select_backend_registration(
    config: Any,
    *,
    registry: BackendRegistry | None = None,
) -> BackendRegistration:
    """Resolve metadata and cross-check component runtime selectors.

    ``program.name`` is the public product identity and maps to an already
    registered structural backend. ``backend.name`` remains the exact plugin
    discovery key. Without either, runtime types may select one of the
    dependency-light built-ins, but never trigger external entry-point scanning.
    """

    selected_registry = registry or get_backend_registry()
    if registry is None:
        register_program_backends(selected_registry)
    program_name_value = select(config, "program.name")
    backend_name_value = select(config, "backend.name")
    runtime_selectors = _runtime_selectors(config)

    if program_name_value is not None:
        program_name = _normalized_selector(program_name_value)
        registration = _resolve_program_backend(program_name, selected_registry)
        selected_path = "program.name"
        selected_value = program_name
        if backend_name_value is not None:
            backend_name = _normalized_selector(backend_name_value)
            backend_registration = _resolve_named_backend(backend_name, selected_registry)
            if backend_registration.name != registration.name:
                raise ValueError(f"Inconsistent training backend selectors: program.name={program_name!r} selects {registration.name!r}, but backend.name={backend_name!r} selects {backend_registration.name!r}")
    elif backend_name_value is None:
        registration = _resolve_builtin_from_runtime_types(
            runtime_selectors,
            selected_registry,
        )
        selected_path = runtime_selectors[0][0]
        selected_value = runtime_selectors[0][1]
    else:
        backend_name = _normalized_selector(backend_name_value)
        registration = _resolve_named_backend(backend_name, selected_registry)
        selected_path = "backend.name"
        selected_value = backend_name

    conflicts = [f"{path}={runtime_type!r}" for path, runtime_type in runtime_selectors if runtime_type not in registration.runtime_types]
    if conflicts:
        raise ValueError(f"Inconsistent training backend selectors: {selected_path}={selected_value!r} selects {registration.name!r}, but {', '.join(conflicts)} are not registered runtime types for it.")
    return registration


def select_backend(
    config: Any,
    *,
    registry: BackendRegistry | None = None,
) -> BackendSpec:
    """Construct only the selected backend's structural contract implementation."""

    selected_registry = registry or get_backend_registry()
    if registry is None:
        register_program_backends(selected_registry)
    registration = select_backend_registration(config, registry=selected_registry)
    return selected_registry.create(registration.name)


__all__ = ["select_backend", "select_backend_registration"]
