# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict, dependency-light registry for model-family backend plugins.

External plugins use the ``teleboost.programs`` entry-point group.  Discovery
is deliberately selector-driven: resolving ``backend.name=acme`` asks metadata
only for the entry point named ``acme`` and never imports unrelated plugins.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from threading import RLock
from typing import Any, Final, TypeAlias

from teleboost.programs.backend_contract import BackendFactory, BackendSpec

BACKEND_API_VERSION: Final = 1
BACKEND_ENTRY_POINT_GROUP: Final = "teleboost.programs"

EntryPointProvider: TypeAlias = Callable[..., Iterable[Any]]

_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")


class BackendRegistryError(RuntimeError):
    """Base class for backend registry failures."""


class BackendRegistrationError(BackendRegistryError, ValueError):
    """A backend registration is invalid."""


class BackendAPIVersionError(BackendRegistrationError):
    """A plugin targets an unsupported backend API version."""


class BackendCollisionError(BackendRegistrationError):
    """A canonical name or runtime type is already registered."""


class BackendNotFoundError(BackendRegistryError, LookupError):
    """No built-in or selected external backend matches a selector."""


class BackendPluginLoadError(BackendRegistryError):
    """The selected external backend entry point could not be loaded."""


class BackendFactoryError(BackendRegistryError):
    """A registered backend's lazy factory could not construct its spec."""


def _normalize_declared_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BackendRegistrationError(f"Backend {field_name} must be a string; got {type(value).__name__}")
    normalized = value.strip().lower()
    if not normalized or not _NAME_PATTERN.fullmatch(normalized):
        raise BackendRegistrationError(f"Invalid backend {field_name} {value!r}; use lowercase letters, digits, '.', '_' or '-'")
    return normalized


def _normalize_selector(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise BackendNotFoundError("Backend selector is missing or empty")
    return normalized


def _normalize_string_set(values: Iterable[str], *, field_name: str) -> frozenset[str]:
    if isinstance(values, str):
        raise BackendRegistrationError(f"Backend {field_name} must be an iterable of strings, not one string")
    try:
        return frozenset(_normalize_declared_name(value, field_name=field_name) for value in values)
    except TypeError as exc:
        raise BackendRegistrationError(f"Backend {field_name} must be an iterable of strings") from exc


@dataclass(frozen=True, slots=True)
class BackendRegistration:
    """Stable plugin metadata plus a lazy backend-spec factory.

    ``name`` is the canonical value accepted by ``backend.name`` and must
    equal an external plugin's entry-point name. ``runtime_types`` contains
    concrete actor/trainer/data selector values for an already registered
    backend; it is not used to scan or eagerly import external plugins.
    """

    name: str
    factory: BackendFactory
    runtime_types: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    dependency_hint: str | None = None
    api_version: int = BACKEND_API_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_declared_name(self.name, field_name="name"),
        )
        object.__setattr__(
            self,
            "runtime_types",
            _normalize_string_set(self.runtime_types, field_name="runtime_types"),
        )
        object.__setattr__(
            self,
            "capabilities",
            _normalize_string_set(self.capabilities, field_name="capabilities"),
        )
        if not callable(self.factory):
            raise BackendRegistrationError(f"Backend {self.name!r} factory must be callable")
        if self.dependency_hint is not None:
            if not isinstance(self.dependency_hint, str):
                raise BackendRegistrationError(f"Backend {self.name!r} dependency_hint must be a string or None")
            hint = self.dependency_hint.strip()
            object.__setattr__(self, "dependency_hint", hint or None)

    @property
    def selectors(self) -> frozenset[str]:
        """All selectors understood after this registration is installed."""

        return self.runtime_types | {self.name}


class BackendRegistry:
    """Registry of built-in and selector-loaded external backends."""

    def __init__(
        self,
        *,
        entry_point_group: str = BACKEND_ENTRY_POINT_GROUP,
        entry_point_provider: EntryPointProvider | None = None,
    ) -> None:
        self._entry_point_group = entry_point_group
        # Keep the stdlib provider lookup lazy as well. Besides making tests
        # easy to isolate, this avoids freezing discovery state at import time.
        self._entry_point_provider = entry_point_provider
        self._by_name: dict[str, BackendRegistration] = {}
        self._by_selector: dict[str, BackendRegistration] = {}
        self._builtin_names: set[str] = set()
        self._lock = RLock()

    @property
    def entry_point_group(self) -> str:
        return self._entry_point_group

    def register_builtin(self, registration: BackendRegistration) -> BackendRegistration:
        """Register a protected in-tree backend."""

        return self._register(registration, builtin=True)

    def register_external(self, registration: BackendRegistration) -> BackendRegistration:
        """Register an external backend without replacing any existing entry."""

        return self._register(registration, builtin=False)

    # ``register`` is a convenient spelling for explicit programmatic plugins.
    register = register_external

    def _register(
        self,
        registration: BackendRegistration,
        *,
        builtin: bool,
    ) -> BackendRegistration:
        if not isinstance(registration, BackendRegistration):
            raise BackendRegistrationError(f"Backend registry accepts BackendRegistration objects; got {type(registration).__name__}")
        if type(registration.api_version) is not int or registration.api_version != BACKEND_API_VERSION:
            raise BackendAPIVersionError(f"Backend {registration.name!r} uses API version {registration.api_version!r}; TeleBoost supports only version {BACKEND_API_VERSION}")

        with self._lock:
            collisions = {selector: self._by_selector[selector] for selector in registration.selectors if selector in self._by_selector}
            if collisions:
                rendered = ", ".join(f"{selector!r} -> {existing.name!r}" for selector, existing in sorted(collisions.items()))
                protected = sorted({existing.name for existing in collisions.values() if existing.name in self._builtin_names})
                protection = f"; protected built-ins: {protected}" if protected else ""
                raise BackendCollisionError(f"Cannot register backend {registration.name!r}; selector collision(s): {rendered}{protection}")

            self._by_name[registration.name] = registration
            for selector in registration.selectors:
                self._by_selector[selector] = registration
            if builtin:
                self._builtin_names.add(registration.name)
        return registration

    def resolve(self, name: object) -> BackendRegistration:
        """Resolve one selector, loading at most its exact external entry point."""

        selector = _normalize_selector(name)
        with self._lock:
            registration = self._by_selector.get(selector)
            if registration is not None:
                return registration

            # Serialize discovery so concurrent requests for the same plugin
            # cannot both load and race to register it.
            self._load_selected_external(selector)
            registration = self._by_selector.get(selector)
        if registration is not None:
            return registration

        # The entry-point registration is required to use the selected
        # canonical name, so reaching here indicates an internal invariant bug.
        raise BackendPluginLoadError(f"External backend {selector!r} loaded without registering its selector")

    def resolve_canonical(self, name: object) -> BackendRegistration:
        """Resolve a canonical backend name, never a runtime-type alias.

        External discovery remains exact-name driven.  A registered runtime
        selector therefore cannot accidentally make ``backend.name`` accept a
        non-canonical value.
        """

        selector = _normalize_selector(name)
        with self._lock:
            registration = self._by_name.get(selector)
            runtime_owner = self._by_selector.get(selector)
            known = sorted(self._by_name)
            if registration is not None:
                return registration
            if runtime_owner is not None:
                raise BackendNotFoundError(f"Unknown canonical backend name {selector!r}. It is a runtime type for {runtime_owner.name!r}; backend.name accepts canonical names only: {known}.")

            self._load_selected_external(selector)
            registration = self._by_name.get(selector)
        if registration is None:
            raise BackendPluginLoadError(f"External backend {selector!r} loaded without registering its canonical name")
        return registration

    def create(self, name: object) -> BackendSpec:
        """Resolve a backend and invoke only that registration's lazy factory."""

        registration = self.resolve(name)
        try:
            backend = registration.factory()
        except Exception as exc:
            hint = f" Dependency hint: {registration.dependency_hint}." if registration.dependency_hint else ""
            raise BackendFactoryError(f"Backend {registration.name!r} factory failed: {type(exc).__name__}: {exc}.{hint}") from exc
        if not isinstance(backend, BackendSpec):
            raise BackendFactoryError(f"Backend {registration.name!r} factory returned {type(backend).__name__}, which does not implement BackendSpec")
        try:
            backend_name = _normalize_declared_name(backend.name, field_name="spec name")
        except BackendRegistrationError as exc:
            raise BackendFactoryError(f"Backend {registration.name!r} factory returned a spec with an invalid name: {exc}") from exc
        if backend_name != registration.name:
            raise BackendFactoryError(f"Backend {registration.name!r} factory returned a spec named {backend_name!r}")
        return backend

    def _load_selected_external(self, selector: str) -> None:
        try:
            provider = self._entry_point_provider or metadata.entry_points
            discovered = tuple(
                provider(
                    group=self._entry_point_group,
                    name=selector,
                )
            )
        except Exception as exc:
            raise BackendPluginLoadError(f"Failed to discover external backend {selector!r} in entry-point group {self._entry_point_group!r}: {type(exc).__name__}: {exc}") from exc

        matches = tuple(entry_point for entry_point in discovered if getattr(entry_point, "group", self._entry_point_group) == self._entry_point_group and getattr(entry_point, "name", None) == selector)
        if not matches:
            with self._lock:
                known = sorted(self._by_name)
            raise BackendNotFoundError(f"Unknown backend {selector!r}. Registered canonical names: {known}. No matching {self._entry_point_group!r} entry point is installed.")
        if len(matches) != 1:
            providers = [self._describe_entry_point(item) for item in matches]
            raise BackendPluginLoadError(f"External backend {selector!r} is ambiguous: found {len(matches)} entry points in {self._entry_point_group!r}: {providers}")

        entry_point = matches[0]
        try:
            loaded = entry_point.load()
        except Exception as exc:
            raise BackendPluginLoadError(f"Failed to load external backend {selector!r} from {self._describe_entry_point(entry_point)}: {type(exc).__name__}: {exc}") from exc

        registration = self._coerce_entry_point_value(
            loaded,
            selector=selector,
            entry_point=entry_point,
        )
        if registration.name != selector:
            raise BackendPluginLoadError(f"External backend entry point {selector!r} returned registration for {registration.name!r}; entry-point name and canonical backend name must match exactly")
        self.register_external(registration)

    def _coerce_entry_point_value(
        self,
        loaded: object,
        *,
        selector: str,
        entry_point: Any,
    ) -> BackendRegistration:
        if isinstance(loaded, BackendRegistration):
            return loaded
        if not callable(loaded):
            raise BackendPluginLoadError(f"External backend {selector!r} entry point {self._describe_entry_point(entry_point)} returned {type(loaded).__name__}; expected BackendRegistration or a zero-argument factory returning one")
        try:
            registration = loaded()
        except Exception as exc:
            raise BackendPluginLoadError(f"External backend {selector!r} zero-argument registration factory from {self._describe_entry_point(entry_point)} failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(registration, BackendRegistration):
            raise BackendPluginLoadError(f"External backend {selector!r} registration factory returned {type(registration).__name__}; expected BackendRegistration")
        return registration

    @staticmethod
    def _describe_entry_point(entry_point: Any) -> str:
        value = getattr(entry_point, "value", "<unknown target>")
        distribution = getattr(entry_point, "dist", None)
        distribution_name = getattr(distribution, "name", None)
        suffix = f" from distribution {distribution_name!r}" if distribution_name else ""
        return f"{getattr(entry_point, 'name', '<unnamed>')!r} ({value}){suffix}"


_DEFAULT_REGISTRY = BackendRegistry()
_DEFAULT_REGISTRY_LOCK = RLock()


def get_backend_registry() -> BackendRegistry:
    """Return the process-wide backend registry.

    The public SPI does not auto-install in-tree training factories.  The
    composition root that owns those factories registers them explicitly.
    """

    return _DEFAULT_REGISTRY


def register_builtin(registration: BackendRegistration) -> BackendRegistration:
    return get_backend_registry().register_builtin(registration)


def register_external(registration: BackendRegistration) -> BackendRegistration:
    return get_backend_registry().register_external(registration)


def resolve(name: object) -> BackendRegistration:
    return get_backend_registry().resolve(name)


def resolve_canonical(name: object) -> BackendRegistration:
    return get_backend_registry().resolve_canonical(name)


def create(name: object) -> BackendSpec:
    return get_backend_registry().create(name)


__all__ = [
    "BACKEND_API_VERSION",
    "BACKEND_ENTRY_POINT_GROUP",
    "BackendAPIVersionError",
    "BackendCollisionError",
    "BackendFactoryError",
    "BackendNotFoundError",
    "BackendPluginLoadError",
    "BackendRegistration",
    "BackendRegistrationError",
    "BackendRegistry",
    "BackendRegistryError",
    "create",
    "get_backend_registry",
    "register_builtin",
    "register_external",
    "resolve",
    "resolve_canonical",
]
