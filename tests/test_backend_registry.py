# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure unit coverage for the dependency-light backend plugin SPI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import pytest

from teleboost.programs import (
    BACKEND_API_VERSION,
    BACKEND_ENTRY_POINT_GROUP,
    BackendAPIVersionError,
    BackendCollisionError,
    BackendFactoryError,
    BackendNotFoundError,
    BackendPluginLoadError,
    BackendRegistration,
    BackendRegistry,
    BackendSpec,
)


class _Backend:
    name = "demo"

    def validate_capabilities(self, config: Any) -> None:
        pass

    def validate_reward(self, config: Any) -> None:
        pass

    def prepare_tokenizer(self, config: Any) -> tuple[Any, Any]:
        return None, None

    def resolve_worker_and_group(self, config: Any) -> tuple[type[Any], type[Any]]:
        return object, object

    def register_reward_workers(
        self,
        config: Any,
        role_worker_mapping: dict[Any, Any],
        mapping: dict[Any, Any],
        global_pool_id: str,
    ) -> None:
        pass

    def collate_fn(self, config: Any):
        return lambda batch: batch

    def trainer_cls(self, config: Any) -> type[Any]:
        return object


def _registration(
    name: str,
    *,
    runtime_types: frozenset[str] = frozenset(),
    factory=_Backend,
    api_version: int = BACKEND_API_VERSION,
    dependency_hint: str | None = None,
) -> BackendRegistration:
    return BackendRegistration(
        name=name,
        runtime_types=runtime_types,
        factory=factory,
        api_version=api_version,
        dependency_hint=dependency_hint,
    )


@dataclass
class _FakeEntryPoint:
    name: str
    payload: object
    group: str = BACKEND_ENTRY_POINT_GROUP
    value: str = "example.backend:registration"
    loads: int = 0

    def load(self) -> object:
        self.loads += 1
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


def test_registration_resolves_canonical_and_runtime_types_and_creates_lazily():
    factory_calls = []

    def factory() -> BackendSpec:
        factory_calls.append(True)
        return _Backend()

    registration = BackendRegistration(
        name="demo",
        runtime_types=frozenset({"demo-runtime"}),
        capabilities=frozenset({"context-parallel"}),
        factory=factory,
    )
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: ())

    assert registry.register_builtin(registration) is registration
    assert registry.resolve("demo") is registration
    assert registry.resolve("DEMO-RUNTIME") is registration
    assert factory_calls == []
    assert isinstance(registry.create("demo"), BackendSpec)
    assert factory_calls == [True]
    assert registration.capabilities == frozenset({"context-parallel"})


def test_canonical_and_runtime_type_collisions_are_hard_failures():
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: ())
    registry.register_builtin(_registration("builtin", runtime_types=frozenset({"shared-runtime"})))

    with pytest.raises(BackendCollisionError, match="protected built-ins"):
        registry.register_external(_registration("builtin"))
    with pytest.raises(BackendCollisionError, match="shared-runtime"):
        registry.register_external(_registration("plugin", runtime_types=frozenset({"shared-runtime"})))
    with pytest.raises(BackendCollisionError, match="builtin"):
        registry.register_external(_registration("other", runtime_types=frozenset({"builtin"})))

    assert registry.resolve("builtin").name == "builtin"


def test_canonical_resolution_never_accepts_a_runtime_type_alias():
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: ())
    registration = _registration(
        "builtin",
        runtime_types=frozenset({"builtin-runtime"}),
    )
    registry.register_builtin(registration)

    assert registry.resolve_canonical("builtin") is registration
    with pytest.raises(BackendNotFoundError, match="runtime type.*canonical"):
        registry.resolve_canonical("builtin-runtime")


@pytest.mark.parametrize("invalid_version", [2, True, 1.0, "1"])
def test_api_version_is_checked_before_registry_mutation(invalid_version):
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: ())

    with pytest.raises(BackendAPIVersionError, match="supports only version 1"):
        registry.register_external(_registration("future", api_version=invalid_version))
    with pytest.raises(BackendNotFoundError, match="Unknown backend 'future'"):
        registry.resolve("future")


def test_unknown_backend_error_names_the_selector_and_plugin_group():
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return ()

    registry = BackendRegistry(entry_point_provider=provider)

    with pytest.raises(BackendNotFoundError) as caught:
        registry.resolve("missing")

    assert "'missing'" in str(caught.value)
    assert BACKEND_ENTRY_POINT_GROUP in str(caught.value)
    assert calls == [{"group": BACKEND_ENTRY_POINT_GROUP, "name": "missing"}]


@pytest.mark.parametrize("use_factory", [False, True])
def test_entry_point_accepts_registration_or_zero_argument_factory(use_factory: bool):
    registration = _registration("selected")
    payload = (lambda: registration) if use_factory else registration
    selected = _FakeEntryPoint("selected", payload)
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: (selected,))

    assert registry.resolve("selected") is registration
    assert selected.loads == 1


def test_external_discovery_loads_only_the_exact_selected_plugin():
    selected_registration = _registration("selected", runtime_types=frozenset({"selected-runtime"}))
    selected = _FakeEntryPoint("selected", selected_registration)
    unrelated = _FakeEntryPoint("unrelated", _registration("unrelated"))
    calls = []

    # Return both entries intentionally. The registry must still filter before
    # calling EntryPoint.load(), in addition to requesting metadata by name.
    def provider(**kwargs):
        calls.append(kwargs)
        return selected, unrelated

    registry = BackendRegistry(entry_point_provider=provider)

    assert registry.resolve("selected").name == "selected"
    assert registry.resolve("selected-runtime").name == "selected"
    assert selected.loads == 1
    assert unrelated.loads == 0
    assert calls == [{"group": BACKEND_ENTRY_POINT_GROUP, "name": "selected"}]


def test_entry_point_name_must_equal_canonical_registration_name():
    entry_point = _FakeEntryPoint("selected", _registration("different"))
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: (entry_point,))

    with pytest.raises(BackendPluginLoadError, match="must match exactly"):
        registry.resolve("selected")


def test_entry_point_errors_explain_the_required_payload_shape():
    bad_payload = _FakeEntryPoint("selected", object())
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: (bad_payload,))

    with pytest.raises(BackendPluginLoadError, match="zero-argument factory"):
        registry.resolve("selected")


def test_backend_factory_error_includes_optional_dependency_hint():
    def missing_dependency():
        raise ModuleNotFoundError("optional_runtime")

    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: ())
    registry.register_builtin(
        _registration(
            "demo",
            factory=missing_dependency,
            dependency_hint="pip install teleboost-demo",
        )
    )

    with pytest.raises(BackendFactoryError) as caught:
        registry.create("demo")

    assert "ModuleNotFoundError: optional_runtime" in str(caught.value)
    assert "pip install teleboost-demo" in str(caught.value)


def test_backend_factory_spec_name_must_match_registration():
    registry = BackendRegistry(entry_point_provider=lambda **_kwargs: ())

    def mismatched_factory():
        backend = _Backend()
        backend.name = "different"
        return backend

    registry.register_builtin(_registration("demo", factory=mismatched_factory))

    with pytest.raises(BackendFactoryError, match="returned a spec named 'different'"):
        registry.create("demo")


def test_importing_backend_contract_does_not_load_training_or_family_runtimes():
    code = """
import sys
import teleboost.programs.backend_contract
blocked = (
    'verl', 'ray', 'wan',
    'flash_attn', 'flash_attn_3', 'flash_attn_interface',
)
loaded = [
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
]
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
