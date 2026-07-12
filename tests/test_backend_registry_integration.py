# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Integration coverage for built-in and external backend selection."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from teleboost.programs.selection import select_backend, select_backend_registration
from teleboost.programs import (
    BACKEND_ENTRY_POINT_GROUP,
    BackendRegistration,
    BackendRegistry,
)
from teleboost.programs.backend_metadata import WAN_BACKEND
from teleboost.programs.backends import register_program_backends


class _ExternalBackend:
    name = "acme"

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
        return list

    def trainer_cls(self, config: Any) -> type[Any]:
        return object


@dataclass
class _EntryPoint:
    registration: BackendRegistration
    name: str = "acme"
    group: str = BACKEND_ENTRY_POINT_GROUP
    value: str = "acme.backend:registration"
    loads: int = 0

    def load(self) -> BackendRegistration:
        self.loads += 1
        return self.registration


def _registry(entry_point: _EntryPoint | None = None):
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return () if entry_point is None else (entry_point,)

    registry = BackendRegistry(entry_point_provider=provider)
    register_program_backends(registry)
    return registry, calls


def _external_entry_point(factory=_ExternalBackend) -> _EntryPoint:
    return _EntryPoint(
        BackendRegistration(
            name="acme",
            factory=factory,
            runtime_types=frozenset({"acme-runtime"}),
            capabilities=frozenset({"media.custom"}),
        )
    )


def test_builtin_registration_metadata_is_complete_and_exact():
    assert WAN_BACKEND.capabilities == {
        "algorithm.grpo",
        "generation.latent",
        "media.video",
        "parallel.context",
        "parallel.fsdp",
        "reward.video_vlm",
    }


def test_public_registry_does_not_auto_install_program_builtins():
    code = """
import sys
from teleboost.programs import get_backend_registry
from teleboost.programs import BackendNotFoundError

registry = get_backend_registry()
try:
    registry.resolve_canonical('wan')
except BackendNotFoundError:
    pass
else:
    raise AssertionError('public registry unexpectedly auto-installed program builtins')
assert not any(name.startswith('recipes.backends') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_program_backend_registration_and_factory_imports_are_lazy():
    code = """
import sys
from teleboost.programs import get_backend_registry
from teleboost.programs.backends import register_program_backends

registry = get_backend_registry()
register_program_backends(registry)
assert registry.resolve_canonical('wan').name == 'wan'
family_modules = {
    'teleboost.programs.wan.backend',
}
assert family_modules.isdisjoint(sys.modules), family_modules & set(sys.modules)

backend = registry.create('wan')
assert backend.name == 'wan'
assert 'teleboost.programs.wan.backend' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_external_plugin_is_discovered_only_by_exact_backend_name():
    entry_point = _external_entry_point()
    registry, calls = _registry(entry_point)
    config = {
        "backend": {"name": "acme"},
        "actor_rollout_ref": {"type": "acme-runtime"},
        "trainer": {"type": "acme-runtime"},
        "data": {"type": "acme-runtime"},
    }

    backend = select_backend(config, registry=registry)

    assert isinstance(backend, _ExternalBackend)
    assert calls == [{"group": BACKEND_ENTRY_POINT_GROUP, "name": "acme"}]
    assert entry_point.loads == 1


def test_external_runtime_type_never_triggers_plugin_discovery():
    entry_point = _external_entry_point()
    registry, calls = _registry(entry_point)
    config = {"actor_rollout_ref": {"type": "acme-runtime"}}

    with pytest.raises(ValueError, match="exact backend.name"):
        select_backend_registration(config, registry=registry)

    assert calls == []
    assert entry_point.loads == 0


def test_selected_plugin_runtime_types_are_cross_checked_after_registration():
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return _ExternalBackend()

    entry_point = _external_entry_point(factory)
    registry, _calls = _registry(entry_point)
    config = {
        "backend": {"name": "acme"},
        "actor_rollout_ref": {"type": "wan"},
    }

    with pytest.raises(ValueError, match="not registered runtime types"):
        select_backend(config, registry=registry)

    assert entry_point.loads == 1
    assert factory_calls == []


def test_plugin_documentation_tracks_the_public_registry_contract():
    root = Path(__file__).resolve().parents[1]
    architecture = (root / "recipes/README.md").read_text(encoding="utf-8")
    support_matrix = (root / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")

    assert '[project.entry-points."teleboost.programs"]' in architecture
    assert "teleboost/programs/backend_metadata.py" in architecture
    assert "teleboost/runtime/model_backends.py" not in architecture
    assert "external backend entry-point registry" in support_matrix
