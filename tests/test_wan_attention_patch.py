# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""The Wan integration belongs to TeleBoost, not the vendored source tree."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

from teleboost import apply_runtime_patches
from teleboost.models.wan.attention.runtime import (
    install_wan_attention_adapter,
    is_wan_attention_adapter_installed,
    wan_attention,
    wan_flash_attention,
)


def _load_wan_modules():
    apply_runtime_patches()

    import wan.modules.attention as attention_module
    import wan.modules.model as model_module

    return attention_module, model_module


attention_module, model_module = _load_wan_modules()


def test_adapter_is_explicit_idempotent_and_reversible():
    original_attention = attention_module.attention
    original_flash = attention_module.flash_attention
    original_model_flash = model_module.flash_attention

    handle = install_wan_attention_adapter("wan")
    try:
        assert install_wan_attention_adapter("wan") is handle
        assert is_wan_attention_adapter_installed("wan")
        assert attention_module.attention is wan_attention
        assert attention_module.flash_attention is wan_flash_attention
        assert model_module.flash_attention is wan_flash_attention
    finally:
        handle.uninstall()

    assert not is_wan_attention_adapter_installed("wan")
    assert attention_module.attention is original_attention
    assert attention_module.flash_attention is original_flash
    assert model_module.flash_attention is original_model_flash
    assert handle.uninstall() is False


def test_vendored_wan_does_not_depend_on_teleboost():
    root = Path(__file__).resolve().parents[1] / "third_party" / "wan"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            if any(name == "teleboost" or name.startswith("teleboost.") for name in names):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def _fake_wan_namespace(monkeypatch, namespace: str):
    package = ModuleType(namespace)
    package.__path__ = []
    modules = ModuleType(f"{namespace}.modules")
    modules.__path__ = []
    attention = ModuleType(f"{namespace}.modules.attention")

    def original_flash(*_args, **_kwargs):
        return "original-flash"

    def original_attention(*_args, **_kwargs):
        return "original-attention"

    attention.flash_attention = original_flash
    attention.attention = original_attention
    monkeypatch.setitem(sys.modules, namespace, package)
    monkeypatch.setitem(sys.modules, f"{namespace}.modules", modules)
    monkeypatch.setitem(sys.modules, f"{namespace}.modules.attention", attention)
    return modules, attention, original_flash, original_attention


def test_install_before_consumers_and_late_uninstall(monkeypatch):
    namespace = "_teleboost_test_wan_late"
    modules, attention, original_flash, _ = _fake_wan_namespace(monkeypatch, namespace)
    handle = install_wan_attention_adapter(namespace)

    model = ModuleType(f"{namespace}.modules.model")
    clip = ModuleType(f"{namespace}.modules.clip")
    model.flash_attention = attention.flash_attention
    clip.flash_attention = attention.flash_attention
    modules.flash_attention = attention.flash_attention
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setitem(sys.modules, clip.__name__, clip)

    assert model.flash_attention is wan_flash_attention
    assert clip.flash_attention is wan_flash_attention
    assert modules.flash_attention is wan_flash_attention
    handle.uninstall()

    assert model.flash_attention is original_flash
    assert clip.flash_attention is original_flash
    assert "flash_attention" not in vars(modules)


def test_uninstall_does_not_overwrite_a_later_foreign_patch(monkeypatch):
    namespace = "_teleboost_test_wan_conflict"
    _, attention, _, _ = _fake_wan_namespace(monkeypatch, namespace)
    model = ModuleType(f"{namespace}.modules.model")
    model.flash_attention = attention.flash_attention
    monkeypatch.setitem(sys.modules, model.__name__, model)

    handle = install_wan_attention_adapter(namespace)

    def foreign_patch(*_args, **_kwargs):
        return "foreign"

    model.flash_attention = foreign_patch
    attention.flash_attention = foreign_patch
    handle.uninstall()

    assert model.flash_attention is foreign_patch
    assert attention.flash_attention is foreign_patch


def test_namespaces_are_isolated(monkeypatch):
    first = "wan"
    second = "_teleboost_test_wan_two"
    _, first_attention, _, _ = _fake_wan_namespace(monkeypatch, first)
    _, second_attention, second_flash, _ = _fake_wan_namespace(monkeypatch, second)

    handle = install_wan_attention_adapter(first)
    try:
        assert first_attention.flash_attention is wan_flash_attention
        assert second_attention.flash_attention is second_flash
    finally:
        handle.uninstall()
