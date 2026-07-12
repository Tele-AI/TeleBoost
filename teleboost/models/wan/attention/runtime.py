# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""TeleBoost-owned attention adapter for the upstream Wan runtime.

TeleBoost keeps this integration outside the upstream package: a Wan model is
imported first, then its module-level attention bindings are replaced
explicitly before model construction. Installation is a namespace singleton
and reversible for tests and tools. The returned handle is owned by that
namespace's model lifecycle; repeated calls return the same handle rather than
independent leases.

This adapter is deliberately not part of :func:`teleboost.patches.apply`.
That process-wide lifecycle is gated by ``verl`` and covers verl drift shims;
the Wan adapter is installed only by the Wan model lifecycle.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from types import ModuleType

from teleboost.models.wan.attention.ops import (
    wan_attention,
    wan_flash_attention,
)


@dataclass
class _Binding:
    module: ModuleType
    name: str
    original: object
    replacement: object


@dataclass
class WanAttentionAdapterHandle:
    """A reversible set of module bindings for one Wan package namespace."""

    namespace: str
    original_flash_attention: object
    original_attention: object
    _bindings: list[_Binding] = field(default_factory=list)
    active: bool = True

    def uninstall(self) -> bool:
        if not self.active:
            return False
        for binding in reversed(self._bindings):
            if getattr(binding.module, binding.name, None) is binding.replacement:
                setattr(binding.module, binding.name, binding.original)

        # clip/model may have been imported after installation and cached the
        # replacement. Restore only our exact object so a later owner is never
        # overwritten during teardown.
        for module_name in (
            f"{self.namespace}.modules.model",
            f"{self.namespace}.modules.clip",
        ):
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, "flash_attention", None) is wan_flash_attention:
                setattr(module, "flash_attention", self.original_flash_attention)

        modules_package = sys.modules.get(f"{self.namespace}.modules")
        if modules_package is not None and vars(modules_package).get("flash_attention") is wan_flash_attention:
            # The package exposes this name lazily. If it was first resolved
            # while our adapter was active, delete the cache and restore that
            # lazy state; a pre-existing cache is already covered by bindings.
            delattr(modules_package, "flash_attention")

        self.active = False
        if _HANDLES.get(self.namespace) is self:
            del _HANDLES[self.namespace]
        return True


_HANDLES: dict[str, WanAttentionAdapterHandle] = {}


def _bind(
    handle: WanAttentionAdapterHandle,
    module: ModuleType,
    name: str,
    replacement: object,
) -> None:
    original = getattr(module, name)
    if original is replacement:
        return
    setattr(module, name, replacement)
    handle._bindings.append(_Binding(module, name, original, replacement))


def install_wan_attention_adapter(namespace: str = "wan") -> WanAttentionAdapterHandle:
    """Install the adapter into an imported Wan namespace.

    Production uses the canonical ``"wan"`` namespace. The argument remains
    configurable so isolated tests can provide a synthetic namespace. The
    function patches bindings already cached by ``model``/``clip`` and leaves
    future imports to bind from the patched attention module.
    """

    namespace = namespace.rstrip(".")
    current = _HANDLES.get(namespace)
    if current is not None and current.active:
        return current

    attention_module = importlib.import_module(f"{namespace}.modules.attention")
    original_flash = attention_module.flash_attention
    original_attention = attention_module.attention
    handle = WanAttentionAdapterHandle(
        namespace=namespace,
        original_flash_attention=original_flash,
        original_attention=original_attention,
    )
    _bind(handle, attention_module, "flash_attention", wan_flash_attention)
    _bind(handle, attention_module, "attention", wan_attention)

    loaded_targets = (
        (f"{namespace}.modules.model", "flash_attention", wan_flash_attention),
        (f"{namespace}.modules.clip", "flash_attention", wan_flash_attention),
        (f"{namespace}.modules", "flash_attention", wan_flash_attention),
    )
    for module_name, attribute, replacement in loaded_targets:
        module = sys.modules.get(module_name)
        # Inspect __dict__ directly: hasattr() would invoke the vendored lazy
        # __getattr__ and create a cache merely because the adapter installed.
        if module is not None and attribute in vars(module):
            _bind(handle, module, attribute, replacement)

    _HANDLES[namespace] = handle
    return handle


def uninstall_wan_attention_adapter(namespace: str = "wan") -> bool:
    """Undo a prior installation without disturbing later foreign patches."""

    handle = _HANDLES.get(namespace.rstrip("."))
    return False if handle is None else handle.uninstall()


def is_wan_attention_adapter_installed(namespace: str = "wan") -> bool:
    handle = _HANDLES.get(namespace.rstrip("."))
    return bool(handle is not None and handle.active)
