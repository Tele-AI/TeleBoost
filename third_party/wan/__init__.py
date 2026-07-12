# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified by TeleBoost contributors in 2026 to avoid importing every model,
# tokenizer, and pipeline when a single Wan submodule is requested.
"""Lazy public exports for the vendored Wan package."""

from importlib import import_module

_EXPORTS = {
    "configs": (".configs", None),
    "distributed": (".distributed", None),
    "modules": (".modules", None),
    "WanI2V": (".image2video", "WanI2V"),
    "WanT2V": (".text2video", "WanT2V"),
    "WanFLF2V": (".first_last_frame2video", "WanFLF2V"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
