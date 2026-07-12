# Copyright (c) 2025 TeleAI-infra Team (TeleTron)
# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0

"""Safe configuration serialization and dotted-object imports."""

from __future__ import annotations

import json
import os
import pickle
from importlib import import_module

import yaml

try:
    from yaml import CSafeDumper as SafeDumper
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeDumper, SafeLoader


def load_file(file_path, *, allow_unsafe_pickle: bool = False, **kwargs):
    """Load JSON/safe-YAML, or explicitly trusted pickle input."""

    file_path = os.fspath(file_path)
    if file_path.endswith((".pkl", ".pickle")):
        if not allow_unsafe_pickle:
            raise ValueError("Refusing to load pickle without allow_unsafe_pickle=True; pickle files must come from a trusted source.")
        with open(file_path, "rb") as stream:
            return pickle.load(stream, **kwargs)
    if file_path.endswith(".json"):
        with open(file_path, encoding="utf-8") as stream:
            return json.load(stream, **kwargs)
    if file_path.endswith((".yaml", ".yml")):
        if "Loader" in kwargs:
            raise TypeError("Custom YAML loaders are not accepted; YAML is loaded safely")
        with open(file_path, encoding="utf-8") as stream:
            return yaml.load(stream, Loader=SafeLoader, **kwargs)
    raise ValueError(f"Unsupported config file extension: {file_path}")


def save_file(file_path, data, **kwargs) -> None:
    """Save JSON, safe YAML, or an explicitly selected pickle file."""

    file_path = os.fspath(file_path)
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if file_path.endswith((".pkl", ".pickle")):
        with open(file_path, "wb") as stream:
            pickle.dump(data, stream, **kwargs)
    elif file_path.endswith(".json"):
        kwargs.setdefault("indent", 4)
        with open(file_path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, **kwargs)
    elif file_path.endswith((".yaml", ".yml")):
        if "Dumper" in kwargs:
            raise TypeError("Custom YAML dumpers are not accepted; YAML is saved safely")
        with open(file_path, "w", encoding="utf-8") as stream:
            yaml.dump(data, stream, Dumper=SafeDumper, **kwargs)
    else:
        raise ValueError(f"Unsupported config file extension: {file_path}")


def import_function(function_name: str, sep: str = "."):
    """Resolve a dotted object path without coupling config I/O to datasets."""

    parts = function_name.split(sep)
    module_name = ".".join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])


__all__ = ["import_function", "load_file", "save_file"]
