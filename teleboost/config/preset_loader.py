# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Load official TeleBoost config presets from the installed package."""

from __future__ import annotations

from importlib import resources
from typing import Any

import yaml

_PRESET_ROOT = "teleboost.config.presets"


class PresetNotFoundError(LookupError):
    """Requested official preset is not present in the installed package."""


def _read_yaml(relative_path: str) -> dict[str, Any]:
    root = resources.files(_PRESET_ROOT)
    path = root.joinpath(*relative_path.split("/"))
    if not path.is_file():
        raise PresetNotFoundError(f"Unknown TeleBoost preset: {relative_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset {relative_path!r} must contain a mapping")
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Preset {relative_path!r} field 'overrides' must be a mapping")
    return data


def load_program_preset(program: str, name: str) -> dict[str, Any]:
    """Load ``presets/programs/<program>/<name>.yaml``.

    Dotted program names use the directory form used by ``recipes/``.
    """

    program_dir = str(program).strip().lower().replace(".", "_")
    preset_name = str(name).strip().lower()
    return _read_yaml(f"programs/{program_dir}/{preset_name}.yaml")


def load_overlay_preset(name: str) -> dict[str, Any]:
    """Load ``presets/overlays/<name>.yaml``."""

    preset_name = str(name).strip().lower()
    return _read_yaml(f"overlays/{preset_name}.yaml")


__all__ = ["PresetNotFoundError", "load_overlay_preset", "load_program_preset"]
