# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Config serialization must not turn untrusted files into code execution."""

from __future__ import annotations

import pickle

import pytest
import yaml

from teleboost.config.io import load_file, save_file
from teleboost.engines.teletron.config import Config


def test_yaml_uses_safe_loader(tmp_path):
    marker = tmp_path / "yaml-executed"
    config = tmp_path / "unsafe.yaml"
    config.write_text(
        f"!!python/object/apply:os.system ['touch {marker.as_posix()}']\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        load_file(config)

    assert not marker.exists()


def test_pickle_requires_explicit_trust_opt_in(tmp_path):
    config = tmp_path / "legacy.pkl"
    with config.open("wb") as stream:
        pickle.dump({"answer": 42}, stream)

    with pytest.raises(ValueError, match="allow_unsafe_pickle=True"):
        load_file(config)

    assert load_file(config, allow_unsafe_pickle=True) == {"answer": 42}
    assert Config.load(config, allow_unsafe_pickle=True).answer == 42


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
def test_safe_config_roundtrip(tmp_path, suffix):
    path = tmp_path / f"config.{suffix}"
    payload = {"name": "teleboost", "nested": {"value": 3}}

    save_file(path, payload)

    assert load_file(path) == payload


def test_unknown_config_extension_has_actionable_error(tmp_path):
    path = tmp_path / "config.txt"
    path.write_text("not a config", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported config file extension"):
        load_file(path)
