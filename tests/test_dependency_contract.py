# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep install manifests synchronized with the validated runtime."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _key_value_file(name: str) -> dict[str, str]:
    values = {}
    for raw in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _constraints() -> dict[str, Requirement]:
    requirements = {}
    for raw in (ROOT / "constraints/torch2.9-cu128.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            requirements[requirement.name.lower()] = requirement
    return requirements


def test_python_and_pinned_stack_contract_are_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pins = _constraints()

    assert project["requires-python"] == ">=3.11"
    assert str(pins["pandas"].specifier) == "==3.0.1"
    assert str(pins["torch"].specifier) == "==2.9.1"
    assert str(pins["verl"].specifier) == "==0.7.1"
    assert str(pins["vllm"].specifier) == "==0.14.0"
    assert str(pins["flash-attn"].specifier) == "==2.8.3.post1"


def test_standard_requirements_is_a_single_manifest_entrypoint():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "-c constraints/torch2.9-cu128.txt" in requirements
    assert "-e .[train,wan,dpo,reward]" in requirements


def test_wan_runtime_uses_only_the_top_level_installed_package():
    launchers = [
        (ROOT / "recipes/wan_grpo_fsdp/run.sh").read_text(encoding="utf-8"),
        (ROOT / "recipes/wan_dpo_teletron/run.sh").read_text(encoding="utf-8"),
    ]
    for launcher in launchers:
        assert "import wan" in launcher
        assert "project_root}/third_party" not in launcher
        assert "third_party.wan" not in launcher

    preprocessor = (ROOT / "teleboost/datasets/preprocessing/wan.py").read_text(encoding="utf-8")
    assert 'import_module("wan.configs")' in preprocessor
    assert 'import_module("wan.modules.t5")' in preprocessor
    assert "third_party" not in preprocessor


def test_external_source_revisions_are_immutable_and_documented():
    verl = _key_value_file("constraints/upstreams/verl.txt")
    megatron = _key_value_file("constraints/upstreams/megatron-lm.txt")
    flash3 = _key_value_file("constraints/upstreams/flash-attn-3.txt")
    install = (ROOT / "INSTALL.md").read_text(encoding="utf-8")
    flash3_installer = (ROOT / "tools/install_flash_attn_3.sh").read_text(encoding="utf-8")

    assert verl["VERSION"] == "0.7.1"
    assert re.fullmatch(r"[0-9a-f]{7,40}", verl["UPSTREAM_COMMIT"])
    assert re.fullmatch(r"[0-9a-f]{40}", megatron["UPSTREAM_COMMIT"])
    assert re.fullmatch(r"[0-9a-f]{40}", flash3["UPSTREAM_COMMIT"])
    assert re.fullmatch(r"[0-9a-f]{40}", flash3["CUTLASS_COMMIT"])
    assert flash3["UPSTREAM_SUBDIRECTORY"] == "hopper"
    assert flash3["BUILD_PROFILE"] == "teleboost.wan.sm90"
    assert megatron["UPSTREAM_COMMIT"] in install
    assert "constraints/upstreams/flash-attn-3.txt" in install
    assert "--no-deps --force-reinstall" in flash3_installer
    assert "submodule update --init --depth 1 csrc/cutlass" in flash3_installer
    assert "root package does not bundle the Wan\nupstream runtime" in install
