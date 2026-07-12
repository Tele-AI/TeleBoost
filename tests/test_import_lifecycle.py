# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""The library package must not mutate third-party runtimes on import."""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_import_teleboost_does_not_apply_runtime_patches():
    result = _run_isolated("import teleboost; import teleboost.patches as p; assert p._APPLIED is False")

    assert result.returncode == 0, result.stderr


def test_import_teleboost_does_not_load_wan_or_flash_extensions():
    result = _run_isolated("import sys; import teleboost; assert not any(n == 'wan' or n.startswith('wan.') for n in sys.modules); assert 'flash_attn' not in sys.modules; assert 'flash_attn_3' not in sys.modules; assert 'flash_attn_interface' not in sys.modules")

    assert result.returncode == 0, result.stderr


def test_import_program_selection_does_not_bootstrap_family_or_verl_runtime():
    result = _run_isolated("import sys; import teleboost.programs.selection; import teleboost.patches as p; assert p._APPLIED is False; assert 'verl' not in sys.modules; assert 'teleboost.programs.wan.backend' not in sys.modules; assert 'teleboost.models.wan.family' not in sys.modules")

    assert result.returncode == 0, result.stderr


def test_removed_backend_facades_are_not_retained():
    result = _run_isolated(
        "import importlib.util\n"
        "def exists(name):\n"
        "    try:\n"
        "        return importlib.util.find_spec(name) is not None\n"
        "    except ModuleNotFoundError:\n"
        "        return False\n"
        "removed = [\n"
        "    'recipes.grpo.backend',\n"
        "    'recipes.grpo.trainer',\n"
        "    'teleboost.runtime.transfer_queue',\n"
        "    'teleboost.workers.sharding_manager.diffusion',\n"
        "    'teleboost.workers.sharding_manager.identity',\n"
        "    'teleboost.workers.sharding_manager.runtime',\n"
        "    'teleboost.training.core.fsdp',\n"
        "    'teleboost.integrations.verl_fsdp_merge',\n"
        "    'teleboost.models.wan_teletron',\n"
        "    'teleboost.models.wan_family',\n"
        "    'teleboost.models.wan22_dual_model',\n"
        "    'recipes.bgpo.trainer',\n"
        "    'recipes.bgpo.algorithm',\n"
        "    'recipes.vipo.trainer',\n"
        "    'recipes.vipo.algorithm',\n"
        "    'recipes.tempflow.trainer',\n"
        "    'recipes.backends',\n"
        "    'recipes.backends.selection',\n"
        "    'recipes.backends.wan',\n"
        "    'recipes.backends.common',\n"
        "    'teleboost.runtime.model_backends',\n"
        "]\n"
        "present = [name for name in removed if exists(name)]\n"
        "assert not present, present\n"
    )

    assert result.returncode == 0, result.stderr


def test_grpo_worker_bootstrap_boundary_calls_explicit_lifecycle():
    result = _run_isolated("import teleboost; calls = []; teleboost.apply_runtime_patches = lambda *, require_verl=True: calls.append(require_verl) or True; import teleboost.patches.lifecycle as runtime; assert runtime.PATCHES_APPLIED is True; assert calls == [True]")

    assert result.returncode == 0, result.stderr


def test_runtime_patches_are_an_explicit_idempotent_lifecycle():
    result = _run_isolated(
        "import teleboost; import teleboost.patches as p; "
        "assert teleboost.apply_runtime_patches() is True; "
        "assert p._APPLIED is True; "
        "assert teleboost.apply_runtime_patches() is True; "
        "from teleboost.models.wan.attention.runtime import is_wan_attention_adapter_installed; "
        "assert not is_wan_attention_adapter_installed()"
    )

    assert result.returncode == 0, result.stderr
