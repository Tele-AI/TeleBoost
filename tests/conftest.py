# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# conftest.py
import importlib
import multiprocessing as mp
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# These modules import the real verl/TensorDict/Ray stack at collection time.
# In the core profile they are excluded before import; in the training profile
# missing dependencies are a hard configuration error instead of a green skip.
_TRAINING_ENV_MODULES = {
    "tests/test_backend_startup_validation_training.py",
    "tests/test_csv_dpo_dataset.py",
    "tests/test_grpo_mismatch_diagnostics.py",
    "tests/test_joint_reward_collectives.py",
    "tests/test_reward_device_scope.py",
    "tests/test_review_hardening.py",
    "tests/test_teleboost_algorithm_integration.py",
    "tests/test_transition_parity.py",
    "tests/test_validation_video_save.py",
    "tests/test_video_vlm_adapter.py",
    "tests/test_vlm_media_uuid.py",
    "tests/test_vllm_tokenizer_bootstrap.py",
    "tests/test_wan_attention_fallback.py",
    "tests/test_wan_attention_patch.py",
    "tests/test_wan_patches_idempotent.py",
}

_HEAVY_ENV_MODULES = {
    "tests/unit_tests/models/test_dpo_i2v_cp_compare.py",
    "tests/unit_tests/models/test_parallel_wan_teletron_model.py",
    "tests/unit_tests/teletron/context_parallel/test_context_parallel_mixin.py",
    "tests/unit_tests/teletron/context_parallel/test_cp_attention.py",
}

_HEAVY_LANE_MODULES = {
    "wan": {
        "tests/unit_tests/models/test_dpo_i2v_cp_compare.py",
        "tests/unit_tests/models/test_parallel_wan_teletron_model.py",
        "tests/unit_tests/teletron/context_parallel/test_context_parallel_mixin.py",
        "tests/unit_tests/teletron/context_parallel/test_cp_attention.py",
    },
}

# This is a torchrun executable, not a pytest test module (it has ``main`` and
# no test functions).  Keeping it out of pytest collection avoids reporting a
# misleading dependency skip; tests/README.md documents its explicit command.
_STANDALONE_DISTRIBUTED = {
    "tests/special_distributed/test_cp_grad_reduce.py",
    "tests/unit_tests/teletron/context_parallel/test_context_parallel_mixin_stateless.py",
    "tests/unit_tests/teletron/context_parallel/test_forward_attn_precision.py",
}


def pytest_addoption(parser):
    group = parser.getgroup("teleboost test profiles")
    group.addoption(
        "--profile",
        choices=("core", "training", "heavy"),
        default="core",
        help="select exactly one TeleBoost test profile (default: core)",
    )
    group.addoption(
        "--heavy-lane",
        choices=tuple(sorted(_HEAVY_LANE_MODULES)),
        default=None,
        help="required with --profile=heavy: wan",
    )


def _import_required(names: tuple[str, ...], *, profile: str) -> None:
    failures = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise pytest.UsageError(f"--profile={profile} dependency import failed: " + " | ".join(failures))


def _validate_distribution_versions(
    required_versions: dict[str, str],
    *,
    contract: str,
) -> None:
    mismatched = []
    for distribution, expected in required_versions.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            mismatched.append(f"{distribution}=<unknown> (expected {expected})")
            continue
        if installed != expected:
            mismatched.append(f"{distribution}={installed} (expected {expected})")
    if mismatched:
        raise pytest.UsageError(f"{contract} version contract failed: " + ", ".join(mismatched))


def _validate_training_versions() -> None:
    _validate_distribution_versions(
        {
            "diffusers": "0.39.0",
            "numpy": "1.26.4",
            "ray": "2.56.0",
            "tensordict": "0.10.0",
            "torch": "2.9.1",
            "transformers": "4.57.6",
            "verl": "0.7.1",
        },
        contract="training",
    )
    # vLLM is an optional, mutually exclusive inference profile. If present,
    # enforce the TeleBoost-tested override rather than silently accepting an
    # arbitrary version outside verl's own support range.
    try:
        installed_vllm = version("vllm")
    except PackageNotFoundError:
        return
    if installed_vllm != "0.14.0":
        raise pytest.UsageError(f"training version contract failed: vllm={installed_vllm} (expected 0.14.0 when installed)")


def _preflight_training() -> None:
    _import_required(
        (
            "verl",
            "tensordict",
            "ray",
            "datasets",
            "diffusers",
            "easydict",
            "peft",
            "hydra",
        ),
        profile="training",
    )
    _validate_training_versions()


def _require_idle_gpus(lane: str, count: int) -> None:
    # Shared-card heavy failures are ambiguous (neighbor OOM vs real bug), so
    # refuse up front instead of failing later. A green run on shared cards is
    # still a green run — opt in with TELEBOOST_HEAVY_SHARED_GPUS_OK=1.
    if os.environ.get("TELEBOOST_HEAVY_SHARED_GPUS_OK", "0") == "1":
        return
    import torch

    busy = []
    for index in range(count):
        free, total = torch.cuda.mem_get_info(index)
        used_gib = (total - free) / 2**30
        if used_gib > 2.0:
            busy.append(f"cuda:{index} has {used_gib:.1f}GiB in use")
    if busy:
        raise pytest.UsageError(f"--profile=heavy --heavy-lane={lane} wants exclusive GPUs: " + "; ".join(busy) + ". Set TELEBOOST_HEAVY_SHARED_GPUS_OK=1 to run on shared cards anyway.")


def _preflight_heavy(lane: str) -> None:
    _import_required(("torch",), profile=f"heavy/{lane}")
    import torch

    minimum_gpus = 4
    if torch.cuda.device_count() < minimum_gpus:
        raise pytest.UsageError(f"--profile=heavy --heavy-lane={lane} needs at least {minimum_gpus} visible GPUs; found {torch.cuda.device_count()}")
    _require_idle_gpus(lane, torch.cuda.device_count())
    _validate_distribution_versions(
        {"megatron-core": "0.16.1"},
        contract="heavy/wan",
    )
    _import_required(
        ("verl", "tensordict", "megatron", "einops"),
        profile="heavy/wan",
    )
    _validate_training_versions()


def _validate_profile_manifests() -> None:
    lane_union = set().union(*_HEAVY_LANE_MODULES.values())
    if lane_union != _HEAVY_ENV_MODULES:
        raise pytest.UsageError("heavy test manifest drift: heavy modules must equal the lane union")
    overlaps = _TRAINING_ENV_MODULES & _HEAVY_ENV_MODULES
    if overlaps:
        raise pytest.UsageError("test profile manifests overlap: " + ", ".join(sorted(overlaps)))
    declared = _TRAINING_ENV_MODULES | _HEAVY_ENV_MODULES | _STANDALONE_DISTRIBUTED
    missing = sorted(path for path in declared if not (_REPO_ROOT / path).is_file())
    if missing:
        raise pytest.UsageError("test profile manifest references missing files: " + ", ".join(missing))


def pytest_configure(config):
    _validate_profile_manifests()
    profile = config.getoption("--profile")
    explicit_standalone = []
    wrong_profile = []
    for argument in config.args:
        relative = _relative_test_path(str(argument).split("::", 1)[0])
        if relative in _STANDALONE_DISTRIBUTED:
            explicit_standalone.append(relative)
        # Explicitly named paths bypass pytest_ignore_collect, so an env-gated
        # module named under the wrong profile would die on its own imports
        # (raw RuntimeError) instead of the profile contract. Refuse loudly.
        elif profile != "training" and relative in _TRAINING_ENV_MODULES:
            wrong_profile.append(f"{relative} (needs --profile=training)")
        elif profile != "heavy" and relative in _HEAVY_ENV_MODULES:
            wrong_profile.append(f"{relative} (needs --profile=heavy --heavy-lane=<lane>)")
    if explicit_standalone:
        raise pytest.UsageError("standalone distributed programs must be run with torchrun, not pytest: " + ", ".join(sorted(explicit_standalone)))
    if wrong_profile:
        raise pytest.UsageError("environment-gated test modules named under the wrong profile: " + ", ".join(sorted(wrong_profile)))

    lane = config.getoption("--heavy-lane")
    markexpr = str(getattr(config.option, "markexpr", "") or "")
    if profile == "core" and ("training_env" in markexpr or "heavy_env" in markexpr):
        raise pytest.UsageError("Do not select environment markers directly; use --profile=training or --profile=heavy --heavy-lane=<lane>")
    if profile == "training":
        if lane is not None:
            raise pytest.UsageError("--heavy-lane is only valid with --profile=heavy")
        _preflight_training()
    elif profile == "heavy":
        if lane is None:
            raise pytest.UsageError("--profile=heavy requires --heavy-lane=wan")
        _preflight_heavy(lane)
    elif lane is not None:
        raise pytest.UsageError("--heavy-lane is only valid with --profile=heavy")


def _relative_test_path(path) -> str:
    try:
        return Path(str(path)).resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return Path(str(path)).as_posix()


def pytest_ignore_collect(collection_path, config):
    relative = _relative_test_path(collection_path)
    if relative in _STANDALONE_DISTRIBUTED:
        return True
    path = Path(str(collection_path))
    if not path.is_file() or not path.name.startswith("test_") or path.suffix != ".py":
        return None

    profile = config.getoption("--profile")
    if profile == "training":
        return None if relative in _TRAINING_ENV_MODULES else True
    if profile == "heavy":
        selected = _HEAVY_LANE_MODULES[config.getoption("--heavy-lane")]
        return None if relative in selected else True
    return relative in _TRAINING_ENV_MODULES or relative in _HEAVY_ENV_MODULES


def pytest_collection_modifyitems(config, items):
    for item in items:
        relative = _relative_test_path(item.path)
        if relative in _TRAINING_ENV_MODULES:
            item.add_marker("training_env")
        if relative in _HEAVY_ENV_MODULES:
            item.add_marker("heavy_env")

    profile = config.getoption("--profile")
    lane = config.getoption("--heavy-lane")
    selected_items = []
    deselected_items = []
    for item in items:
        relative = _relative_test_path(item.path)
        if profile == "training":
            selected = item.get_closest_marker("training_env") is not None
        elif profile == "heavy":
            selected = relative in _HEAVY_LANE_MODULES[lane] and item.get_closest_marker("heavy_env") is not None
        else:
            selected = item.get_closest_marker("training_env") is None and item.get_closest_marker("heavy_env") is None
        (selected_items if selected else deselected_items).append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items


# Repository-only upstream fixtures. Production launchers never synthesize
# these paths: deployed Wan runtimes must already expose their canonical
# top-level package. Tests add the fixture root explicitly so the vendored
# mirror resolves with the same ``wan`` import name as an installed runtime.
_VENDORED_FIXTURE_ROOTS = (_REPO_ROOT / "third_party",)
for _fixture_root in _VENDORED_FIXTURE_ROOTS:
    fixture_root = str(_fixture_root)
    if fixture_root not in sys.path:
        sys.path.insert(0, fixture_root)


@pytest.fixture(autouse=True)
def set_spawn_method():
    mp.set_start_method("spawn", force=True)
