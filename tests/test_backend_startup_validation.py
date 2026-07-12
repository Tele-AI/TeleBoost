# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure configuration tests for early Wan backend/reward validation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from teleboost.programs import get_program, program_names
from teleboost.programs.backend_api import BackendSpec
from teleboost.programs.backend_metadata import WAN_BACKEND, WAN_RUNTIME_TYPES
from teleboost.programs.common import require_actor_strategy
from teleboost.programs.selection import select_backend
from teleboost.programs.wan.backend import WanBackendSpec


def _config(
    backend: str,
    *,
    reward_type: str = "single",
    adapter: str = "",
    enable: bool = True,
):
    return OmegaConf.create(
        {
            "trainer": {"type": backend, "use_rm": True},
            "actor_rollout_ref": {
                "type": backend,
                "actor": {
                    "strategy": "fsdp",
                    "tempflow": {"branch": {"enable": False}},
                },
                "pixel_weight": {"enable": False},
            },
            "algorithm": {"bgpo": {"enable": False}},
            "reward": {
                "reward_model": {
                    "enable": enable,
                    "strategy": "diffusion",
                    "type": reward_type,
                    "adapter": adapter,
                }
            },
        }
    )


@pytest.mark.parametrize("alias", ["diffusion", "wan"])
def test_explicit_wan_backend_aliases(alias):
    backend = select_backend(_config(alias))
    assert isinstance(backend, WanBackendSpec)
    assert isinstance(backend, BackendSpec)


def test_builtin_metadata_is_the_single_dependency_light_identity_source():
    assert WAN_BACKEND.runtime_types == WAN_RUNTIME_TYPES == {"diffusion", "wan"}


def test_backend_name_is_preferred_and_compatible_with_runtime_type():
    cfg = _config("diffusion")
    cfg.backend = {"name": "wan"}

    assert isinstance(select_backend(cfg), WanBackendSpec)


@pytest.mark.parametrize(
    "program_name",
    [
        "wan.grpo.fsdp",
        "wan.bgpo.fsdp",
        "wan.vipo.fsdp",
        "wan.tempflow.fsdp",
        "wan.dpo.teletron",
    ],
)
def test_program_name_selects_wan_backend(program_name):
    cfg = _config("diffusion")
    cfg.program = {"name": program_name}

    backend = select_backend(cfg)

    assert isinstance(backend, WanBackendSpec)
    assert backend.name == get_program(program_name).backend_name


def test_unknown_program_name_is_rejected_before_backend_fallback():
    cfg = _config("diffusion")
    cfg.program = {"name": "wan.fake.fsdp"}

    with pytest.raises(ValueError, match=r"Unknown training program.*program\.name"):
        select_backend(cfg)


def test_program_registry_contains_only_public_wan_programs():
    names = program_names()
    assert names == program_names(public_only=True)
    assert names
    assert all(name.startswith("wan.") for name in names)


@pytest.mark.parametrize("backend", ["video_diffusion", "wan21", "wan2.1", "wan22", "diffusin", ""])
def test_unknown_or_removed_backend_aliases_are_rejected(backend):
    cfg = _config(backend) if backend else OmegaConf.create({"trainer": {"use_rm": True}})
    with pytest.raises(ValueError, match="Unknown training backend"):
        select_backend(cfg)


def test_backend_name_accepts_only_canonical_family_name():
    cfg = _config("diffusion")
    cfg.backend = {"name": "diffusion"}

    with pytest.raises(ValueError, match=r"Unknown training backend.*backend\.name"):
        select_backend(cfg)


def test_joint_and_video_vlm_adapter_is_rejected():
    cfg = _config("diffusion", reward_type="joint", adapter="video_vlm")
    with pytest.raises(ValueError, match="joint VLM aggregation is not implemented"):
        select_backend(cfg).validate_reward(cfg)


def test_unknown_reward_adapter_is_rejected():
    cfg = _config("diffusion", adapter="unknown")
    with pytest.raises(ValueError, match="Invalid reward.reward_model.adapter"):
        select_backend(cfg).validate_reward(cfg)


def test_wan_grpo_rejects_megatron_strategy_early():
    cfg = _config("diffusion")
    cfg.actor_rollout_ref.actor.strategy = "megatron"

    with pytest.raises(NotImplementedError, match="requires.*strategy=fsdp"):
        select_backend(cfg).validate_capabilities(cfg)


def test_wan_grpo_requires_an_explicit_strategy():
    cfg = _config("diffusion")
    del cfg.actor_rollout_ref.actor.strategy

    with pytest.raises(ValueError, match="Missing required configuration key"):
        select_backend(cfg).validate_capabilities(cfg)


def test_base_contract_strategy_helper_does_not_impose_fsdp():
    cfg = _config("diffusion")
    cfg.actor_rollout_ref.actor.strategy = "megatron"

    assert require_actor_strategy(cfg, backend_name="test_backend", supported={"megatron"}) == "megatron"
