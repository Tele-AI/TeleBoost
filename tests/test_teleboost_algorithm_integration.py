# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Training-stack integration tests for algorithm/trainer composition."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from teleboost.programs.wan.bgpo import BGPOMixin, RayBGPOTrainer
from teleboost.programs.wan.tempflow import RayTempFlowTrainer, TrajectoryBranchMixin
from teleboost.training.families.wan import RayWanTrainer, WanGenerationMixin
from teleboost.programs.wan.vipo import RayVIPOTrainer, VIPOMixin
from teleboost.programs.wan.backend import WanBackendSpec
from teleboost.training.rewarding.joint_reward_trainer import JointRewardMixin
from teleboost.training.core.trainer import RayTeleBoostTrainer


def test_algorithms_namespace_is_depointed():
    """The shared namespace exports math and joint infrastructure only."""
    import teleboost.algorithms as alg

    joint_methods = {m for m in dir(JointRewardMixin) if not m.startswith("__")}
    assert {"_compute_joint_parallel_reward", "_precompute_joint_advantages"} <= joint_methods
    for name in ("BGPOMixin", "VIPOMixin", "TrajectoryBranchMixin"):
        assert not hasattr(alg, name), f"{name} leaked back into the shared namespace"


def test_bgpo_joint_guard_reads_canonical_reward_path():
    """BGPO guard reads the canonical reward.reward_model config path."""

    class TrainerStub(BGPOMixin):
        config = OmegaConf.create(
            {
                "algorithm": {"bgpo": {"enable": True}},
                "reward": {"reward_model": {"type": "joint"}},
            }
        )

    with pytest.raises(ValueError, match="BGPO.*reward_model.type=joint"):
        TrainerStub()._is_bgpo_enabled()


def test_per_algorithm_trainer_composition():
    """Each recipes trainer composes exactly its own extension."""
    base_mro = RayTeleBoostTrainer.__mro__
    assert JointRewardMixin in base_mro
    for ext in (BGPOMixin, VIPOMixin, TrajectoryBranchMixin):
        assert ext not in base_mro, f"{ext.__name__} must not be on the base trainer"

    assert BGPOMixin in RayBGPOTrainer.__mro__
    assert VIPOMixin in RayVIPOTrainer.__mro__
    assert TrajectoryBranchMixin in RayTempFlowTrainer.__mro__
    assert WanGenerationMixin in RayWanTrainer.__mro__
    for trainer_cls in (RayBGPOTrainer, RayVIPOTrainer, RayTempFlowTrainer):
        assert RayWanTrainer in trainer_cls.__mro__

    base = RayTeleBoostTrainer
    assert RayWanTrainer._build_gen_batch is not base._build_gen_batch
    assert RayBGPOTrainer._transform_rewards is not base._transform_rewards
    assert RayBGPOTrainer._transform_advantages is not base._transform_advantages
    assert RayVIPOTrainer._transform_advantages is not base._transform_advantages
    assert RayTempFlowTrainer._pre_rollout_transform is not base._pre_rollout_transform
    assert RayTempFlowTrainer._compute_algorithm_advantage is not base._compute_algorithm_advantage


def test_backend_selects_trainer_from_flags():
    """Backend flags select one trainer and reject silent flag stacking."""
    spec = WanBackendSpec()

    def cfg(bgpo=False, vipo=False, branch=False):
        return OmegaConf.create(
            {
                "algorithm": {"bgpo": {"enable": bgpo}},
                "actor_rollout_ref": {
                    "pixel_weight": {"enable": vipo},
                    "actor": {"tempflow": {"branch": {"enable": branch}}},
                },
            }
        )

    assert spec.trainer_cls(cfg()) is RayWanTrainer
    assert spec.trainer_cls(cfg(bgpo=True)) is RayBGPOTrainer
    assert spec.trainer_cls(cfg(vipo=True)) is RayVIPOTrainer
    assert spec.trainer_cls(cfg(branch=True)) is RayTempFlowTrainer

    with pytest.raises(ValueError, match="Multiple driver-phase algorithms"):
        spec.trainer_cls(cfg(bgpo=True, vipo=True))


def test_model_neutral_base_requires_family_generation_policy():
    trainer = object.__new__(RayTeleBoostTrainer)

    with pytest.raises(NotImplementedError, match="family-owned"):
        trainer._build_gen_batch(SimpleNamespace())


def test_wan_trainer_owns_latent_generation_inputs():
    trainer = object.__new__(RayWanTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "sampling_steps": 2,
                "num_frames": 5,
                "w": 16,
                "h": 16,
                "shift": 1.0,
                "vae_stride": [4, 8, 8],
                "latent_channels": 16,
                "rollout": {"n": 2},
            }
        }
    )
    new_batch = DataProto(
        batch=TensorDict(
            {
                "context": torch.zeros(2, 1),
                "context_orig_lengths": torch.ones(2, dtype=torch.long),
                "null_context": torch.zeros(2, 1),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "caption": np.asarray(["p0", "p1"], dtype=object),
        },
    )

    gen_batch = trainer._build_gen_batch(new_batch)

    assert gen_batch.batch.batch_size[0] == 4
    assert gen_batch.batch["input_latents"].shape == (4, 16, 2, 2, 2)
    assert gen_batch.batch["sigma_schedule"].shape == (4, 3)
    assert list(gen_batch.non_tensor_batch["caption"]) == ["p0", "p0", "p1", "p1"]


def test_vipo_missing_pixel_map_fails_instead_of_falling_back():
    batch = SimpleNamespace(batch={"advantages": torch.ones(2)})

    with pytest.raises(RuntimeError, match="pixel_weight_maps.*missing"):
        VIPOMixin()._apply_vipo_broadcast(batch, {})


def test_vipo_rejects_already_dense_advantages():
    batch = SimpleNamespace(
        batch={
            "advantages": torch.ones(2, 1),
            "pixel_weight_maps": torch.ones(2, 1, 1, 1),
        }
    )

    with pytest.raises(ValueError, match="one scalar advantage per sample"):
        VIPOMixin()._apply_vipo_broadcast(batch, {})


class _BGPOStub(BGPOMixin):
    config = OmegaConf.create(
        {
            "algorithm": {"bgpo": {"enable": True}},
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "reward": {"reward_model": {"type": "single"}},
        }
    )


def test_bgpo_missing_prior_fails_instead_of_falling_back():
    output = SimpleNamespace(batch={"rewards": torch.ones(4)})
    source = SimpleNamespace(non_tensor_batch={})

    with pytest.raises(ValueError, match="no numeric 'prior'"):
        _BGPOStub()._apply_bgpo_on_rewards(output, source, {})


def test_bgpo_accepts_one_prior_per_group_without_divisibility_guessing():
    output = SimpleNamespace(
        batch={
            "rewards": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "advantages": torch.ones(4),
        }
    )
    # Two prompt groups and rollout.n=2.  The previous divisibility heuristic
    # incorrectly collapsed these two distinct priors into one.
    source = SimpleNamespace(non_tensor_batch={"prior": np.asarray([0.25, 0.75])})

    result = _BGPOStub()._apply_bgpo_on_advantages(output, source, {})

    assert result.batch["advantages"].shape == (4,)
    assert result.batch["bgpo_weight"].shape == (4,)


def test_bgpo_rejects_inconsistent_expanded_priors():
    output = SimpleNamespace(batch={"rewards": torch.ones(4)})
    source = SimpleNamespace(non_tensor_batch={"prior": np.asarray([0.1, 0.2, 0.3, 0.3])})

    with pytest.raises(ValueError, match="identical within each prompt group"):
        _BGPOStub()._apply_bgpo_on_rewards(output, source, {})
