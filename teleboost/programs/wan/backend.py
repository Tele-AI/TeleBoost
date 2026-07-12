# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Wan model-family backend implementation."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from teleboost.programs.common import (
    enabled_driver_phase_algorithms,
    require_actor_strategy,
)
from teleboost.reward.routing import (
    VIDEO_VLM_ADAPTER,
    is_video_vlm_reward_config,
    reward_model_adapter,
)

logger = logging.getLogger(__name__)


class WanBackendSpec:
    """Construction hooks for a Wan-family GRPO run."""

    name = "wan"

    def validate_capabilities(self, config: Any) -> None:
        require_actor_strategy(config, backend_name=self.name, supported={"fsdp"})

    def validate_reward(self, config: Any) -> None:
        rm_type = config.reward.reward_model.type
        adapter = reward_model_adapter(config)
        valid_types = ["single", "joint"]
        if rm_type not in valid_types:
            raise ValueError(f"Invalid reward.reward_model.type: {rm_type}. Must be one of {valid_types}")
        if adapter and adapter != VIDEO_VLM_ADAPTER:
            raise ValueError(f"Invalid reward.reward_model.adapter: {adapter}. Must be {VIDEO_VLM_ADAPTER!r}")
        if rm_type == "joint" and adapter:
            raise ValueError("reward.reward_model.type=joint cannot be combined with a VLM adapter; joint registry-backed heads and the vLLM reward router are separate, and joint VLM aggregation is not implemented")

    def prepare_tokenizer(self, config: Any) -> tuple[Any, Any]:
        """Download the Wan checkpoint and build its tokenizer/processor."""

        from teleboost.models.wan.family import TOKENIZER_SUBPATH
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        try:
            local_path = copy_to_local(config.actor_rollout_ref.model.path)
            logger.info("Model downloaded to %s", local_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download model from {config.actor_rollout_ref.model.path}: {exc}") from exc

        tokenizer_subpath = config.actor_rollout_ref.get(
            "tokenizer_subpath",
            TOKENIZER_SUBPATH,
        )
        tokenizer_path = os.path.join(local_path, tokenizer_subpath)
        try:
            tokenizer = hf_tokenizer(tokenizer_path)
            processor = hf_processor(local_path, use_fast=True)
            logger.info("Loaded tokenizer from %s", tokenizer_path)
            return tokenizer, processor
        except Exception as exc:
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {exc}") from exc

    def resolve_worker_and_group(self, config: Any) -> tuple[type[Any], type[Any]]:
        strategy = require_actor_strategy(
            config,
            backend_name=self.name,
            supported={"fsdp"},
        )

        use_critic = config.algorithm.get("adv_estimator", "grpo") == "gae"
        if use_critic and hasattr(config, "critic"):
            if strategy != config.critic.strategy:
                raise ValueError(f"Actor strategy ({strategy}) must match critic strategy ({config.critic.strategy})")

        if strategy == "fsdp":
            from verl.single_controller.ray import RayWorkerGroup

            from teleboost.training.families.wan.fsdp_worker import WanActorRolloutRefWorker

            return RayWorkerGroup, WanActorRolloutRefWorker

        raise NotImplementedError(f"Wan GRPO currently requires actor_rollout_ref.actor.strategy=fsdp; got {strategy!r}. Megatron is implemented only by teleboost.programs.wan.dpo.")

    def register_reward_workers(
        self,
        config: Any,
        role_worker_mapping: dict[Any, Any],
        mapping: dict[Any, Any],
        global_pool_id: str,
    ) -> None:
        from verl.trainer.ppo.ray_trainer import Role

        if not bool(config.trainer.get("use_rm", True)):
            return

        strategy = config.reward.reward_model.strategy
        rm_type = config.reward.reward_model.type

        def register_role(role: Any, worker_cls: type[Any]) -> None:
            import ray

            role_worker_mapping[role] = ray.remote(worker_cls)
            mapping[role] = global_pool_id
            logger.info("Registered %s with worker %s", role, worker_cls.__name__)

        if strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker

            register_role(Role.RewardModel, RewardModelWorker)
            return

        if strategy != "diffusion":
            raise NotImplementedError(f"Unknown reward model strategy: {strategy}")

        if is_video_vlm_reward_config(config):
            # verl owns the colocated vLLM reward lifecycle, but its resource
            # manager still requires a pool mapping for Role.RewardModel.
            mapping[Role.RewardModel] = global_pool_id
            logger.info("video_vlm reward delegated to verl-managed RewardModelManager (colocated on global_pool)")
        elif rm_type == "joint":
            from teleboost.reward.execution.worker import (
                JointRewardModelWorker,
            )

            register_role(Role.RewardModel, JointRewardModelWorker)
            logger.info("Using JointRewardModelWorker for joint mode")
        else:
            from teleboost.reward.execution.worker import (
                UnifiedRewardModelWorker,
            )

            register_role(Role.RewardModel, UnifiedRewardModelWorker)
            logger.info(
                "Using UnifiedRewardModelWorker for single mode (type=%r)",
                rm_type,
            )

    def collate_fn(self, config: Any) -> Callable[..., Any]:
        del config
        from verl.utils.dataset.rl_dataset import wan_preprocessed_collate_function

        return wan_preprocessed_collate_function

    def trainer_cls(self, config: Any) -> type[Any]:
        """Select exactly one Wan driver-phase algorithm trainer."""

        from teleboost.training.families.wan import RayWanTrainer

        enabled = enabled_driver_phase_algorithms(config)
        if len(enabled) > 1:
            raise ValueError(f"Multiple driver-phase algorithms enabled: {enabled}. Each run selects ONE per-algorithm trainer; to combine, define an explicit combination trainer (see recipes/README.md).")
        if not enabled:
            return RayWanTrainer
        if enabled[0] == "bgpo":
            from teleboost.programs.wan.bgpo import RayBGPOTrainer

            return RayBGPOTrainer
        if enabled[0] == "vipo":
            from teleboost.programs.wan.vipo import RayVIPOTrainer

            return RayVIPOTrainer

        from teleboost.programs.wan.tempflow import RayTempFlowTrainer

        return RayTempFlowTrainer


__all__ = ["WanBackendSpec"]
