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
"""
Unified Reward Model Worker

This module provides a unified worker that can dynamically load any reward model
from the Registry based on configuration. This eliminates the need for separate
worker classes for each reward model type.

Usage:
    In config:
        reward.reward_model:
            type: single
            model_name: hps  # Any registered model name
            model_path: /path/to/model
            extra_config:
                model_type: ViT-H-14

    Or for joint mode (handled by ``JointRewardModelWorker`` below):
        reward.reward_model:
            type: joint
            joint:
                models:
                    aesthetic:
                      weight: 1.0
                      extra_config:
                        clip_model_path: /path/to/model
                      ...

    This allows Hydra overrides like:
        reward.reward_model.joint.models.aesthetic.extra_config.clip_model_path=/new/path
"""

import logging
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import (
    DistProfiler as WorkerProfiler,
)
from verl.utils.debug import (
    DistProfilerExtension as WorkerProfilerExtension,
)
from verl.utils.debug import (
    ProfilerConfig,
)
from verl.utils.device import get_device_id

from teleboost.reward.execution.collectives import (
    allgather_variable_batch,
    normalize_gathered_rewards,
    synchronized_failure_count,
)
from teleboost.reward import RewardRegistry, create_reward_model
from teleboost.reward.contract import RewardConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UnifiedRewardModelWorker(Worker, WorkerProfilerExtension):
    """
    Unified Reward Model Worker that dynamically loads models from Registry.

    Handles single-model mode only: loads one model based on ``model_name``.
    ``reward.reward_model.type=joint`` is served by :class:`JointRewardModelWorker`
    below (ALL_TO_ALL dispatch; the selected backend routes it there).

    All models are loaded from the RewardRegistry, ensuring consistent
    initialization and interface.

    Note: This class inherits from Worker (not RewardModelWorker) because
    it manages Registry-based models that have their own initialization logic.
    """

    def __init__(self, config, cuda_visible_devices=None):
        """Initialize the UnifiedRewardModelWorker.

        Args:
            config: Reward model configuration (OmegaConf DictConfig)
            cuda_visible_devices: CUDA visible devices configuration
        """
        Worker.__init__(self, cuda_visible_devices=cuda_visible_devices)
        WorkerProfilerExtension.__init__(self, WorkerProfiler(rank=self.rank, config=ProfilerConfig(**OmegaConf.to_object(config.get("profiler", DictConfig({}))))))
        self.config = config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the reward model(s) based on configuration."""
        # Initialize torch.distributed if not already done
        if not dist.is_initialized():
            from verl.utils.device import get_nccl_backend

            dist.init_process_group(backend=get_nccl_backend())

        self.reward_models: dict[str, Any] = {}
        self.model_configs: dict[str, RewardConfig] = {}

        # Get global rank and world size
        # Use different names to avoid conflict with parent class properties
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        rm_type = self.config.get("type", "single")

        if rm_type == "joint":
            raise ValueError("reward.reward_model.type=joint is served by JointRewardModelWorker (ALL_TO_ALL dispatch); the selected backend routes joint there. UnifiedRewardModelWorker handles single-model mode only.")
        self._init_single_model()

        # Log available models
        available = RewardRegistry.list_available()
        logger.info(f"[UnifiedRM] Available models: {available}")
        logger.info(f"[UnifiedRM] Initialized models: {list(self.reward_models.keys())}")

    def _init_single_model(self):
        """Initialize a single reward model."""
        model_name = self.config.get("model_name", None)

        if model_name is None:
            # Fallback: try to infer from type
            rm_type = self.config.get("type", "single")
            if rm_type in RewardRegistry.list_available():
                model_name = rm_type
            else:
                raise ValueError(f"No model_name specified and type '{rm_type}' is not a registered model. Available models: {RewardRegistry.list_available()}")

        # Canonical flat model_path (matches verl + the VLM adapter); nested
        # model.path is a legacy fallback only.
        model_path = self.config.get("model_path", "") or (self.config.model.get("path", "") if (hasattr(self.config, "model") and self.config.model) else "")

        extra_config = OmegaConf.to_container(self.config.get("extra_config", {}), resolve=True) if self.config.get("extra_config") else {}

        config = RewardConfig(
            name=model_name,
            model_path=model_path,
            weight=1.0,
            dp_fraction=1.0,
            rank_offset=0,
            enabled=True,
            normalize=self.config.get("normalize", True),
            mps_percentage=self.config.get("mps_percentage", 0),
            extra_config=extra_config,
        )

        self._create_and_init_model(model_name, config)

    def _create_and_init_model(self, model_name: str, config: RewardConfig):
        """Create and initialize a reward model from the registry."""
        try:
            model = create_reward_model(
                name=model_name,
                config=config,
                rank=self._global_rank,
                world_size=self._world_size,
            )
            model.init_model()

            self.reward_models[model_name] = model
            self.model_configs[model_name] = config

            logger.info(f"[UnifiedRM] Initialized '{model_name}'")

        except Exception as e:
            logger.error(f"[UnifiedRM] Failed to initialize '{model_name}': {e}")
            raise

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="blue")
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """Compute reward scores with the single loaded model.

        DP_COMPUTE_PROTO dispatch: data arrives already split by the framework.
        """
        start_time = time.time()
        data = data.to(get_device_id())
        return self._compute_single_mode(data, start_time)

    def _compute_single_mode(self, data: DataProto, start_time: float) -> DataProto:
        """Compute scores for single-model mode."""
        model_name = list(self.reward_models.keys())[0]
        model = self.reward_models[model_name]

        # Single mode: data is already split per worker by DP_COMPUTE_PROTO
        result = model.compute_batch_score(data)

        elapsed = time.time() - start_time
        logger.info(f"[UnifiedRM] {model_name} compute time: {elapsed:.2f}s")
        return result


class JointRewardModelWorker(Worker, WorkerProfilerExtension):
    """
    Joint Reward Model Worker for executing multiple reward heads.

    Uses ALL_TO_ALL dispatch so each worker receives the full data and handles
    its own DP splitting. The implemented collective protocol is all-active:
    every enabled model runs on every rank, and ``BaseRewardModel`` rejects
    disjoint ``dp_fraction``/``rank_offset`` configurations at startup.

    This worker intentionally stops at raw reward-head execution. It returns
    ``<name>_rewards`` tensors only; the trainer driver computes per-prompt
    advantages, dynamic reward weights, final ``rewards``, and ``advantages``.
    That boundary matters: joint weighting is a function of advantage-space
    group statistics, not a static reward-space aggregation.

    Key difference from UnifiedRewardModelWorker in single mode:
    - Single mode: Uses DP_COMPUTE_PROTO, data pre-split by framework
    - Joint mode: Uses ALL_TO_ALL, data not pre-split, models handle splitting

    Note: This class inherits from Worker (not RewardModelWorker) because
    it manages Registry-based models that have their own initialization logic.
    """

    def __init__(self, config, cuda_visible_devices=None):
        """Initialize the JointRewardModelWorker.

        Args:
            config: Reward model configuration (OmegaConf DictConfig)
            cuda_visible_devices: CUDA visible devices configuration
        """
        Worker.__init__(self, cuda_visible_devices=cuda_visible_devices)
        WorkerProfilerExtension.__init__(self, WorkerProfiler(rank=self.rank, config=ProfilerConfig(**OmegaConf.to_object(config.get("profiler", DictConfig({}))))))
        self.config = config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize joint reward models from configuration."""
        # Initialize torch.distributed if not already done
        if not dist.is_initialized():
            from verl.utils.device import get_nccl_backend

            dist.init_process_group(backend=get_nccl_backend())

        self.reward_models: dict[str, Any] = {}
        self.model_configs: dict[str, RewardConfig] = {}

        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Initialize models from joint configuration
        joint_cfg = self.config.get("joint", {})
        models_cfg = joint_cfg.get("models", {})

        if not models_cfg:
            raise ValueError("JointRewardModelWorker requires 'joint.models' configuration")

        if isinstance(models_cfg, list | tuple):
            raise TypeError("joint.models must be a mapping keyed by reward model name, not a list")
        model_items = list(OmegaConf.to_container(models_cfg, resolve=True).items())

        for model_name, model_cfg in model_items:
            if not model_cfg.get("enabled", True):
                continue

            # Build RewardConfig for this model
            extra_config = model_cfg.get("extra_config", {})
            if hasattr(extra_config, "items"):
                extra_config = dict(extra_config)

            # Each model can have its own DP configuration:
            # - dp_fraction: fraction of workers that will run this model
            # - rank_offset: starting rank for the model's active workers
            # - mps_percentage: GPU resource allocation via MPS
            config = RewardConfig(
                name=model_name,
                model_path=model_cfg.get("model_path", "") or model_cfg.get("model", {}).get("path", ""),
                extra_config=extra_config,
                weight=model_cfg.get("weight", 1.0),
                normalize=model_cfg.get("normalize", True),
                dp_fraction=model_cfg.get("dp_fraction", 1.0),
                rank_offset=model_cfg.get("rank_offset", 0),
                mps_percentage=model_cfg.get("mps_percentage", 0),
            )

            self._create_and_init_model(model_name, config)

        logger.info(f"[JointRM] Initialized {len(self.reward_models)} models: {list(self.reward_models.keys())}")

    def _create_and_init_model(self, model_name: str, config: RewardConfig):
        """Create and initialize a reward model from the registry."""
        try:
            model = create_reward_model(
                name=model_name,
                config=config,
                rank=self._global_rank,
                world_size=self._world_size,
            )
            model.init_model()

            self.reward_models[model_name] = model
            self.model_configs[model_name] = config

            logger.info(f"[JointRM] Initialized '{model_name}'")

        except Exception as e:
            import traceback

            logger.error(f"[JointRM] Failed to initialize '{model_name}': {e}")
            logger.error(f"[JointRM] Traceback:\n{traceback.format_exc()}")
            raise

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    @WorkerProfiler.annotate(color="green")
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute raw reward-head scores using all joint models with per-model DP.

        Architecture:
        - ALL_TO_ALL dispatch: each worker receives the FULL batch
        - Every model uses all-active world-size DP
        - Workers take contiguous, potentially uneven local slices
        - Length-aware AllGather reconstructs results in input order
        - Return one ``<name>_rewards`` tensor per model

        This enables:
        - Multiple reward heads sharing each GPU via MPS
        - Correct batch_size output matching input
        """
        start_time = time.time()
        data = data.to(get_device_id())

        full_batch_size = data.batch.batch_size[0]
        all_rewards = {}

        for model_name, model in self.reward_models.items():
            local_rewards = None
            local_error = None
            try:
                # Each model uses its own DP configuration
                # compute_batch_score_for_joint returns LOCAL results (for this worker's portion)
                result = model.compute_batch_score_for_joint(data)
                reward_key = model.REWARD_KEY
                local_rewards = result.batch[reward_key]
                if not isinstance(local_rewards, torch.Tensor) or local_rewards.ndim == 0:
                    raise TypeError(f"{model_name} must return a reward tensor with a batch dimension, got {type(local_rewards).__name__} shape={getattr(local_rewards, 'shape', None)}")

                world_size = dist.get_world_size() if dist.is_initialized() else 1
                rank = dist.get_rank() if dist.is_initialized() else 0
                base_size, remainder = divmod(full_batch_size, world_size)
                expected_local_size = base_size + int(rank < remainder)
                if local_rewards.shape[0] != expected_local_size:
                    raise ValueError(f"{model_name} returned {local_rewards.shape[0]} local rewards on rank {rank}; expected {expected_local_size} for full batch size {full_batch_size} and world size {world_size}")

                # Device conversion can itself fail (for example after a CUDA
                # OOM). Do it inside the rank-local try block, then synchronize
                # the status before any reward-data collective.
                if dist.is_initialized() and world_size > 1:
                    local_rewards = local_rewards.to(self._collective_device(local_rewards))
            except Exception as e:
                import traceback

                logger.error(f"[JointRM] Error computing {model_name}: {e}")
                logger.error(f"[JointRM] Traceback:\n{traceback.format_exc()}")

                local_error = e

            # Every rank synchronizes the model-compute status before any rank
            # enters the variable-length reward gather.  Previously a failed
            # rank continued to the next head while its peers blocked forever
            # in all_gather.  A shared fail-fast decision keeps the collective
            # sequence identical on every rank and never silently drops a
            # reward head.
            self._raise_if_any_rank_failed(model_name, local_error, local_rewards)

            assert local_rewards is not None  # narrowed by the synchronized guard above

            # AllGather to collect rewards from all workers for this model.
            # `base.py` fail-loud guards force dp_size == world_size (the only
            # dispatch the codebase supports), so rank-order concatenation of
            # the contiguous DP slices reconstructs input order.
            if dist.is_initialized() and dist.get_world_size() > 1:
                gathered_rewards = self._allgather_rewards(
                    local_rewards,
                    expected_size=full_batch_size,
                )
            else:
                gathered_rewards = local_rewards

            if gathered_rewards.shape[0] != full_batch_size:
                raise RuntimeError(f"Joint reward model '{model_name}' reconstructed {gathered_rewards.shape[0]} rewards for a batch of {full_batch_size}")

            gathered_rewards = self._normalize_gathered_rewards(
                gathered_rewards,
                enabled=self.model_configs[model_name].normalize,
            )

            # The collective has completed, so restore the historical worker
            # return contract (host reward tensors) and avoid handing Ray a
            # redundant full-batch CUDA allocation from every reward rank.
            all_rewards[model_name] = gathered_rewards.cpu()

        if not all_rewards:
            raise RuntimeError("No rewards computed from any model")

        batch = TensorDict({}, batch_size=full_batch_size)
        for model_name, rewards in all_rewards.items():
            batch[f"{model_name}_rewards"] = rewards

        elapsed = time.time() - start_time
        logger.info(f"[JointRM] Compute time: {elapsed:.2f}s, batch_size={full_batch_size}")

        return DataProto(batch=batch, non_tensor_batch=data.non_tensor_batch)

    @staticmethod
    def _normalize_gathered_rewards(
        rewards: torch.Tensor,
        *,
        enabled: bool,
    ) -> torch.Tensor:
        """Apply a head's normalization exactly once, after full-batch gather."""

        return normalize_gathered_rewards(rewards, enabled=enabled)

    @staticmethod
    def _collective_device(reference: torch.Tensor | None = None):
        """Return a device accepted by the active process-group backend."""

        backend = str(dist.get_backend()).lower()
        if "nccl" in backend:
            if reference is not None and reference.is_cuda:
                return reference.device
            return get_device_id()
        # Gloo tests and CPU deployments should not inherit a CUDA reference.
        return torch.device("cpu")

    def _raise_if_any_rank_failed(
        self,
        model_name: str,
        local_error: Exception | None,
        local_rewards: torch.Tensor | None,
    ) -> None:
        """Synchronize rank-local scoring errors before reward collectives.

        All ranks raise the same error once any rank reports a failure.  The
        original exception remains chained on the rank where it occurred,
        while peers receive the shared summary instead of entering a gather
        that the failed rank will never join.
        """

        collective_device = self._collective_device(local_rewards) if dist.is_initialized() and dist.get_world_size() > 1 else torch.device("cpu")
        failed_count = synchronized_failure_count(
            local_failed=local_error is not None,
            device=collective_device,
        )

        if failed_count:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            message = f"Joint reward model '{model_name}' failed on {failed_count}/{world_size} rank(s); aborting before reward all_gather"
            raise RuntimeError(message) from local_error

    def _allgather_rewards(
        self,
        local_rewards: torch.Tensor,
        *,
        expected_size: int | None = None,
    ) -> torch.Tensor:
        """AllGather a model's local rewards across the world process group.

        The first dimension may differ by one between ranks when the full
        batch is not divisible by world size.  Gather lengths first, pad each
        rank to the maximum length, then trim before rank-order concatenation.
        Since ``split_batch_for_dp`` emits contiguous rank-ordered slices, the
        concatenation reconstructs the original sample order.

        Caller is responsible for verifying ``dist.is_initialized()`` and
        ``world_size > 1``.

        The previous version of this helper carried ``dp_size`` /
        ``rank_offset`` / ``is_active`` parameters intended for disjoint
        per-reward DP groups, but ``dist.all_gather`` here always runs on the
        default world group — so ``dp_size != world_size`` would silently
        mismatch the gathered list length.  The yaml default + ``base.py``
        fail-loud guards now keep ``dp_size == world_size`` invariant; the
        helper reflects that.
        """
        return allgather_variable_batch(
            local_rewards,
            collective_device=self._collective_device(local_rewards),
            expected_size=expected_size,
        )
