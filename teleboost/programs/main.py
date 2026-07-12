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
TeleBoost GRPO entrypoint.

Applies Group Relative Policy Optimization to video diffusion models
with multiple reward signals.

Backend-specific construction (tokenizer, workers, rewards, collation, and
trainer selection) lives behind the dependency-light
``teleboost.programs.backend_api.BackendSpec`` protocol. ``run()`` resolves exactly one
built-in or external implementation through the backend registry.
"""

import logging
import os
from pprint import pprint
from typing import TYPE_CHECKING

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

# Apply TeleBoost patches over upstream verl (cp grad fix, etc.) BEFORE any
# verl import below: subsequent `from verl.X import Y` then resolves to the
# patched symbols.
from teleboost import apply_runtime_patches

if TYPE_CHECKING:
    from teleboost.programs.backend_api import BackendSpec

apply_runtime_patches()

logger = logging.getLogger(__name__)

# Ray environment variables for distributed training
RAY_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # Keep Wan attention selection identical in fresh Ray actor processes.
    "TELEBOOST_WAN_ATTN_BACKEND": os.environ.get("TELEBOOST_WAN_ATTN_BACKEND", "auto"),
    # vLLM 0.14 spawns model workers inside Ray. Keep the scoped
    # sitecustomize tokenizer fix active in those fresh interpreters.
    "TELEBOOST_VLLM_TOKENIZER_REGEX_FIX": os.environ.get("TELEBOOST_VLLM_TOKENIZER_REGEX_FIX", "1"),
}
if "PYTHONPATH" in os.environ:
    # Includes the source checkout and the scoped vLLM sitecustomize directory
    # added by the launchers. Ray applies this before starting worker Python,
    # so multiprocessing-spawned vLLM children inherit it as well.
    RAY_ENV_VARS["PYTHONPATH"] = os.environ["PYTHONPATH"]

for _thread_env_var in (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    if _thread_env_var in os.environ:
        RAY_ENV_VARS[_thread_env_var] = os.environ[_thread_env_var]


def _init_ray(config: DictConfig) -> None:
    """Initialize Ray runtime if not already initialized.

    Upstream verl ≥0.6 places ``num_cpus`` at
    ``config.ray_kwargs.ray_init.num_cpus``. Older verl 0.4-style yamls
    may still place it at ``config.ray_init.num_cpus``.
    """
    if ray.is_initialized():
        logger.info("Ray already initialized, skipping")
        return

    # OmegaConf.select() safely returns None when any segment of the path
    # is missing — needed because the resolved config is struct=True and
    # attribute access raises on missing keys.
    num_cpus = OmegaConf.select(config, "ray_kwargs.ray_init.num_cpus")
    if num_cpus is None:
        # verl 0.4 yaml layout.
        num_cpus = OmegaConf.select(config, "ray_init.num_cpus")
    runtime_env_vars = dict(RAY_ENV_VARS)
    # Align with upstream verl v0.7.1 ``main_ppo``: propagate
    # ``TRANSFER_QUEUE_ENABLE=1`` to every Ray actor process so the
    # ``transfer_queue`` package picks up the toggle uniformly.
    if bool(OmegaConf.select(config, "transfer_queue.enable", default=False)):
        runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
        partition_id = OmegaConf.select(
            config,
            "transfer_queue.partition_id",
            default="teleboost.rollout.video_frames",
        )
        runtime_env_vars["TELEBOOST_TQ_PARTITION"] = str(partition_id)
    ray.init(
        runtime_env={"env_vars": runtime_env_vars},
        num_cpus=num_cpus,
    )
    logger.info(f"Ray initialized with num_cpus={num_cpus}")


def _validate_config(config: DictConfig, backend: "BackendSpec") -> None:
    """
    Validate Dance-GRPO specific configuration.

    The generic reward-strategy check is backend-agnostic; the per-type /
    per-adapter rules for the ``diffusion`` strategy are backend-specific
    and delegated to ``backend.validate_reward``.

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate the model-family strategy and algorithm matrix before any
    # tokenizer download, Ray actor allocation, or GPU model construction.
    backend.validate_capabilities(config)

    # ``reward.reward_model.enable`` only controls verl's vLLM
    # RewardModelManager after schema normalization. Registry-backed single
    # and joint rewards deliberately have it set to false and are controlled
    # by ``trainer.use_rm``. Gating validation on ``enable`` therefore let bad
    # TeleBoost reward configurations fail only after workers were launched.
    if bool(config.trainer.get("use_rm", True)):
        strategy = config.reward.reward_model.strategy
        rm_type = str(config.reward.reward_model.get("type", "")).strip().lower()
        adapter = str(config.reward.reward_model.get("adapter", "") or "").strip().lower()
        valid_strategies = ["fsdp", "megatron", "diffusion"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid reward.reward_model.strategy: {strategy}. Must be one of {valid_strategies}")

        if rm_type == "joint" and adapter:
            raise ValueError("reward.reward_model.type=joint cannot be combined with a VLM adapter; joint VLM aggregation is not implemented")
        if rm_type == "joint" and strategy != "diffusion":
            raise ValueError("reward.reward_model.type=joint currently requires reward.reward_model.strategy=diffusion")

        if strategy == "diffusion":
            backend.validate_reward(config)

    logger.info("Configuration validation passed")


def run_ppo(config: DictConfig) -> None:
    """
    Initialize Ray and start the PPO training task.

    Args:
        config: Hydra configuration
    """
    # Normalize reward config BEFORE any consumer reads it. Runtime code uses
    # ``reward.reward_model.*``; the normalizer also aligns upstream
    # RewardModelManager enable semantics with TeleBoost worker-group rewards.
    from teleboost.reward.config_schema import normalize_reward_config

    from teleboost.programs.selection import select_backend

    normalize_reward_config(config)

    # Perform the same validation in the driver before ray.init(). TaskRunner
    # repeats it in its own interpreter so worker-side config drift still fails
    # loudly, but invalid configurations must not start a Ray cluster first.
    _validate_config(config, select_backend(config))

    _init_ray(config)

    # Phase 2: bootstrap TransferQueue after Ray is up so future rollout /
    # reward / actor workers can shuttle bulk tensors peer-to-peer instead
    # of through the driver. Gated by ``transfer_queue.enable`` (default
    # false). When disabled — or the ``transfer_queue`` package isn't
    # importable — the bootstrap is a no-op and workers use the
    # DataProto path.
    # TransferQueue: align with upstream verl v0.7.1 — top-level
    # ``transfer_queue.enable`` flag, activation via
    # ``TRANSFER_QUEUE_ENABLE=1`` in Ray's runtime_env so every worker
    # process sees it. Upstream stops at the toggle; teleboost goes
    # further and writes ``video_frames`` into TQ via ``kv_put`` so the
    # trainer driver doesn't ray.get the full video tensor.
    if bool(OmegaConf.select(config, "transfer_queue.enable", default=False)):
        # Mirror env var into the driver process (we run kv_batch_get
        # from the trainer too — driver is itself a TQ consumer).
        os.environ["TRANSFER_QUEUE_ENABLE"] = "1"
        partition_id = OmegaConf.select(
            config,
            "transfer_queue.partition_id",
            default="teleboost.rollout.video_frames",
        )
        os.environ["TELEBOOST_TQ_PARTITION"] = str(partition_id)

        from teleboost.engines.transport.transfer_queue import bootstrap as _tq_bootstrap

        _tq_bootstrap()

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    """
    Remote task runner for Dance-GRPO training.

    This class runs on a Ray worker (not the head node) to avoid
    resource contention with the driver process.
    """

    def run(self, config: DictConfig) -> None:
        """
        Execute the training pipeline.

        Args:
            config: Training configuration
        """
        # This class is defined in ``__main__`` (python -m), so cloudpickle
        # ships it BY VALUE: the remote interpreter never imports this module
        # and the module-level apply_runtime_patches() above never ran here.
        apply_runtime_patches()
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Print and resolve configuration
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Construct the (single) training backend; every backend-specific
        # construction step below goes through it. Imported lazily so
        # importing this module stays light.
        from teleboost.programs.selection import select_backend

        backend = select_backend(config)
        logger.info(f"Training backend: {backend.name}")

        # Validate configuration
        _validate_config(config, backend)

        # Build tokenizer/processor (backend downloads its model if needed)
        tokenizer, processor = backend.prepare_tokenizer(config)

        # Resolve worker classes
        ray_worker_group_cls, actor_rollout_worker_cls = backend.resolve_worker_and_group(config)

        # Setup role-worker mapping
        role_worker_mapping = {}
        global_pool_id = "global_pool"

        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {}

        def register_role(role, worker_cls):
            role_worker_mapping[role] = ray.remote(worker_cls)
            mapping[role] = global_pool_id

        # Register actor/rollout worker
        register_role(Role.ActorRollout, actor_rollout_worker_cls)

        # Register reward workers
        backend.register_reward_workers(config, role_worker_mapping, mapping, global_pool_id)

        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Get collate function
        collate_fn = backend.collate_fn(config)

        # Create and run trainer (the backend selects the per-algorithm
        # trainer subclass from the enable flags; loud on conflicts)
        trainer_cls = backend.trainer_cls(config)
        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            collate_fn=collate_fn,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        # init_workers() also launches the colocated vLLMReplica for VLM judge
        # paths. Matches upstream RewardModelManager._initialize_llm_servers
        # lifecycle (call sits inside init_workers, after actor init_model).
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../config", config_name="teleboost_trainer", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main entry point for Dance-GRPO training.

    Uses Hydra for configuration management.

    Args:
        config: Hydra-loaded configuration
    """
    run_ppo(config)


if __name__ == "__main__":
    main()
