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
"""Trainer-side joint reward logic.

Active when ``reward.reward_model.type=joint``. All per-task models live in a
single ``JointRewardModelWorker`` worker group (registered as
``Role.RewardModel`` in ``main_teleboost.py``); the mixin dispatches one
``compute_rm_score`` call and receives raw ``<name>_rewards`` heads.

The weighted joint signal is computed here, not in the worker: each reward
head first becomes a per-prompt GRPO advantage, then
:func:`compute_joint_advantage_weights` derives the convex reward weights from
the advantage matrix. This needs rollout grouping and full-batch trainer
state, so it is a driver concern.

Trainer hooks live on :class:`JointRewardMixin`.
"""

from __future__ import annotations

import logging
import time

import torch
from verl import DataProto

from teleboost.algorithms.grpo.advantage import per_prompt_zscore_advantage
from teleboost.training.rewarding.joint_advantage_weights import compute_joint_advantage_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer mixin
# ---------------------------------------------------------------------------


class JointRewardMixin:
    """Trainer mixin for the ``reward.reward_model.type=joint`` path.

    One execution mode: the unified ``JointRewardModelWorker`` worker group
    (``self.rm_wg``) computes every raw reward head in a single dispatch; the
    mixin converts those heads into per-reward advantages, dynamic
    ``reward_weights``, final ``rewards``, and final ``advantages``.

    Composition with BGPO is rejected at startup
    (BGPO's ``_is_bgpo_enabled`` — ``recipes/wan_bgpo_fsdp/trainer.py`` — raises on ``bgpo.enable=true`` +
    ``reward.reward_model.type=joint``): the BGPO paper does not define how its
    single-scalar rerange composes with advantage-space multi-reward weights.
    """

    # -- worker-side parallel path -----------------------------------------

    def _compute_joint_parallel_reward(
        self,
        gen_batch_output: DataProto,
        metrics: dict,
    ) -> DataProto:
        """Raw joint reward heads via ``JointRewardModelWorker``.

        The worker (registered as ``Role.RewardModel`` in ``main_teleboost.py``)
        owns all reward-head models and emits ``<name>_rewards`` keys in a
        single ``compute_rm_score`` call. It deliberately does not aggregate
        them into ``rewards``; joint weighting happens in advantage space below.
        """
        start_time = time.time()

        reward_input = gen_batch_output.select(
            batch_keys=["video_frames"],
            non_tensor_batch_keys=["caption"],
        )

        # ``JointRewardModelWorker.compute_rm_score`` is registered with
        # ``Dispatch.ALL_TO_ALL``: every worker receives the full batch
        # and returns its own DataProto.  The worker performs an
        # AllGather internally so each rank's returned proto is already
        # the merged view, and the dispatcher hands us a list with one
        # equivalent entry per rank.  Concat-merging would inflate the
        # batch dim, so we just take the first.
        raw = self.rm_wg.compute_rm_score(reward_input)
        if isinstance(raw, list):
            if not raw:
                raise RuntimeError("rm_wg.compute_rm_score returned an empty list")
            result = raw[0]
        else:
            result = raw

        batch_with_rewards = gen_batch_output
        combined_reward = None

        joint_cfg = self.config.reward.reward_model.get("joint", {})
        models_cfg = joint_cfg.get("models", {})
        if isinstance(models_cfg, list | tuple):
            weights = {m.get("name"): m.get("weight", 1.0) for m in models_cfg if m.get("name")}
        else:
            weights = {k: v.get("weight", 1.0) for k, v in models_cfg.items()}

        # Pull per-reward ``<name>_rewards`` keys back into the batch and build
        # a static weighted reward view for logs / diagnostics. The
        # optimization path below overwrites ``rewards`` with the
        # advantage-space dynamic combination.
        for key in result.batch.keys():
            if not key.endswith("_rewards"):
                continue
            name = key[: -len("_rewards")]
            if name == "":  # the bare aggregated ``rewards`` key
                continue

            rewards = result.batch[key]
            batch_with_rewards.batch[key] = rewards

            weight = weights.get(name, 1.0)
            weighted_reward = rewards * weight
            if combined_reward is None:
                combined_reward = weighted_reward
            else:
                combined_reward = combined_reward + weighted_reward

            metrics[f"train/rewards_{name}"] = rewards.mean().item()

        if combined_reward is None:
            raise RuntimeError("No valid rewards computed from any model")

        batch_with_rewards.batch["rewards"] = combined_reward

        metrics["train/rewards"] = combined_reward.mean().item()
        if "log_probs" in batch_with_rewards.batch.keys():
            metrics["train/log_probs"] = batch_with_rewards.batch["log_probs"].mean().item()

        elapsed = time.time() - start_time
        logger.info(f"Joint parallel reward computation took {elapsed:.2f}s")

        return batch_with_rewards

    # -- joint + BGPO precompute -------------------------------------------

    def _precompute_joint_advantages(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
    ) -> tuple[DataProto, bool]:
        """Pre-compute joint-mode advantages + per-reward convex weights.

        Runs BEFORE the standard reward+advantage block so the caller
        can skip the duplicated computation.  When BGPO is also enabled
        the rewards are reranged here and the advantage is re-derived
        from the post-rerange rewards.

        Returns
        -------
        (DataProto, bool)
            Updated batch, and a flag indicating whether the multi-head
            precompute fired (False -> caller must run the standard path).
        """
        reward_output = self._compute_joint_parallel_reward(gen_batch_output, metrics)

        reward_keys = [k for k in reward_output.batch.keys() if k.endswith("_rewards")]
        if not reward_keys:
            return reward_output, False

        per_task_rewards = [reward_output.batch[k].float() for k in reward_keys]
        per_task_advantages = []

        # Per-prompt grouping for the per-task z-score.  Same paper
        # reasoning as the single-reward path in
        # ``ray_trainer.compute_advantage`` (GRPO arxiv
        # 2402.03300 §4.1.2 + DanceGRPO Eq. 10).
        num_repeat = int(self.config.actor_rollout_ref.rollout.n)

        for reward_key in reward_keys:
            reward_tensor = reward_output.batch[reward_key].float()
            adv = per_prompt_zscore_advantage(reward_tensor, num_repeat)
            task_name = reward_key[:-8]
            reward_output.batch[f"{task_name}_advantages"] = adv
            metrics[f"train/{task_name}_advantages"] = adv.mean().item()
            per_task_advantages.append(adv)

        adv_matrix = torch.stack(per_task_advantages, dim=-1)
        weight_matrix = compute_joint_advantage_weights(adv_matrix)

        reward_output.batch["reward_weights"] = weight_matrix
        reward_output.batch["rewards"] = (weight_matrix * torch.stack(per_task_rewards, dim=-1)).sum(dim=-1)
        reward_output.batch["advantages"] = (weight_matrix * adv_matrix).sum(dim=-1)

        # BGPO×joint is blocked upstream (recipes/wan_bgpo_fsdp/trainer.py raises when both
        # are enabled), so there is no post-rerange advantage rewrite here. If that
        # guard is ever removed, this branch must also rebuild
        # ``reward_output.batch["reward_weights"]`` from the post-rerange per-task
        # advantages, else ``train/reward_weight_*`` logs the stale pre-rerange matrix.

        for idx, reward_key in enumerate(reward_keys):
            task_name = reward_key[:-8]
            metrics[f"train/reward_weight_{task_name}"] = weight_matrix[:, idx].mean().item()

        metrics["train/rewards"] = reward_output.batch["rewards"].mean().item()
        return reward_output, True


__all__ = [
    "JointRewardMixin",
]
