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
"""BGPO trainer: Wan generation policy + BGPO's driver-phase hooks.

Selected by ``teleboost.programs.wan.backend.WanBackendSpec.trainer_cls`` when ``algorithm.bgpo.enable``
is true. The pure math lives in ``teleboost.algorithms.bgpo`` (paper
arXiv 2511.18919); the trainer extension below wires it into the base
trainer's seams.
"""

from typing import Any, Optional

import numpy as np
import torch
from verl import DataProto

from teleboost.training.families.wan import RayWanTrainer
from teleboost.algorithms.bgpo import bayes_reliability_weight, binary_rerange_group_rewards


class BGPOMixin:
    """BGPO trainer extension. Composed by ``teleboost/programs/wan/bgpo.py``
    (``RayBGPOTrainer``) on top of the algorithm-agnostic base trainer.

    Attributes referenced from the trainer:
        - ``self.config`` (Hydra DictConfig)
    """

    # -- config accessors ---------------------------------------------------

    # ---- Base-trainer seam adapters ------------------------------------
    # The base trainer calls these generic seams; cooperative super() keeps
    # explicit combination trainers chaining in MRO order.

    def _transform_rewards(self, reward_output, source_batch, metrics):
        if self._is_bgpo_enabled() and source_batch is not None:
            reward_output = self._apply_bgpo_on_rewards(reward_output, source_batch, metrics)
        return super()._transform_rewards(reward_output, source_batch, metrics)

    def _transform_advantages(self, gen_batch_output, gen_batch, metrics):
        # Scale advantages BEFORE any dense broadcast further down the MRO.
        if self._is_bgpo_enabled():
            gen_batch_output = self._apply_bgpo_on_advantages(gen_batch_output, gen_batch, metrics)
        return super()._transform_advantages(gen_batch_output, gen_batch, metrics)

    def _get_bgpo_config(self) -> dict[str, Any]:
        algorithm_cfg = self.config.get("algorithm", {})
        return algorithm_cfg.get("bgpo", {}) or {}

    def _is_bgpo_enabled(self) -> bool:
        enabled = bool(self._get_bgpo_config().get("enable", False))
        if enabled:
            # BGPO optimizes a single scalar reward; the joint path runs a
            # separate in-house multi-reward aggregation. Refuse the mix.
            reward_type = self.config.get("reward", {}).get("reward_model", {}).get("type", "single")
            if reward_type == "joint":
                raise ValueError("BGPO does not support reward.reward_model.type=joint: BGPO optimizes a single scalar reward. Set reward.reward_model.type=single, or algorithm.bgpo.enable=false to use joint without BGPO.")

        return enabled

    def _get_prior_array(self, source_batch: DataProto) -> Optional[np.ndarray]:
        """Return the raw numeric prior payload from the dataset batch."""
        if source_batch is None or "prior" not in source_batch.non_tensor_batch:
            return None
        return np.asarray(source_batch.non_tensor_batch["prior"]).reshape(-1).astype(np.float32)

    @staticmethod
    def _normalize_group_priors(
        prior_arr: np.ndarray,
        *,
        num_groups: int,
        rollout_n: int,
    ) -> np.ndarray:
        """Accept one prior per prompt or a consistently repeated prior.

        The trainer can receive either the source prompt batch (``G`` priors)
        or its rollout-expanded form (``G * n`` priors).  Do not infer which
        representation was supplied merely from divisibility: when ``G == n``
        that would silently discard valid prompt priors.
        """
        if prior_arr.size == num_groups:
            return prior_arr
        if prior_arr.size == num_groups * rollout_n:
            expanded = prior_arr.reshape(num_groups, rollout_n)
            if not np.all(expanded == expanded[:, :1]):
                raise ValueError("BGPO rollout-expanded priors must be identical within each prompt group")
            return expanded[:, 0]
        raise ValueError(f"BGPO prior/reward group mismatch: priors={prior_arr.size}, reward_groups={num_groups}, rollout.n={rollout_n}")

    # -- adaptive weight (RAS) ---------------------------------------------

    def _calculate_adaptive_weight(self, group_rewards: torch.Tensor, prior: float) -> float:
        """RAS per-group centered weight ``w − 1``; caller forms ``w = 1 + α·(w − 1)``.

        Modes: ``"bayes"`` (default, Bayesian posterior reliability gain) and
        ``"no"`` (returns ``0`` → CRT-only ablation).
        """
        bgpo_cfg = self._get_bgpo_config()
        method = bgpo_cfg.get("adaptive_weight_method", "bayes")

        if method == "no":
            return 0.0

        if method == "bayes":
            weight_range = bgpo_cfg.get("bayes_weight_range", [0.5, 1.5])
            if len(weight_range) != 2:
                weight_range = [0.5, 1.5]
            return bayes_reliability_weight(
                group_rewards,
                prior=prior,
                prior_var=float(bgpo_cfg.get("prior_var", 1.0)),
                weight_range=(float(weight_range[0]), float(weight_range[1])),
            )

        raise ValueError(f"Unsupported adaptive_weight_method={method!r}; valid modes: 'bayes', 'no'.")

    # -- reward / advantage application ------------------------------------

    def _apply_bgpo_on_rewards(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        """CRT branch: rearrange per-sample rewards (binary prior-threshold)."""
        if "rewards" not in gen_batch_output.batch:
            raise KeyError("BGPO is enabled but batch['rewards'] is missing")

        prior_arr = self._get_prior_array(source_batch)
        if prior_arr is None:
            raise ValueError("BGPO is enabled but the dataset batch has no numeric 'prior'; refusing to fall back to ordinary GRPO.")

        bgpo_cfg = self._get_bgpo_config()
        rewards = gen_batch_output.batch["rewards"]
        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        if rewards.shape[0] % rollout_n:
            raise ValueError(f"BGPO rewards batch {rewards.shape[0]} is not divisible by rollout.n={rollout_n}")
        num_groups = int(rewards.shape[0] // rollout_n)
        if num_groups <= 0:
            raise ValueError("BGPO requires at least one reward group")
        prior_arr = self._normalize_group_priors(
            prior_arr,
            num_groups=num_groups,
            rollout_n=rollout_n,
        )

        group_rewards_len = num_groups * rollout_n
        rewards_for_groups = rewards[:group_rewards_len]

        use_rerange = bool(bgpo_cfg.get("use_rerange", False))
        rerange_a = float(bgpo_cfg.get("rerange_a", 50.0))
        rerange_temperature = float(bgpo_cfg.get("rerange_temperature", 5.0))
        exp_clamp = float(bgpo_cfg.get("exp_clamp", 30.0))
        reranged_rewards = rewards_for_groups.clone()

        if use_rerange:
            for i in range(num_groups):
                start = i * rollout_n
                end = start + rollout_n
                reranged_rewards[start:end] = binary_rerange_group_rewards(
                    rewards_for_groups[start:end],
                    prior=float(prior_arr[i]),
                    a=rerange_a,
                    temperature=rerange_temperature,
                    exp_clamp=exp_clamp,
                )
        else:
            return gen_batch_output

        # Single-branch: overwrite the group rewards with R̃ in place.
        gen_batch_output.batch["rewards"][:group_rewards_len] = reranged_rewards
        metrics["train/rewards_reranged"] = reranged_rewards.mean().item()
        return gen_batch_output

    def _apply_bgpo_on_advantages(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        """RAS branch: scale scalar advantages by an adaptive group weight."""
        if "advantages" not in gen_batch_output.batch or "rewards" not in gen_batch_output.batch:
            raise KeyError("BGPO is enabled but batch must contain both 'advantages' and 'rewards'")

        prior_arr = self._get_prior_array(source_batch)
        if prior_arr is None:
            raise ValueError("BGPO is enabled but the dataset batch has no numeric 'prior'; refusing to fall back to ordinary GRPO.")

        bgpo_cfg = self._get_bgpo_config()
        alpha = float(bgpo_cfg.get("regularization_term_alpha", 1.0))
        max_scale = float(bgpo_cfg.get("max_adv_scale", 10.0))
        min_scale = float(bgpo_cfg.get("min_adv_scale", 0.01))

        # Reference behavior: when CRT ran, these rewards are already R̃
        # (overwritten in _apply_bgpo_on_rewards); RAS weights off them.
        rewards = gen_batch_output.batch["rewards"]
        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        if rewards.shape[0] % rollout_n:
            raise ValueError(f"BGPO rewards batch {rewards.shape[0]} is not divisible by rollout.n={rollout_n}")
        num_groups = int(rewards.shape[0] // rollout_n)
        if num_groups <= 0:
            raise ValueError("BGPO requires at least one reward group")
        prior_arr = self._normalize_group_priors(
            prior_arr,
            num_groups=num_groups,
            rollout_n=rollout_n,
        )

        group_rewards_len = num_groups * rollout_n
        per_sample_weight = torch.zeros_like(rewards, dtype=torch.float32)

        for i in range(num_groups):
            start = i * rollout_n
            end = start + rollout_n
            per_sample_weight[start:end] = self._calculate_adaptive_weight(
                rewards[start:end].float(),
                float(prior_arr[i]),
            )

        gen_batch_output.batch["bgpo_weight"] = per_sample_weight

        scale = torch.clamp(1.0 + alpha * per_sample_weight, min=min_scale, max=max_scale)
        advantages = gen_batch_output.batch["advantages"]
        if advantages.shape[0] != rewards.shape[0]:
            raise ValueError(f"BGPO advantage/reward batch mismatch: advantages={advantages.shape[0]}, rewards={rewards.shape[0]}")
        # Dense advantages (e.g. VIPO already broadcast) need scale reshaped
        # so we only multiply along the batch axis.
        if advantages.ndim > 1:
            scale_view = scale.view(scale.shape[0], *([1] * (advantages.ndim - 1)))
        else:
            scale_view = scale
        gen_batch_output.batch["advantages"] = advantages * scale_view

        metrics["train/bgpo_weight"] = per_sample_weight[:group_rewards_len].mean().item()
        metrics["train/bgpo_adv_scale"] = scale.mean().item()
        return gen_batch_output


class RayBGPOTrainer(BGPOMixin, RayWanTrainer):
    """Wan trainer + BGPO reward rerange (CRT) and adaptive advantage scaling (RAS)."""
