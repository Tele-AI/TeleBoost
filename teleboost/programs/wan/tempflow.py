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
"""TempFlow trainer: Wan generation policy + trajectory branching.

Selected by ``teleboost.programs.wan.backend.WanBackendSpec.trainer_cls`` when
``actor_rollout_ref.actor.tempflow.branch.enable`` is true. The pure math
lives in ``teleboost.algorithms.tempflow`` (paper arXiv
2508.04324); the trainer extension below wires it into the base trainer's
seams. TempFlow's other lever (noise-weight reweighting) is actor-side
and needs no trainer.
"""

from teleboost.training.families.wan import RayWanTrainer
from teleboost.algorithms.tempflow.trajectory import compute_branched_advantage


class TrajectoryBranchMixin:
    """Trainer hook for TempFlow trajectory-branching advantage.

    Composed by ``RayTempFlowTrainer`` below, giving the base trainer the
    branched-advantage path without the dispatch leaking into ``fit()``.
    Shares ``self.config``. When ``branch.enable`` is False the hooks are
    never taken and training is byte-identical to baseline GRPO.
    """

    # ---- Base-trainer seam adapters ------------------------------------

    def _pre_rollout_transform(self, gen_batch):
        # Branch grouping needs a GLOBALLY-unique prompt id stamped BEFORE
        # the sharded rollout dispatch: each worker numbers prompts from 0
        # locally, so the concatenated branch output would otherwise collide.
        if self._is_branch_enabled():
            self._stamp_branch_global_ids(gen_batch)
        return super()._pre_rollout_transform(gen_batch)

    def _compute_algorithm_advantage(self, gen_batch_output):
        # Replaces the default GRPO advantage with the per-branch-point
        # advantage (Eq.3 / Thm.1) when branching is on; None keeps default.
        if self._is_branch_enabled():
            return self._apply_branched_advantage(gen_batch_output)
        return super()._compute_algorithm_advantage(gen_batch_output)

    def _is_branch_enabled(self) -> bool:
        from omegaconf import OmegaConf

        return bool(
            OmegaConf.select(
                self.config,
                "actor_rollout_ref.actor.tempflow.branch.enable",
                default=False,
            )
        )

    def _apply_branched_advantage(self, data):
        """Swap in the per-(prompt,k) branched advantage (Eq. 3).

        Honors ``algorithm.min_group_std`` — the same freeze floor the baseline
        GRPO advantage uses — so a low-variance branch group can't be re-inflated
        into random-sign advantages.
        """
        from omegaconf import OmegaConf

        min_group_std = float(OmegaConf.select(self.config, "algorithm.min_group_std", default=0.0))
        return compute_branched_advantage(data, min_group_std=min_group_std)

    def _stamp_branch_global_ids(self, gen_batch):
        """Stamp a globally-unique prompt id BEFORE the sharded rollout dispatch.

        Each GPU worker only sees its shard and would otherwise number prompts
        from 0 locally, so the concatenated branch output would collide (the same
        id repeats once per worker). ``arange`` over the (already n-repeated)
        gen_batch gives one stable id per prompt instance; the branch rollout
        copies it onto every branch row and :func:`compute_branched_advantage`
        groups by it. No-op semantics off the branch path (only called when
        enabled). Returns ``gen_batch`` for chaining.
        """
        import numpy as np

        gen_batch.non_tensor_batch["branch_global_prompt_id"] = np.arange(len(gen_batch.batch), dtype=np.int64)
        return gen_batch


class RayTempFlowTrainer(TrajectoryBranchMixin, RayWanTrainer):
    """Wan trainer + TempFlow trajectory-branching advantage (Eq.3 / Thm.1)."""
