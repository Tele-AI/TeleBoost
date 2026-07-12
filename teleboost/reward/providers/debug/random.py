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
"""Zero-weight debug reward — uniform random score per sample.

Purpose: bring up / validate the rollout -> reward -> FSDP-actor GRPO pipeline
(e.g. a multi-GPU FSDP-DP smoke) WITHOUT needing any reward-model weights on
disk. Returns an i.i.d. U(0,1) score per sample, so per-prompt groups have
non-zero reward variance and GRPO advantages are non-degenerate (exercises the
full advantage/normalize/update path). NOT a real reward — do not train for
quality with this.

Use: reward.reward_model.type=single reward.reward_model.model_name=random (REWARD_MODEL_PATH
is ignored but the launcher still requires it be set — pass any placeholder).
"""

import logging

import torch

from teleboost.reward.contract import BaseRewardModel
from teleboost.reward.registry import RewardRegistry

logger = logging.getLogger(__name__)


@RewardRegistry.register("random")
class RandomRewardModel(BaseRewardModel):
    """Weight-free U(0,1) reward for pipeline/infra validation."""

    REWARD_KEY = "random_rewards"

    def init_model(self) -> None:
        if not self.is_active:
            logger.info(f"[random] Rank {self.global_rank} inactive, skipping init")
            return
        logger.info("[random] zero-weight debug reward initialized (no weights loaded)")

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        # Ignore frames/caption by design; just return a random score.
        return float(torch.rand(1).item())
