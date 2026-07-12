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
RAFT Reward Model - Evaluates optical flow quality using RAFT.

This model computes optical flow between consecutive frames and uses
the flow magnitude as a reward signal for video temporal coherence.
"""

import argparse
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from teleboost.reward.contract import BaseRewardModel, RewardConfig
from teleboost.reward.registry import RewardRegistry

logger = logging.getLogger(__name__)


def dict_to_namespace(d: dict) -> argparse.Namespace:
    """Convert a dictionary to an argparse.Namespace."""
    return argparse.Namespace(**d)


@RewardRegistry.register("raft")
class RAFTRewardModel(BaseRewardModel):
    """
    RAFT optical flow reward model.

    Computes optical flow between consecutive video frames using RAFT
    and returns the mean flow magnitude as a reward signal.
    Higher flow = more motion = potentially more dynamic video.

    Config extra_config keys:
        - model_path: Path to RAFT model weights (default from config)
        - stride: Frame stride for flow computation (default: 1)
        - iters: Number of RAFT iterations (default: 20)
    """

    REWARD_KEY = "raft_rewards"

    def __init__(self, config: RewardConfig, global_rank: int, world_size: int):
        super().__init__(config, global_rank, world_size)
        self.raft_model: Optional[torch.nn.Module] = None
        self.stride: int = 1
        self.iters: int = 20

    def init_model(self) -> None:
        """Initialize the RAFT model."""
        if not self.is_active:
            logger.info(f"[raft] Rank {self.global_rank} inactive, skipping init")
            return

        extra = self.config.extra_config or {}
        model_path = extra.get("model_path") or self.config.model_path
        self.stride = extra.get("stride", 1)
        self.iters = extra.get("iters", 20)

        if not model_path:
            raise ValueError("RAFT model requires 'model_path' in config")

        # Configure RAFT arguments
        args_dict = {
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }
        args = dict_to_namespace(args_dict)

        # Load RAFT model
        try:
            from raft.raft import RAFT

            model = RAFT(args)

            # Load state dict, handling 'module.' prefix
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            model.eval()
            model.args.mixed_precision = False

            self.raft_model = model

            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"RAFT model initialized: {num_params:,} params")

        except Exception as e:
            logger.error(f"Failed to load RAFT model: {e}")
            raise

    def _reward_modules(self):
        return [self.raft_model]

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        """
        Compute optical flow score for a video.

        Args:
            video_frames: Video tensor, shape (T, C, H, W)
            caption: Text caption (unused)

        Returns:
            Mean optical flow magnitude
        """
        # Apply stride
        frames = video_frames[:: self.stride]

        if len(frames) < 2:
            logger.warning("Not enough frames for optical flow computation")
            return 0.0

        # Move model and frames to device
        self.raft_model.to(self.get_device())
        frames = frames.to(self.get_device())

        flow_score = self._compute_flow(frames)
        logger.debug(f"RAFT flow score: {flow_score:.4f}")
        return flow_score

    def _compute_flow(self, frames: torch.Tensor) -> float:
        """
        Compute mean optical flow magnitude across frame pairs.

        Args:
            frames: Video frames tensor (T, C, H, W)

        Returns:
            Mean flow magnitude
        """
        from raft.utils import InputPadder

        optical_flows = []

        with torch.no_grad():
            for i in range(len(frames) - 1):
                # Prepare frame pair
                image1 = frames[i].float().unsqueeze(0)
                image2 = frames[i + 1].float().unsqueeze(0)

                # Pad images to be divisible by 8
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                # Compute flow
                _, flow_up = self.raft_model(image1, image2, iters=self.iters, test_mode=True)

                # Compute flow magnitude
                flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
                mean_flow = flow_magnitude.mean().item()
                optical_flows.append(mean_flow)

        return float(np.mean(optical_flows)) if optical_flows else 0.0
