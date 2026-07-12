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
Aesthetic Reward Model - Evaluates aesthetic quality of video frames using CLIP + Linear.

This model computes an aesthetic score based on CLIP visual features
processed through a trained linear regression head.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from teleboost.reward.contract import BaseRewardModel, RewardConfig
from teleboost.reward.registry import RewardRegistry

logger = logging.getLogger(__name__)

# Try to import interpolation mode for transforms
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image

    BICUBIC = Image.BICUBIC


def clip_transform(n_px: int):
    """Create CLIP-compatible image transform."""
    return transforms.Compose([transforms.Resize(n_px, interpolation=BICUBIC, antialias=False), transforms.CenterCrop(n_px), transforms.Lambda(lambda x: x.float().div(255.0)), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])


@RewardRegistry.register("aesthetic")
class AestheticRewardModel(BaseRewardModel):
    """
    Aesthetic quality reward model.

    Uses a CLIP vision encoder to extract features, then applies a trained
    linear head to predict aesthetic scores.

    Config extra_config keys:
        - clip_model_path: Path to CLIP model weights
        - aes_model_path: Path to aesthetic linear head weights
    """

    REWARD_KEY = "aes_rewards"

    def __init__(self, config: RewardConfig, global_rank: int, world_size: int):
        super().__init__(config, global_rank, world_size)
        self.clip_model: Optional[nn.Module] = None
        self.aes_model: Optional[nn.Module] = None
        self.transform = clip_transform(224)

    def init_model(self) -> None:
        """Initialize CLIP and aesthetic models."""
        if not self.is_active:
            logger.info(f"[aesthetic] Rank {self.global_rank} inactive, skipping init")
            return

        extra = self.config.extra_config or {}
        clip_path = extra.get("clip_model_path")
        aes_path = extra.get("aes_model_path")

        if not clip_path or not aes_path:
            raise ValueError("aesthetic model requires 'clip_model_path' and 'aes_model_path' in extra_config")

        # Load CLIP model
        try:
            from teleboost.models.offline_clip import create_offline_clip_model

            self.clip_model = create_offline_clip_model(clip_path, "cpu")
            logger.info(f"Loaded CLIP model from {clip_path}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        # Load aesthetic head
        self.aes_model = self._load_aesthetic_head(aes_path)

        # Log parameter counts
        # Note: clip_model is ViTL14CLIPModel wrapper, not nn.Module directly
        # Access the visual encoder for parameter count
        if hasattr(self.clip_model, "visual"):
            clip_params = sum(p.numel() for p in self.clip_model.visual.parameters())
        else:
            clip_params = 0
            logger.warning("Could not count CLIP parameters")
        aes_params = sum(p.numel() for p in self.aes_model.parameters())
        logger.info(f"Aesthetic model initialized: CLIP={clip_params:,}, Head={aes_params:,} params")

    def _load_aesthetic_head(self, path: str) -> nn.Module:
        """Load the aesthetic linear regression head."""
        model = nn.Linear(768, 1)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Loaded aesthetic head from {path}")
        return model

    def _reward_modules(self):
        return [self.clip_model, self.aes_model]

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        """Compute aesthetic score for a single video.

        Scores **only the first frame** of each video as a proxy for the
        whole clip.

        The LAION ``improved-aesthetic-predictor`` head we use here is an
        *image-only* regression on top of CLIP ViT-L/14 (trained on
        per-image aesthetic ratings); it has no notion of temporal
        coherence.  A paper-faithful application to video would be a
        per-frame mean (treat each frame as an independent image,
        average the scores) — but at ``num_frames=49`` that is 49× the
        CLIP forward cost per reward call.

        First-frame is chosen as an N×-cheaper proxy that empirically
        correlates with overall clip aesthetic quality (diffusion videos
        are usually stylistically committed by frame 0).  The trade-off:
        **aesthetic will not see late-clip degradation** — motion blur,
        last-frame collapse, and any aesthetic drift after frame 0 are
        invisible to this reward.  If you observe such late-frame
        collapse during training while ``train/rewards_aesthetic`` stays
        high, switch to a per-frame-mean aggregation here.

        Args:
            video_frames: Video tensor, shape (T, C, H, W).
            caption: Text caption (unused for aesthetic scoring).

        Returns:
            Normalized aesthetic score.
        """
        # Take first frame: (T, C, H, W) -> (C, H, W)
        frame = video_frames[0]

        # Apply transform and add batch dim
        transformed = self.transform(frame).unsqueeze(0)
        transformed = transformed.to(self.get_device())

        # Move models to device
        self.clip_model.to(self.get_device())
        self.aes_model.to(self.get_device())

        with torch.no_grad():
            # Extract CLIP features
            features = self.clip_model.encode_image(transformed).float()
            features = F.normalize(features, dim=-1)

            # Compute aesthetic score
            score = self.aes_model(features).squeeze(-1)

        # Normalize to reasonable range and return mean
        mean_score = (score / 10.0).mean().item()
        logger.debug(f"Aesthetic score: {mean_score:.4f}")
        return mean_score
