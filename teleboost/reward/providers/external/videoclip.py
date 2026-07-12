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
VideoCLIP Reward Model - Evaluates video-text alignment using VideoCLIP-XL.

This model computes the similarity between video features and text features
using the VideoCLIP-XL model to measure how well the video matches its caption.

NOTE: the VideoCLIP-XL code itself is NOT shipped with TeleBoost — upstream
distributes it under CC-BY-NC-SA-4.0 (non-commercial), which this Apache-2.0
repository cannot redistribute.  See ``init_model`` for how to supply it.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from teleboost.reward.contract import BaseRewardModel, RewardConfig
from teleboost.reward.registry import RewardRegistry

logger = logging.getLogger(__name__)


@RewardRegistry.register("videoclip")
class VideoCLIPRewardModel(BaseRewardModel):
    """
    VideoCLIP-XL video-text alignment reward model.

    Computes cosine similarity between video and text embeddings
    to measure how well a generated video matches its caption.

    Config extra_config keys:
        - model_path: Path to VideoCLIP-XL model weights
        - num_frames: Number of frames to sample from video (default: 8)
        - target_size: Target frame size for preprocessing (default: 224)
    """

    REWARD_KEY = "videoclip_rewards"

    # Normalization constants
    V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def __init__(self, config: RewardConfig, global_rank: int, world_size: int):
        super().__init__(config, global_rank, world_size)
        self.videoclip_model: Optional[torch.nn.Module] = None
        self.num_frames: int = 8
        self.target_size: int = 224

    def init_model(self) -> None:
        """Initialize the VideoCLIP-XL model."""
        if not self.is_active:
            logger.info(f"[videoclip] Rank {self.global_rank} inactive, skipping init")
            return

        extra = self.config.extra_config or {}
        model_path = extra.get("model_path") or self.config.model_path
        self.num_frames = extra.get("num_frames", 8)
        self.target_size = extra.get("target_size", 224)

        if not model_path:
            raise ValueError("VideoCLIP model requires 'model_path' in config")

        try:
            from VideoCLIP_XL.modeling import VideoCLIP_XL
        except ImportError as e:
            if getattr(e, "name", None) in ("utils", "pkg_resources"):
                # A copy IS in place, but it is the verbatim upstream layout:
                # upstream modeling.py uses absolute `from utils. ...` imports
                # that cannot resolve inside a package, and its text_encoder
                # pulls pkg_resources (gone without setuptools).
                raise ImportError(
                    "Found an importable VideoCLIP-XL runtime, but it looks like the unmodified upstream files: change the `from utils....` imports in modeling.py to `from .utils....`, and if `pkg_resources` is missing either install setuptools or change text_encoder.py to `import packaging`."
                ) from e
            raise ImportError(
                "VideoCLIP-XL code is not shipped with TeleBoost: upstream "
                "distributes it under CC-BY-NC-SA-4.0 (non-commercial), which "
                "this Apache-2.0 repository cannot redistribute. To use the "
                "'videoclip' reward, download modeling.py and utils/ from "
                "https://huggingface.co/alibaba-pai/VideoCLIP-XL into an "
                "independently managed VideoCLIP_XL package, accept its license "
                "terms, install it or add its parent to PYTHONPATH before launch, and "
                "adjust modeling.py's `from utils....` imports to relative "
                "`from .utils....` so the copy resolves as a package."
            ) from e

        try:
            model = VideoCLIP_XL()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            self.videoclip_model = model

            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"VideoCLIP-XL model initialized: {num_params:,} params")

        except Exception as e:
            logger.error(f"Failed to load VideoCLIP model: {e}")
            raise

    def _reward_modules(self):
        return [self.videoclip_model]

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        """
        Compute video-text similarity score.

        Args:
            video_frames: Video tensor, shape (T, C, H, W) or (C, T, H, W)
            caption: Text caption to match against

        Returns:
            Cosine similarity score (scaled by 100)
        """
        from VideoCLIP_XL.utils.text_encoder import text_encoder

        self.videoclip_model.to(self.get_device())
        self.videoclip_model.eval()

        with torch.no_grad():
            # Preprocess video
            video_input = self._preprocess_video(video_frames)
            video_input = video_input.to(self.get_device())

            # Get video features
            video_features = self.videoclip_model.vision_model.get_vid_features(video_input)
            video_features = video_features.float()
            video_features = F.normalize(video_features, dim=-1)

            # Get text features
            text_input = text_encoder.tokenize([caption], truncate=True)
            text_input = text_input.to(self.get_device())
            text_features = self.videoclip_model.text_model.encode_text(text_input)
            text_features = text_features.float()
            text_features = F.normalize(text_features, dim=-1)

            # Compute similarity
            similarity = (video_features @ text_features.T) * 100
            score = similarity.view(-1).item()

        logger.debug(f"VideoCLIP similarity score: {score:.4f}")
        return score

    def _preprocess_video(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess video frames for VideoCLIP.

        Args:
            frames: Video tensor, shape (T, C, H, W) or (C, T, H, W)

        Returns:
            Preprocessed tensor, shape (1, T, C, H, W)
        """
        # Handle different input formats
        if frames.dim() == 4:
            # Check if (T, C, H, W) or (C, T, H, W)
            if frames.shape[0] == 3:  # (C, T, H, W)
                frames = frames.permute(1, 0, 2, 3)  # -> (T, C, H, W)

        # Convert to numpy: (T, H, W, C)
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()

        # Sample frames evenly
        total_frames = frames_np.shape[0]
        step = max(1, total_frames // self.num_frames)
        sampled = frames_np[::step][: self.num_frames]

        # Frames arrive RGB from the VAE decode path and VideoCLIP-XL is
        # RGB-trained (arxiv 2410.00741), so no channel flip here.
        vid_tube = []
        for fr in sampled:
            # Resize
            fr = cv2.resize(fr, (self.target_size, self.target_size))

            # Normalize
            fr = self._normalize(fr)

            # Add dimensions: (1, 1, H, W, C)
            fr = np.expand_dims(fr, axis=(0, 1))
            vid_tube.append(fr)

        # Concatenate: (1, T, H, W, C)
        vid_tube = np.concatenate(vid_tube, axis=1)

        # Transpose to (1, T, C, H, W)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))

        return torch.from_numpy(vid_tube).float()

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization."""
        return (data / 255.0 - self.V_MEAN) / self.V_STD
