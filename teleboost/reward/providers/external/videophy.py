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
Videophy Reward Model - Evaluates physical plausibility of videos.

This model uses a pre-trained video understanding model to assess
whether generated videos follow physical laws (e.g., gravity, momentum).
"""

import logging
import re
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from teleboost.reward.contract import BaseRewardModel, RewardConfig
from teleboost.reward.registry import RewardRegistry

logger = logging.getLogger(__name__)

# Default prompt for physics evaluation
PHYSICS_PROMPT = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video follow the physical laws?\nAI: "


@RewardRegistry.register("videophy")
class VideophyRewardModel(BaseRewardModel):
    """
    Video physics plausibility reward model.

    Uses MplugOwl-based model to assess whether videos follow physical laws.
    The model outputs a probability of "Yes" vs "No" to the question
    "Does this video follow the physical laws?"

    Config extra_config keys:
        - model_path: Path to videophy model checkpoint
        - max_length: Maximum sequence length (default: 256)
        - target_size: Target frame size (default: 224)
    """

    REWARD_KEY = "videophy_rewards"

    def __init__(self, config: RewardConfig, global_rank: int, world_size: int):
        super().__init__(config, global_rank, world_size)
        self.videophy_model: Optional[torch.nn.Module] = None
        self.tokenizer = None
        self.processor = None
        self.max_length: int = 256
        self.target_size: int = 224

        # Media token configuration
        self.media_tokens = {"<image>": -1, "<|video|>": -2}
        self.media_lengths = {"<image>": 65, "<|video|>": 65}  # 1 + 64

    def init_model(self) -> None:
        """Initialize the Videophy model."""
        if not self.is_active:
            logger.info(f"[videophy] Rank {self.global_rank} inactive, skipping init")
            return

        extra = self.config.extra_config or {}
        model_path = extra.get("model_path") or self.config.model_path
        self.max_length = extra.get("max_length", 256)
        self.target_size = extra.get("target_size", 224)

        if not model_path:
            raise ValueError("Videophy model requires 'model_path' in config")

        try:
            from transformers.models.llama.tokenization_llama import LlamaTokenizer
            from Videophy.mplug_owl_video import (
                MplugOwlConfig,
                MplugOwlForConditionalGeneration,
                MplugOwlImageProcessor,
                MplugOwlProcessor,
            )

            # Load tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

            # Load model
            logger.info("Loading Videophy model...")
            self.videophy_model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=MplugOwlConfig.from_pretrained(model_path))
            self.videophy_model.eval()
            logger.info("Videophy model loaded")

            # Load processor
            image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
            self.processor = MplugOwlProcessor(image_processor, self.tokenizer)

            num_params = sum(p.numel() for p in self.videophy_model.parameters())
            logger.info(f"Videophy model initialized: {num_params:,} params")

        except Exception as e:
            logger.error(f"Failed to load Videophy model: {e}")
            raise

    def _reward_modules(self):
        return [self.videophy_model]

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        """
        Compute physics plausibility score for a video.

        Args:
            video_frames: Video tensor, shape (T, C, H, W) or (C, T, H, W)
            caption: Text caption (used for context but not primary scoring)

        Returns:
            Probability that the video follows physical laws
        """
        self.videophy_model.to(self.get_device())

        # Prepare inputs
        inputs = self._prepare_inputs(video_frames, caption)

        # Get score
        score = self._compute_physics_score(inputs)

        logger.debug(f"Videophy score: {score:.4f}")
        return score

    def _prepare_inputs(self, video_frames: torch.Tensor, caption: str) -> dict[str, Any]:
        """Prepare model inputs from video frames.

        Expected input shapes:
        - (T, C, H, W): T frames, C channels (usually 3)
        - (C, T, H, W): Alternative format where C=3 and T is frame count

        We distinguish by checking if dim 1 == 3 (RGB channels) for (C,T,H,W) format.
        """
        if video_frames.dim() == 4:
            # Check if it's (C, T, H, W) format by looking at second dimension
            # If shape[1] == 3, it's likely (T, C, H, W) already
            # If shape[0] == 3 and shape[1] != 3, it's (C, T, H, W)
            if video_frames.shape[0] == 3 and video_frames.shape[1] != 3:
                # (C, T, H, W) -> (T, C, H, W)
                video_frames = video_frames.permute(1, 0, 2, 3)
            elif video_frames.shape[1] == 3:
                # Already (T, C, H, W)
                pass
            else:
                # Ambiguous, log warning and assume (T, C, H, W)
                logger.warning(f"Ambiguous video shape {video_frames.shape}, assuming (T, C, H, W)")

        # Now video_frames should be (T, C, H, W)
        logger.debug(f"Video frames shape after processing: {video_frames.shape}")

        # Resize frames
        video_resized = self._resize_frames(video_frames)

        # Extract text tokens
        text_data = self._tokenize_conversation(0)

        # Build batch
        # video_resized is (T, C, H, W)
        # Model expects (B, C, T, H, W)
        video_tensor = video_resized.unsqueeze(0)  # (1, T, C, H, W)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

        batch = {
            "video": video_tensor,
            "text": text_data,
            "caption": caption,
        }

        return self._collate_batch([batch])

    def _resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize video frames to target size with center crop."""
        T, C, H, W = frames.shape
        target_H = target_W = self.target_size

        # Flatten time into batch
        frames = frames.reshape(T, C, H, W)

        # Resize keeping aspect ratio
        scale = max(target_H / H, target_W / W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        resized = F.interpolate(frames, size=(new_H, new_W), mode="bilinear", align_corners=False)

        # Center crop
        top = (new_H - target_H) // 2
        left = (new_W - target_W) // 2
        cropped = resized[:, :, top : top + target_H, left : left + target_W]

        return cropped  # (T, C, H, W)

    def _tokenize_conversation(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize the physics evaluation conversation."""
        max_length = self.max_length
        conversation = PHYSICS_PROMPT

        # Initialize
        if self.tokenizer.bos_token_id > 0:
            enc_chunk = [self.tokenizer.bos_token_id]
        else:
            enc_chunk = []

        label_chunk = []
        enc_length = 0

        # Parse conversation with media tokens
        pattern = "|".join(map(re.escape, list(self.media_tokens.keys()) + ["AI: ", "\nHuman: "]))
        chunk_strs = re.split(f"({pattern})", conversation)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]

        for idx, chunk_str in enumerate(chunk_strs):
            if enc_length >= max_length + 1:
                break

            if idx == 0:
                tokens = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                enc_chunk.extend(tokens)
                enc_length = len(enc_chunk)
                label_chunk = [0] * enc_length
            else:
                if chunk_str in self.media_tokens:
                    media_len = self.media_lengths[chunk_str]
                    if enc_length + media_len > max_length + 1:
                        break
                    enc_chunk.extend([self.media_tokens[chunk_str]] * media_len)
                    enc_length += media_len
                    label_chunk.extend([0] * media_len)
                elif idx > 0 and chunk_strs[idx - 1] == "AI: ":
                    tokens = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                    if enc_length + len(tokens) >= max_length:
                        tokens = tokens[: max_length - enc_length]
                    tokens.append(self.tokenizer.eos_token_id)
                    enc_chunk.extend(tokens)
                    enc_length += len(tokens)
                    label_chunk.extend([1] * len(tokens))
                else:
                    tokens = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                    if enc_length + len(tokens) >= max_length + 1:
                        tokens = tokens[: max_length + 1 - enc_length]
                    enc_chunk.extend(tokens)
                    enc_length += len(tokens)
                    label_chunk.extend([0] * len(tokens))

        # Pad to max_length
        if enc_length < max_length + 1:
            padding_length = max_length + 1 - enc_length
            enc_chunk.extend([self.tokenizer.pad_token_id] * padding_length)
            label_chunk.extend([0] * padding_length)

        # Create masks
        non_padding_mask = [1 if i < enc_length - 1 else 0 for i in range(max_length)]

        # Non-media mask
        enc_tensor = torch.tensor(enc_chunk)
        non_media_mask = (enc_tensor >= 0).long()[1 : max_length + 1]

        return {
            "input_ids": torch.tensor(enc_chunk).long(),
            "non_padding_mask": torch.tensor(non_padding_mask).long(),
            "non_media_mask": non_media_mask,
            "prompt_mask": torch.tensor(label_chunk[1:]).long(),
        }

    def _collate_batch(self, batch: list[dict]) -> dict[str, Any]:
        """Collate a batch of inputs."""
        video = torch.cat([b["video"] for b in batch], dim=0)

        return {
            "pixel_values": None,
            "video_pixel_values": video,
            "input_ids": torch.stack([b["text"]["input_ids"] for b in batch]).long(),
            "labels": torch.stack([b["text"]["input_ids"] for b in batch]).long().clone(),
            "num_images": torch.zeros(len(batch)).long(),
            "num_videos": torch.ones(len(batch)).long(),
            "non_padding_mask": torch.stack([b["text"]["non_padding_mask"] for b in batch]).long(),
            "non_media_mask": torch.stack([b["text"]["non_media_mask"] for b in batch]).long(),
            "prompt_mask": torch.stack([b["text"]["prompt_mask"] for b in batch]).long(),
        }

    def _compute_physics_score(self, inputs: dict[str, Any]) -> float:
        """Compute the physics plausibility score."""
        # Move inputs to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if v.dtype == torch.float:
                    v = v.bfloat16()
                inputs[k] = v.to(self.get_device())

        with torch.no_grad():
            outputs = self.videophy_model(
                pixel_values=inputs["pixel_values"],
                video_pixel_values=inputs["video_pixel_values"],
                labels=None,
                num_images=inputs["num_images"],
                num_videos=inputs["num_videos"],
                input_ids=inputs["input_ids"],
                non_padding_mask=inputs["non_padding_mask"],
                non_media_mask=inputs["non_media_mask"],
                prompt_mask=inputs["prompt_mask"],
            )

            logits = outputs["logits"]
            score = self._extract_entailment_score(logits, inputs["input_ids"])

        return score

    def _extract_entailment_score(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> float:
        """Extract Yes/No probability from logits.

        Paper: VideoPhy / VideoCon-Physics (arxiv 2406.03520).  The
        canonical prompt ends with ``AI:`` and the model emits the
        Yes/No token at the position right before the first padding
        token (or at the last position when the sequence fills the
        whole input window with no padding).

        Implementation notes:

        * The previous Python-loop version
          ``for i in range(...): if pad: i = i - 1; break`` only worked
          by coincidence — the assignment to ``i`` doesn't affect the
          ``range`` iterator, and the immediate ``break`` happens to
          carry the post-decrement value out of the loop.  The same
          loop also silently mishandles two edge cases:

            (1) first token is pad → ``i = -1`` and Python negative
                indexing reads ``logits[0][-1]`` (last position) which
                is the wrong token to use as the entailment site;
            (2) ``input_ids[0]`` is empty → loop body never runs and
                ``i`` is undefined → ``NameError`` on the next line.

          This rewrite makes the intent explicit, drops the Python
          loop, and turns both edge cases into a fail-loud error
          rather than a silent miscalibration.
        * For the canonical prompt this is byte-identical with the
          previous behaviour (the first pad sits right after the
          ``AI:`` so the entailment position is one before it).
        """
        softmax = nn.Softmax(dim=2)
        logits = softmax(logits)

        token_id_yes = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        token_id_no = self.tokenizer.encode("No", add_special_tokens=False)[0]

        seq = input_ids[0]
        if seq.numel() == 0:
            raise RuntimeError("VideoPhy: input_ids[0] is empty — cannot locate the entailment position. This indicates an upstream tokenization / collation bug.")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Tokenizer has no pad token: the prompt fills the whole
            # window without padding, so the entailment site is the
            # final position.
            pos = seq.shape[0] - 1
        else:
            pad_positions = (seq == pad_id).nonzero(as_tuple=True)[0]
            if pad_positions.numel() == 0:
                # No padding at all — entailment site is the last token.
                pos = seq.shape[0] - 1
            else:
                # Position just before the first pad.
                pos = int(pad_positions[0].item()) - 1
                if pos < 0:
                    raise RuntimeError("VideoPhy: input_ids[0] starts with a pad token (first-token padding leaves no entailment position to extract). Check the prompt + tokenizer alignment.")

        prob_yes = logits[0][pos][token_id_yes]
        prob_no = logits[0][pos][token_id_no]
        score = (prob_yes / (prob_yes + prob_no)).item()

        return score
