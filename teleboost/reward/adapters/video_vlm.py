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
"""Video-VLM reward adapter for diffusion batches.

Upstream verl keeps model lifecycle in ``RewardModelManager`` and calls
sample-level ``compute_score`` functions from reward managers. This adapter
fills the gap for diffusion batches: convert ``video_frames`` tensors to mp4
files, call the async ``teleboost.reward.adapters.video_vlm_score.compute_score``
against the router address supplied by upstream, and assemble a reward tensor.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import tempfile
import uuid
from typing import Any

import numpy as np
import torch

from teleboost.config.access import select
from teleboost.reward.routing import is_video_vlm_reward_config
from teleboost.reward.adapters.common import build_reward_tensor, require_judge_success


def _reward_model_served_name(config: Any, *env_names: str) -> str:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return str(select(config, "reward.reward_model.model_path", "") or "")


def compute_video_vlm_reward(config: Any, data: Any, *, reward_router_address: str):
    if not is_video_vlm_reward_config(config):
        raise RuntimeError("video_vlm adapter requires reward.reward_model.adapter=video_vlm")

    import cv2

    from teleboost.reward.adapters.video_vlm_score import compute_score

    if "video_frames" not in data.batch.keys():
        raise ValueError("video_vlm reward requires batch['video_frames']")
    videos = data.batch["video_frames"]
    batch_size = videos.shape[0]
    non_tensor = data.non_tensor_batch or {}
    captions_arr = non_tensor.get("caption")
    captions = list(captions_arr) if captions_arr is not None else [""] * batch_size
    if len(captions) != batch_size:
        raise ValueError(f"video_vlm reward captions length {len(captions)} != videos length {batch_size}")

    tmp_dir = tempfile.mkdtemp(prefix="teleboost_video_vlm_reward_")
    try:
        video_urls: list[str] = []
        fps = float(select(config, "actor_rollout_ref.video_fps", 8) or 8)
        for index in range(batch_size):
            sample = videos[index]
            if sample.dim() != 4:
                raise ValueError(f"video_frames[{index}].dim()={sample.dim()}, expected 4 (C,T,H,W)")
            channels, frames, height, width = sample.shape
            if channels != 3:
                # A non-RGB sample written into a color VideoWriter produces a
                # silently-empty/garbage mp4, then the judge scores garbage.
                raise ValueError(f"video_frames[{index}] has {channels} channels; video_vlm reward expects 3-channel RGB (C,T,H,W).")
            # NumPy has no bfloat16 representation.
            frame_np = sample.permute(1, 2, 3, 0).float().clamp(0, 1).mul(255.0).cpu().numpy()
            frame_np = frame_np.astype(np.uint8)
            out_path = os.path.join(tmp_dir, f"rollout_{index:04d}.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open {out_path} (mp4v codec missing?); a zero-byte mp4 would silently score 0 at the judge.")
            for frame_index in range(frames):
                writer.write(cv2.cvtColor(frame_np[frame_index], cv2.COLOR_RGB2BGR))
            writer.release()
            with open(out_path, "rb") as video_file:
                encoded = base64.b64encode(video_file.read()).decode("ascii")
            video_urls.append(f"data:video/mp4;base64,{encoded}")

        served_name = _reward_model_served_name(
            config,
            "VIDEO_VLM_REWARD_MODEL_SERVED_NAME",
            "QWEN_REWARD_MODEL_SERVED_NAME",
        )

        async def _one(video_url: str, caption: str):
            return await compute_score(
                data_source="teleboost_diffusion",
                solution_str=None,
                ground_truth={"caption": caption},
                extra_info={
                    "video_url": video_url,
                    "media_uuid": f"teleboost-video-{uuid.uuid4().hex}",
                    "caption": caption,
                    "reward_model_name": served_name,
                },
                reward_router_address=reward_router_address,
            )

        async def _all():
            return await asyncio.gather(*[_one(video_url, caption) for video_url, caption in zip(video_urls, captions, strict=True)])

        results = asyncio.run(_all())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    require_judge_success(results, "video VLM")
    scores = torch.tensor([float(r.get("score", 0.0)) for r in results], dtype=torch.float32)
    return build_reward_tensor(scores, batch_size)


__all__ = ["compute_video_vlm_reward"]
