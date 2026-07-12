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
"""Video VLM judge reward served by the colocated vLLM router.

This is the generic video adapter entrypoint for ``reward.reward_model.adapter=
video_vlm``. The trainer passes ``reward_router_address`` — the HTTP front of
the colocated vLLMReplica spawned at ``init_workers`` time — and this module
POSTs the multimodal chat request directly. Diffusion has no ``solution_str``;
the encoded video data URL is pulled from ``extra_info["video_url"]``.

Output contract: ``{"score": float in [0, 1], "raw": str, "dim1..5": float}``.
``RewardLoopWorker.run_single`` extracts ``score`` and surfaces the rest
as reward_extra_info when used through the upstream reward-loop contract.
"""

from __future__ import annotations

import logging
from typing import Any

from teleboost.reward.adapters._vlm_score import (
    dimension_patterns,
    labelled_score_pattern,
    parse_structured_score,
)

logger = logging.getLogger(__name__)


# Structured Chinese video-quality eval prompt. 5 dims, each 0–100, plus
# an explicit ``合计:XX分`` line we parse below. Output format kept terse
# so weak-instruction models still hit the regex.
EVAL_PROMPT_TEMPLATE = """请你作为一个专业视频质量评估助手，参考以下评分标准和格式，对给定的视频进行多维度质量评估。请严格按照输出格式，以客观、公正、结构化的方式打分。

评估维度（每项满分100分）：
1. 视觉审美（Aesthetics）：构图、光影、色彩、美感。高分=艺术性；扣分=凌乱/失衡/灰暗。
2. 局部变形（Distortion）：人物/物体形态、肢体扭曲、结构突变、突然消失。高分=结构自然；扣分=扭曲/断裂。
3. 视觉伪影与不一致（Artifacts/Inconsistency）：马赛克、色块、条纹、边缘断裂、纹理模糊。高分=一致；扣分=有伪影。
4. 清晰度（Sharpness）：细节、边缘锐利、辨识度。高分=清晰；扣分=模糊。
5. 视觉一致性（Consistency）：时间连贯性、跳帧、镜头突变、画面稳定。高分=连贯；扣分=跳跃/抖动。

评分规则：
- 每维度 0–100，越好越高。
- 合计为五项的算术平均，保留整数。
- 严重失真大胆给低分（<30）。
- 不同视频之间评分应拉开差距。

输出格式（严格遵守）：
dim1:XX分,dim2:XX分,dim3:XX分,dim4:XX分,dim5:XX分,合计:XX分

风格要求：
- 禁止解释性文字。
- 禁止"我认为/可能/大致"等模糊词。
- 一次性返回评估结果。

视频描述（原始 prompt）：{caption}
"""


_TOTAL_PATTERNS = tuple(labelled_score_pattern(label) for label in ("合计", "综合得分", "总分", "最终", "评分"))

_DIM_PATTERNS = dimension_patterns(
    (
        "dim1_aesthetics",
        "dim2_distortion",
        "dim3_artifacts",
        "dim4_sharpness",
        "dim5_consistency",
    )
)


def parse_eval_score(output_text: str) -> dict[str, Any]:
    """Pull the total score and per-dim scores out of the video VLM reply.

    Returns ``{"score_raw": 0..100, "score": 0..1, "dim?_*": float, "raw": str}``.
    ``score`` is normalized to [0, 1] because GRPO advantage z-scores work
    cleaner on small magnitudes; downstream reward_manager wraps it back
    into the training tensor.
    """
    return parse_structured_score(
        output_text,
        total_patterns=_TOTAL_PATTERNS,
        dimensions=_DIM_PATTERNS,
    )


def _build_messages(video_url: str, caption: str, media_uuid: str | None = None) -> list[dict[str, Any]]:
    """Construct a chat message using vllm's OpenAI-style video_url.

    vllm's OpenAI ``/v1/chat/completions`` part-type vocabulary is
    OpenAI-extended: ``image_url`` / ``audio_url`` / ``video_url``,
    each with a nested ``{"url": ...}``. The Hugging Face
    ``transformers`` chat-template format (``{"type": "video",
    "video": "..."}``) is a different convention and doesn't apply
    here — vllm 0.17's ``_parse_chat_message_content_part`` raises
    ``NotImplementedError: Unknown part type: video`` for it.
    Using ``video_url`` lets vllm decode the video frames internally
    (matching the pre-X3 ``fetch_video`` path in our v0.4 fork), so
    reward distributions stay comparable across upgrades.
    """
    eval_text = EVAL_PROMPT_TEMPLATE.format(caption=caption or "")
    video_part = {
        "type": "video_url",
        "video_url": {"url": video_url},
    }
    if media_uuid:
        # vLLM uses a content-part UUID as the multimodal cache key.
        # Supplying a per-sample UUID prevents repeated/fixed videos in the
        # same reward batch from colliding in vLLM's processor cache.
        video_part["uuid"] = media_uuid
    return [
        {
            "role": "user",
            "content": [
                video_part,
                {"type": "text", "text": eval_text},
            ],
        }
    ]


async def _post_chat_completion(
    router_address: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 120.0,
    max_retries: int = 4,
) -> str:
    """POST to the colocated video VLM ``/v1/chat/completions``.

    Returns the assistant's text response. Retries on transport errors
    and 5xx with exponential backoff. We bypass upstream's
    ``RewardLoopWorker._post_request`` because that helper hardcodes
    the ``classify`` / ``v1/embeddings`` / ``v1/completions`` endpoints;
    multimodal chat needs ``v1/chat/completions``.
    """
    url = f"http://{router_address}/v1/chat/completions"
    import aiohttp

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status >= 500:
                        raise RuntimeError(f"reward server {resp.status}: {await resp.text()}")
                    resp.raise_for_status()
                    body = await resp.json()
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            wait_s = min(2**attempt, 8)
            logger.warning(
                "video VLM reward POST attempt %d/%d failed: %s; retry in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                wait_s,
            )
            import asyncio

            await asyncio.sleep(wait_s)
    raise RuntimeError(f"video VLM reward POST exhausted retries: {last_exc}")


async def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    *,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    **_unused,
) -> dict[str, Any]:
    """A video VLM judges the rollout video. Async to overlap samples.

    Wired by ``RewardLoopWorker.run_single`` (see upstream
    ``verl.experimental.reward_loop.reward_manager.naive``).

    Args:
        data_source: ignored. Diffusion has a single data_source.
        solution_str: ignored. Diffusion has no text response.
        ground_truth: optional dict with ``caption``.
        extra_info: must carry a self-contained ``video_url`` data URL.
            ``video_path`` remains a local single-node compatibility form.
            Optional ``caption`` falls back to
            ``ground_truth.get("caption")``.
        reward_router_address: host:port of the colocated vLLM server.
            None means the reward server isn't up — return default.
        reward_model_tokenizer: ignored. The vLLM endpoint tokenizes server-side.

    Returns:
        dict with at least ``score`` (float in [0, 1]).
    """
    extra_info = extra_info or {}
    video_url = extra_info.get("video_url")
    if not video_url:
        video_path = extra_info.get("video_path")
        if video_path:
            video_url = f"file://{video_path}"
    if not video_url:
        logger.error("compute_score: no video_url/video_path in extra_info; returning 0")
        return {"score": 0.0, "raw": "<missing video URL>", "failed": True}

    caption = extra_info.get("caption")
    if caption is None and isinstance(ground_truth, dict):
        caption = ground_truth.get("caption")
    caption = caption or ""
    media_uuid = extra_info.get("media_uuid")
    if media_uuid is not None:
        media_uuid = str(media_uuid)

    if reward_router_address is None:
        logger.error("compute_score: reward_router_address is None; returning 0")
        return {"score": 0.0, "raw": "<no router>", "failed": True}

    messages = _build_messages(video_url, caption, media_uuid)
    payload = {
        "model": extra_info.get("reward_model_name", "video-vlm"),
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    try:
        text = await _post_chat_completion(reward_router_address, payload)
    except Exception as exc:
        logger.exception("compute_score: POST failed, returning 0; %s", exc)
        return {"score": 0.0, "raw": f"<error: {exc}>", "failed": True}

    return parse_eval_score(text)
