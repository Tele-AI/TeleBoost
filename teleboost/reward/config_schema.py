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
"""Reward config schema normalization.

Runtime code reads ``config.reward.reward_model.*``. This module keeps that
shape stable and normalizes ``reward.reward_model.enable`` to mean "launch
upstream RewardModelManager" instead of "use a TeleBoost reward worker group".
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf, open_dict

from teleboost.reward.routing import VIDEO_VLM_ADAPTER

logger = logging.getLogger(__name__)


# TeleBoost-specific keys not handled by upstream `migrate_legacy_reward_impl`.
# They're hoisted aside before the upstream call and re-attached under
# `config.reward.reward_model.*` afterwards.
_TELEBOOST_REWARD_MODEL_KEYS = (
    "type",  # single | joint
    "model_name",  # hps | aesthetic | raft | videoclip | videophy | qwen | ...
    "adapter",  # video_vlm — rollout-to-reward data adapter
    "strategy",  # diffusion | fsdp | megatron — TeleBoost worker strategy
    "joint",  # joint reward sub-config (per-model weights, mps, etc.)
    "weights",  # weights table for joint reward
    "extra_config",  # reward-model-specific extras (model paths, etc.)
    "normalize",  # whether to z-score the reward
    "micro_batch_size_per_gpu",
)


def normalize_reward_config(config: DictConfig) -> DictConfig:
    """Normalize reward config in-place and return ``config``.

    The canonical public shape is ``reward.reward_model.*``. If upstream verl
    defaults also provide a top-level ``reward_model.*`` block, fold the
    fields TeleBoost needs into the canonical namespace before runtime code
    reads the config.
    """
    if "reward_model" not in config:
        _normalize_reward_model_enable(config)
        return config

    extras = {}
    source_rm = config.reward_model
    for key in _TELEBOOST_REWARD_MODEL_KEYS:
        value = source_rm.get(key) if hasattr(source_rm, "get") else None
        if value is not None:
            extras[key] = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value

    _prune_rollout_keys_for_upstream(config, source_rm)

    from verl.experimental.reward_loop import migrate_legacy_reward_impl

    migrate_legacy_reward_impl(config)

    with open_dict(config.reward.reward_model):
        for key, value in extras.items():
            config.reward.reward_model[key] = value

    _normalize_reward_model_enable(config)

    adapter = str(config.reward.reward_model.get("adapter") or "").lower()
    logger.info(
        "Normalized reward config (adapter=%s, upstream RewardModelManager=%s)",
        adapter or "<none>",
        config.reward.reward_model.enable,
    )
    return config


def _prune_rollout_keys_for_upstream(config: DictConfig, source_rm: DictConfig) -> None:
    source_rollout = source_rm.get("rollout") if hasattr(source_rm, "get") else None
    if not isinstance(source_rollout, DictConfig):
        return

    target_rollout_keys = set()
    if "reward" in config and "reward_model" in config.reward and "rollout" in config.reward.reward_model:
        target_rollout_keys = set(config.reward.reward_model.rollout.keys())
    if not target_rollout_keys:
        return

    stale = [key for key in list(source_rollout.keys()) if key not in target_rollout_keys]
    if not stale:
        return

    with open_dict(source_rollout):
        for key in stale:
            del source_rollout[key]
    logger.info("Dropped upstream reward_model.rollout keys not in the active schema: %s", stale)


def _normalize_reward_model_enable(config: DictConfig) -> None:
    """Align upstream enable semantics with TeleBoost reward dispatch.

    Upstream ``reward.reward_model.enable`` means "start RewardModelManager"
    for vLLM/SGLang reward models. TeleBoost's HPS/joint/etc. worker-group
    rewards are controlled by ``trainer.use_rm`` and must not launch the
    upstream manager.
    """
    adapter = str(config.reward.reward_model.get("adapter") or "").lower()
    with open_dict(config.reward.reward_model):
        config.reward.reward_model.enable = adapter == VIDEO_VLM_ADAPTER and bool(config.reward.reward_model.get("enable"))


__all__ = ["normalize_reward_config"]
