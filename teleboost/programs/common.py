# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by peer program/backend implementations."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np

from teleboost.config.access import select


def enabled_driver_phase_algorithms(config: Any) -> list[str]:
    """Return enabled Wan driver-phase algorithms in canonical order."""

    enabled = []
    if bool(select(config, "algorithm.bgpo.enable", False)):
        enabled.append("bgpo")
    if bool(select(config, "actor_rollout_ref.pixel_weight.enable", False)):
        enabled.append("vipo")
    if bool(select(config, "actor_rollout_ref.actor.tempflow.branch.enable", False)):
        enabled.append("tempflow")
    return enabled


def require_actor_strategy(
    config: Any,
    *,
    backend_name: str,
    supported: Collection[str],
) -> str:
    """Validate an actor strategy without imposing one on the contract."""

    normalized_supported = tuple(sorted({str(value).strip().lower() for value in supported if str(value).strip()}))
    if not normalized_supported:
        raise ValueError(f"{backend_name} backend declared no supported actor strategies")

    strategy_value = select(config, "actor_rollout_ref.actor.strategy")
    if strategy_value is None:
        raise ValueError(f"Missing required configuration key actor_rollout_ref.actor.strategy; {backend_name} GRPO supports {list(normalized_supported)}.")
    strategy = str(strategy_value).strip().lower()
    if strategy not in normalized_supported:
        if len(normalized_supported) == 1:
            requirement = f"requires actor_rollout_ref.actor.strategy={normalized_supported[0]}"
        else:
            requirement = f"supports actor_rollout_ref.actor.strategy in {list(normalized_supported)}"
        raise NotImplementedError(f"{backend_name} GRPO currently {requirement}; got {strategy!r}.")
    return strategy


def prompt_collate_function(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate prompt-only samples for image or video prompt backends."""

    captions = []
    passthrough: dict[str, list[Any]] = {key: [] for key in ("id", "index", "prior")}
    for row in data_list:
        prompt = next(
            (row[key] for key in ("caption", "prompt", "text", "raw_prompt") if key in row),
            None,
        )
        extra_info = row.get("extra_info")
        if prompt is None and isinstance(extra_info, dict):
            prompt = next(
                (extra_info[key] for key in ("caption", "prompt", "text") if key in extra_info),
                None,
            )
        if prompt is None:
            raise KeyError("Prompt sample must contain caption/prompt/text/raw_prompt")
        captions.append(str(prompt))
        for key in passthrough:
            passthrough[key].append(row.get(key))

    out = {"caption": np.asarray(captions, dtype=object)}
    for key, values in passthrough.items():
        if any(value is not None for value in values):
            out[key] = np.asarray(values, dtype=object)
    return out


__all__ = [
    "enabled_driver_phase_algorithms",
    "prompt_collate_function",
    "require_actor_strategy",
]
