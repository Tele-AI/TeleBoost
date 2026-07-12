# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared reward adapter helpers."""

from __future__ import annotations

from typing import Any

import torch


def build_reward_tensor(scores: torch.Tensor, batch_size: int):
    from tensordict import TensorDict
    from verl import DataProto

    return DataProto(
        batch=TensorDict({"rewards": scores}, batch_size=[batch_size]),
        non_tensor_batch={},
    )


def require_judge_success(results: list[dict[str, Any]], what: str) -> None:
    failed = [result for result in results if result.get("failed")]
    if not failed:
        return
    failed_fraction = len(failed) / max(len(results), 1)
    raise RuntimeError(f"{what} judge failed for {len(failed)}/{len(results)} samples ({failed_fraction:.1%}). First failure: {failed[0].get('raw', '')!r}. Refusing to train on failed judge rewards; fix the judge before resuming.")


__all__ = ["build_reward_tensor", "require_judge_success"]
