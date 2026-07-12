# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tensor protocols shared by distributed reward workers.

This module intentionally depends only on PyTorch.  Reward worker classes also
depend on Ray, verl, and TensorDict, which made their collective correctness
tests impossible in a lightweight checkout.  Keeping the numerical/collective
protocol here lets CPU tests exercise the exact production implementation.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def zscore_normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize a complete reward vector, preserving degenerate batches."""
    if values.numel() <= 1:
        return values
    mean = values.mean()
    std = values.std()
    if not torch.isfinite(std) or float(std.item()) == 0.0:
        return values - mean
    return (values - mean) / std


def normalize_gathered_rewards(rewards: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    """Apply a reward head's normalization exactly once after global gather."""
    return zscore_normalize(rewards) if enabled else rewards


def synchronized_failure_count(*, local_failed: bool, device: torch.device | str | int) -> int:
    """Return the number of failed ranks before any data-dependent gather."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return int(local_failed)
    failure = torch.tensor([int(local_failed)], dtype=torch.int32, device=device)
    dist.all_reduce(failure, op=dist.ReduceOp.SUM)
    return int(failure.item())


def allgather_variable_batch(
    local_rewards: torch.Tensor,
    *,
    collective_device: torch.device | str | int,
    expected_size: int | None = None,
) -> torch.Tensor:
    """Gather uneven contiguous rank shards by length, pad, gather, and trim."""
    if not torch.is_tensor(local_rewards) or local_rewards.ndim < 1:
        raise TypeError(f"local_rewards must be a tensor with a batch dimension, got {type(local_rewards).__name__} shape={getattr(local_rewards, 'shape', None)}")

    world_size = dist.get_world_size()
    local_rewards = local_rewards.to(collective_device)
    local_length = torch.tensor(
        [local_rewards.shape[0]],
        dtype=torch.int64,
        device=collective_device,
    )
    gathered_lengths = [torch.zeros_like(local_length) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, local_length)
    lengths = [int(length.item()) for length in gathered_lengths]

    total_size = sum(lengths)
    if expected_size is not None and total_size != expected_size:
        raise RuntimeError(f"Joint reward gather received rank lengths {lengths} (total={total_size}), expected full batch size {expected_size}")
    if total_size == 0:
        return local_rewards[:0]

    max_length = max(lengths)
    padded = local_rewards.new_zeros((max_length, *local_rewards.shape[1:]))
    padded[: local_rewards.shape[0]] = local_rewards
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat(
        [rank_rewards[:rank_length] for rank_rewards, rank_length in zip(gathered, lengths, strict=True)],
        dim=0,
    )


__all__ = [
    "allgather_variable_batch",
    "normalize_gathered_rewards",
    "synchronized_failure_count",
    "zscore_normalize",
]
