# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared Wan rollout/recompute transition invariants.

Wan generation normally operates on one unbatched latent ``(C,T,H,W)``,
while a few tests and adapters use ``(B,C,T,H,W)``.  Keeping that distinction
explicit is important: treating the leading channel axis of a 4-D latent as a
batch axis turns one sample log-probability into ``C`` policy terms.

This module is intentionally CPU-only and dependency-light so the same shape
contract is used by rollout, actor recompute, and unit tests.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from teleboost.algorithms.grpo.policy_scalars import assert_per_sample_shape
from teleboost.algorithms.rollout_contract import VALID_POLICY_FORWARD_KINDS

__all__ = [
    "align_wan_log_probs_for_loss",
    "compute_wan_pixel_weight_maps_with_fallback",
    "compute_flow_grpo_window",
    "finalize_wan_transition_fields",
    "make_wan_solver_metadata",
    "reduce_wan_log_density",
    "validate_wan_solver_metadata",
]


WAN_SOLVER_METADATA_KEYS = (
    "solver_id",
    "sigma_form",
    "logprob_reduction",
    "policy_forward_kind",
)


def make_wan_solver_metadata(
    *,
    batch_size: int,
    sigma_form: str,
    pixel_enabled: bool,
) -> dict[str, np.ndarray]:
    """Build per-sample solver metadata that survives DataProto transforms."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    reduction = "channel_sum_dense" if pixel_enabled else "mean"
    values = {
        "solver_id": f"wan_{sigma_form}",
        "sigma_form": sigma_form,
        "logprob_reduction": reduction,
        "policy_forward_kind": "training_forward",
    }
    return {key: np.full(batch_size, value, dtype=object) for key, value in values.items()}


def validate_wan_solver_metadata(
    metadata: Mapping[str, object],
    *,
    batch_size: int,
    expected_contract,
) -> None:
    """Validate the actual rollout metadata at the actor/loss boundary."""
    missing = [key for key in WAN_SOLVER_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Wan rollout is missing solver metadata required for policy recompute: {missing}")

    scalar_values: dict[str, str] = {}
    for key in WAN_SOLVER_METADATA_KEYS:
        values = np.asarray(metadata[key], dtype=object).reshape(-1)
        if values.size != batch_size:
            raise ValueError(f"Wan rollout metadata {key!r} has {values.size} values, expected batch size {batch_size}")
        unique = {str(value) for value in values.tolist()}
        if len(unique) != 1:
            raise ValueError(f"Wan rollout metadata {key!r} is inconsistent within the batch: {sorted(unique)}")
        scalar_values[key] = unique.pop()

    policy_forward_kind = scalar_values["policy_forward_kind"]
    if policy_forward_kind not in VALID_POLICY_FORWARD_KINDS:
        raise ValueError(f"policy_forward_kind={policy_forward_kind!r} invalid; expected one of {VALID_POLICY_FORWARD_KINDS}")
    expected_contract.assert_matches_record(
        solver_id=scalar_values["solver_id"],
        sigma_form=scalar_values["sigma_form"],
        logprob_reduction=scalar_values["logprob_reduction"],
    )


def compute_wan_pixel_weight_maps_with_fallback(
    pixel_weight_fn: Callable[..., torch.Tensor],
    *,
    videos: torch.Tensor,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, Exception | None]:
    """Compute VIPO maps, returning uniform dense weights on batch failure.

    ``latents`` is the finalized rollout tensor ``(B,S,C,T,H,W)``. Its latent
    geometry is authoritative even when video decoding or DINO inference is
    the operation that failed.
    """
    if latents.ndim != 6 or latents.shape[1] <= 0:
        raise ValueError(f"Wan transition latents must be (B,S,C,T,H,W) with S>0, got {tuple(latents.shape)}")
    batch_size = int(latents.shape[0])
    target_time = int(latents.shape[3])
    target_size = (int(latents.shape[4]), int(latents.shape[5]))
    expected_shape = (batch_size, target_time, *target_size)

    try:
        if videos.ndim != 5 or videos.shape[0] != batch_size:
            raise ValueError(f"video_frames must be (B,C,T,H,W) with the rollout batch B={batch_size}, got {tuple(videos.shape)}")
        maps = pixel_weight_fn(
            videos=videos,
            target_size=target_size,
            target_time=target_time,
            device=videos.device,
        )
        if not torch.is_tensor(maps) or tuple(maps.shape) != expected_shape:
            shape = tuple(maps.shape) if torch.is_tensor(maps) else type(maps)
            raise ValueError(f"pixel_weight_fn returned shape {shape}, expected {expected_shape}")
        return maps, None
    except Exception as error:
        fallback = torch.ones(
            expected_shape,
            dtype=torch.float32,
            device=latents.device,
        )
        return fallback, error


def reduce_wan_log_density(
    log_density: torch.Tensor,
    *,
    pixel_enabled: bool,
) -> torch.Tensor:
    """Apply Wan's log-prob reduction without guessing the leading axis.

    ``(C,T,H,W)`` is one unbatched sample and therefore reduces to a scalar in
    baseline mode. ``(B,C,T,H,W)`` keeps ``B`` and reduces to ``[B]``.  VIPO
    keeps the dense temporal/spatial axes and sums only the channel axis.
    """
    if log_density.ndim not in (4, 5):
        raise ValueError(f"Wan log-density must be (C,T,H,W) or (B,C,T,H,W), got shape {tuple(log_density.shape)}")

    batched = log_density.ndim == 5
    channel_dim = 1 if batched else 0
    if pixel_enabled:
        return log_density.sum(dim=channel_dim)
    if batched:
        return log_density.mean(dim=(1, 2, 3, 4))
    return log_density.mean()


def align_wan_log_probs_for_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    batch_size: int,
    pixel_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize rollout/recompute ranks and enforce the loss boundary.

    Baseline Wan admits exactly one scalar old/new log-prob and advantage per
    sample.  VIPO admits exactly one dense ``(T,H,W)`` field per sample.  These
    assertions run before subtraction/``exp`` so PyTorch broadcasting cannot
    silently reinterpret channels or mix samples.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if not pixel_enabled:
        new_log_probs = new_log_probs.flatten()
        old_log_probs = old_log_probs.flatten()
        assert_per_sample_shape(
            batch_size,
            new_log_probs,
            old_log_probs,
            advantages,
            names=["new_log_probs", "old_log_probs", "advantages"],
        )
        return new_log_probs, old_log_probs

    # ``wan_step`` returns (T,H,W) for the usual one-sample actor micro-batch,
    # whereas the stored rollout tensor retains its leading batch dimension.
    if new_log_probs.ndim + 1 == old_log_probs.ndim:
        new_log_probs = new_log_probs.unsqueeze(0)
    elif new_log_probs.ndim == old_log_probs.ndim + 1 and new_log_probs.shape[0] == 1:
        new_log_probs = new_log_probs.squeeze(0)

    expected = tuple(old_log_probs.shape)
    if tuple(new_log_probs.shape) != expected:
        raise AssertionError(f"VIPO log-prob shape mismatch at loss boundary: new={tuple(new_log_probs.shape)}, old={expected}")
    if not expected or expected[0] != batch_size:
        raise AssertionError(f"VIPO log-probs must start with B={batch_size}, got {expected}")
    if tuple(advantages.shape) != expected:
        raise AssertionError(f"VIPO advantages must exactly match dense log-probs at loss boundary: advantages={tuple(advantages.shape)}, log_probs={expected}")
    return new_log_probs, old_log_probs


def compute_flow_grpo_window(
    window_size: int,
    window_range: Sequence[int],
    num_steps: int,
    *,
    randint: Callable[[int, int], int] | None = None,
) -> tuple[int, int] | None:
    """Choose a contiguous SDE window with at least one trainable transition.

    The global transition ``num_steps - 1`` ends at sigma zero and is removed
    before training.  A size-one window is therefore never allowed to select
    only that terminal transition; this avoids a random empty batch.
    """
    window_size = int(window_size)
    num_steps = int(num_steps)
    if window_size <= 0:
        return None
    if num_steps < 2:
        raise ValueError("Flow-GRPO needs at least two sampling steps so one transition remains after excluding the final sigma->0 transition")
    if len(window_range) != 2:
        raise ValueError(f"sde_window_range must contain [start, end], got {tuple(window_range)}")

    start_min, start_max = (int(window_range[0]), int(window_range[1]))
    if start_min < 0:
        raise ValueError(f"sde_window_range start must be >= 0, got {start_min}")
    start_max = min(start_max, num_steps)
    if start_max <= start_min:
        raise ValueError(f"sde_window_range has no transitions after clamping: [{start_min}, {start_max})")
    if window_size > start_max - start_min:
        raise ValueError(f"sde_window_size={window_size} exceeds available range [{start_min}, {start_max})")

    latest_start = start_max - window_size
    # Any valid window must contain an index before the terminal transition.
    latest_start = min(latest_start, num_steps - 2)
    if latest_start < start_min:
        raise ValueError("The configured Flow-GRPO window contains only the final sigma->0 transition; widen or move sde_window_range")

    choose = randint or random.randint
    start = start_min if latest_start == start_min else int(choose(start_min, latest_start))
    if not start_min <= start <= latest_start:
        raise ValueError(f"window sampler returned start={start}, expected {start_min} <= start <= {latest_start}")
    return start, start + window_size


def finalize_wan_transition_fields(
    fields: Mapping[str, torch.Tensor],
    *,
    transition_indices: Sequence[int],
    num_steps: int,
) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
    """Drop the global terminal transition consistently from every field.

    Every input field must use layout ``[B, S, ...]`` and cover the same raw
    transition indices.  The last element is sliced only when it is the global
    ``num_steps - 1`` sigma-to-zero transition.  Returned tensors are views.
    """
    indices = tuple(int(i) for i in transition_indices)
    if not indices:
        raise ValueError("Wan rollout produced no transition indices")
    if not fields:
        raise ValueError("Wan rollout produced no transition fields")
    if any(i < 0 or i >= num_steps for i in indices):
        raise ValueError(f"transition indices {indices} fall outside [0, {num_steps})")
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"Wan transition indices must be contiguous, got {indices}")

    raw_steps = len(indices)
    batch_size = None
    checked: dict[str, torch.Tensor] = {}
    for name, tensor in fields.items():
        if not torch.is_tensor(tensor) or tensor.ndim < 2:
            shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor)
            raise ValueError(f"transition field {name!r} must be a [B,S,...] tensor, got {shape}")
        if tensor.shape[1] != raw_steps:
            raise ValueError(f"transition field {name!r} has S={tensor.shape[1]}, expected {raw_steps} for indices {indices}")
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"transition field {name!r} has B={tensor.shape[0]}, expected B={batch_size}")
        checked[name] = tensor

    if indices[-1] == num_steps - 1:
        indices = indices[:-1]
        checked = {name: tensor[:, :-1] for name, tensor in checked.items()}

    if not indices:
        raise ValueError("Wan rollout contains only the final sigma->0 transition; no trainable transition remains")
    expected_steps = len(indices)
    bad = {name: tensor.shape[1] for name, tensor in checked.items() if tensor.shape[1] != expected_steps}
    if bad:  # defensive: keeps future field-specific slicing changes honest
        raise AssertionError(f"Wan transition fields are misaligned after terminal trim: {bad}, expected S={expected_steps}")
    return checked, indices
