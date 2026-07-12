# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Small driver-side helpers for decoded rollout payloads."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "RolloutRuntimeConfig",
    "as_1d_float_tensor",
    "cpu_payload",
    "dataproto_batch_item",
    "dataproto_meta_info",
    "drop_batch_tensor",
    "move_payload",
    "object_array",
    "runtime_device",
    "video_tensor_to_uint8_frames",
]


def drop_batch_tensor(data: Any, key: str) -> bool:
    """Idempotently remove ``key`` from a DataProto-like tensor plane."""
    batch = getattr(data, "batch", None)
    if batch is None or key not in batch.keys():
        return False
    data.pop(batch_keys=[key])
    return True


def object_array(values: Sequence[Any]) -> Any:
    """Return an object-dtype array, falling back to a list in minimal environments."""

    try:
        return np.asarray(list(values), dtype=object)
    except Exception:  # pragma: no cover - numpy is a production dependency
        return list(values)


def as_1d_float_tensor(value: Any, *, key: str, expected_batch: int) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    tensor = tensor.detach().cpu().flatten().float()
    if tensor.numel() != expected_batch:
        raise ValueError(f"{key} has {tensor.numel()} values, expected {expected_batch}")
    return tensor


def dataproto_batch_item(data: Any, key: str) -> Any:
    batch = getattr(data, "batch", None)
    if batch is None:
        raise ValueError("DataProto missing batch")
    try:
        return batch[key]
    except Exception as exc:
        raise ValueError(f"DataProto batch missing required key {key!r}") from exc


def dataproto_meta_info(data: Any) -> Mapping[str, Any]:
    meta = getattr(data, "meta_info", None)
    if meta is None:
        raise ValueError("DataProto missing meta_info")
    return meta


def map_tensors(value: Any, tensor_fn: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, Mapping):
        return {k: map_tensors(v, tensor_fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_tensors(v, tensor_fn) for v in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(v, tensor_fn) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{field.name: map_tensors(getattr(value, field.name), tensor_fn) for field in fields(value)},
        )
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{key: map_tensors(item, tensor_fn) for key, item in vars(value).items()})
    return value


def cpu_payload(value: Any) -> Any:
    return map_tensors(value, lambda tensor: tensor.detach().cpu())


def move_payload(value: Any, device: torch.device | str | None) -> Any:
    if device is None:
        return value
    return map_tensors(value, lambda tensor: tensor.to(device))


def runtime_device(device: torch.device | str | None) -> torch.device | str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def video_tensor_to_uint8_frames(
    video_frames: torch.Tensor,
    *,
    clamp: bool,
) -> np.ndarray:
    """Convert ``(C,T,H,W)`` torch video to NumPy ``(T,H,W,C)`` uint8.

    Conversion to float32 is deliberately before ``numpy()`` because NumPy
    cannot represent torch bfloat16 tensors.
    """
    if video_frames.ndim != 4:
        raise ValueError(f"video_frames must be (C,T,H,W), got {tuple(video_frames.shape)}")
    frames = video_frames.permute(1, 2, 3, 0).float()
    if clamp:
        frames = frames.clamp(0, 1)
    return (frames.cpu().numpy() * 255).astype(np.uint8)


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    """Runtime settings shared by model-specific rollout adapters."""

    group_size: int
    sigma_form: str = "flow_grpo"
    cfg_text_scale: float = 1.0
    cfg_img_scale: float = 1.0
    num_timesteps: int | None = None
    timestep_shift: float | None = None
    ratio_norm: bool = False
