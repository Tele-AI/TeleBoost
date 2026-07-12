# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Neutral actor-update utilities shared by family-specific actors."""

from __future__ import annotations

from typing import Any, Iterable

import torch

__all__ = [
    "aggregate_numeric_metrics",
    "freeze_module",
    "iter_train_parameters",
    "module_dtype",
    "module_device_or_none",
    "train_module",
]


def train_module(module: Any) -> Any:
    transformer = getattr(module, "transformer", None)
    return transformer if transformer is not None else module


def iter_train_parameters(module: Any) -> Iterable[Any]:
    parameters = getattr(train_module(module), "parameters", None)
    if not callable(parameters):
        return iter(())
    return parameters()


def module_dtype(module: Any, *, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    try:
        return next(iter_train_parameters(module)).dtype
    except (AttributeError, StopIteration):
        return default


def module_device_or_none(module: Any, *, cpu_is_none: bool = False) -> torch.device | None:
    try:
        param = next(iter_train_parameters(module))
    except (AttributeError, StopIteration):
        return None
    device = param.device
    if cpu_is_none and device.type == "cpu":
        return None
    return device


def freeze_module(module: Any) -> Any:
    if module is None:
        return None
    if hasattr(module, "eval"):
        module.eval()
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for param in parameters():
            param.requires_grad_(False)
    return module


def aggregate_numeric_metrics(
    metric_dicts: list[dict],
    *,
    joint_loss: float,
    extra: dict[str, Any] | None = None,
) -> dict:
    merged: dict = {}
    sums: dict[str, list[float]] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                merged.setdefault(key, value)
                continue
            sums.setdefault(key, []).append(float(value))
    for key, values in sums.items():
        if key.endswith("ratio_max"):
            merged[key] = max(values)
        elif key.endswith("ratio_min"):
            merged[key] = min(values)
        else:
            merged[key] = sum(values) / len(values)
    merged["joint/loss"] = joint_loss
    if extra:
        merged.update(extra)
    return merged


class UpdateInstrumentation:
    """No-op observation hooks for the micro-batched update loop.

    Families override what they need (memory tracing, OOM summaries, one-shot
    profiler stepping, allocator defrag). Hooks observe — they must not mutate
    gradients or losses.
    """

    def on_entry(self) -> None: ...

    def pre_backward(self, micro_idx: int) -> None: ...

    def post_backward(self, micro_idx: int) -> None: ...

    def after_micro(self, micro_idx: int) -> None: ...

    def on_oom(self, micro_idx: int) -> None: ...

    def on_finish(self) -> None: ...


def run_micro_update(
    *,
    optimizer: Any,
    train_module: Any,
    micro_losses: Any,
    aggregate: Any,
    cp_group: Any = None,
    max_grad_norm: Any = None,
    reduce_parameters_fn: Any = None,
    clip_parameters_fn: Any = None,
    instrumentation: UpdateInstrumentation | None = None,
    pre_step_hook: Any = None,
) -> dict:
    """The shared micro-batched FSDP update loop for prompt families.

    One backward per (sample, timestep) micro (the FSDP layer-reuse gradient
    constraint documented in the family dataproto modules), CP normalization
    via engines.fsdp.execution.cp_gradient_contract, FSDP-aware gradient
    clipping, and a single optimizer step.
    """

    import torch

    from teleboost.engines.fsdp.execution import (
        cp_gradient_contract,
        is_fsdp_module,
        sum_all_reduce_grads,
    )

    hooks = instrumentation or UpdateInstrumentation()
    hooks.on_entry()

    cp_scale, fsdp_reduces_cp = cp_gradient_contract(train_module, cp_group)

    joint_loss = 0.0
    micro_metrics: list[dict] = []
    try:
        for micro_idx, (scaled_loss, metrics) in enumerate(micro_losses):
            scaled_loss = scaled_loss * cp_scale  # 1.0 unless the FSDP world-reduce spans cp
            hooks.pre_backward(micro_idx)
            try:
                scaled_loss.backward()
            except torch.OutOfMemoryError:
                hooks.on_oom(micro_idx)
                raise
            hooks.post_backward(micro_idx)
            joint_loss += float(scaled_loss.detach()) / cp_scale
            micro_metrics.append(metrics)
            hooks.after_micro(micro_idx)
    finally:
        hooks.on_finish()

    cp_grad_reduced = 0
    if cp_group is not None and not fsdp_reduces_cp:
        parameters = reduce_parameters_fn() if reduce_parameters_fn is not None else iter_train_parameters(train_module)
        cp_grad_reduced = sum_all_reduce_grads(parameters, cp_group)

    grad_norm = None
    if max_grad_norm is not None:
        if is_fsdp_module(train_module):
            # FSDP shards gradients: the vanilla utility would clip against a
            # rank-LOCAL shard norm; FSDP's method all-reduces the global norm
            # first (symmetric collective at this same point on every rank).
            grad_norm = train_module.clip_grad_norm_(float(max_grad_norm))
        else:
            parameters = clip_parameters_fn() if clip_parameters_fn is not None else iter_train_parameters(train_module)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))

    if pre_step_hook is not None:
        pre_step_hook()
    optimizer.step()
    optimizer.zero_grad()

    out = aggregate(micro_metrics, joint_loss)
    if grad_norm is not None:
        out["optim/grad_norm"] = float(torch.as_tensor(grad_norm).detach().float().cpu())
    if cp_grad_reduced:
        out["optim/cp_grad_reduce_tensors"] = cp_grad_reduced
    out["optim/stepped"] = True
    return out
