# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""FSDP-managed execution for model-specific call sites."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

__all__ = [
    "ensure_fsdp_forward_dispatch",
    "fsdp_managed_call",
    "is_fsdp_module",
]


_CALLABLE_KWARG = "__teleboost_fsdp_callable"
_ARGS_KWARG = "__teleboost_fsdp_args"
_KWARGS_KWARG = "__teleboost_fsdp_kwargs"
_INSTALLED_ATTR = "_teleboost_fsdp_dispatch_installed"
_ORIGINAL_FORWARD_ATTR = "_teleboost_original_forward"


def is_fsdp_module(module: Any) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:  # pragma: no cover - minimal environments
        return False
    return isinstance(module, FSDP)


def ensure_fsdp_forward_dispatch(module: Any) -> Any:
    """Install the managed-call branch on an FSDP module's forward boundary."""

    if module is None or getattr(module, _INSTALLED_ATTR, False):
        return module
    original_forward = module.forward

    def _forward_with_teleboost_dispatch(self, *args: Any, **kwargs: Any):
        fn = kwargs.pop(_CALLABLE_KWARG, None)
        if fn is None:
            return original_forward(*args, **kwargs)
        op_args = kwargs.pop(_ARGS_KWARG, ())
        op_kwargs = kwargs.pop(_KWARGS_KWARG, {})
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected TeleBoost FSDP dispatch kwargs: {unexpected}")
        return fn(self, *op_args, **op_kwargs)

    module.__dict__[_ORIGINAL_FORWARD_ATTR] = original_forward
    module.forward = MethodType(_forward_with_teleboost_dispatch, module)
    module.__dict__[_INSTALLED_ATTR] = True
    return module


def fsdp_managed_call(module: Any, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn(module, *args, **kwargs)`` through the FSDP forward boundary."""

    if is_fsdp_module(module):
        return module(
            **{
                _CALLABLE_KWARG: fn,
                _ARGS_KWARG: args,
                _KWARGS_KWARG: kwargs,
            }
        )
    return fn(module, *args, **kwargs)


def sum_all_reduce_grads(parameters, group) -> int:
    """SUM-all-reduce existing grads over ``group``; returns the tensor count."""

    import torch.distributed as dist

    reduced = 0
    for param in parameters:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
            reduced += 1
    return reduced


def cp_gradient_contract(train_module: Any, cp_group: Any) -> tuple[float, bool]:
    """Resolve the CP gradient-normalization contract for one update phase.

    CP ranks hold sequence shards whose per-rank gradients SUM to the true
    gradient. Two mechanically different but numerically equal realizations:

    - FSDP-wrapped module: FSDP's world-spanning reduce MEANS over dp*cp
      ranks, so the caller pre-scales the loss by cp_size and must NOT run
      any explicit reduction. Returns ``(cp_size, True)``.
    - Plain module (single-rank dev escape hatch): the caller runs
      ``sum_all_reduce_grads`` over the cp group after backward, loss
      unscaled. Returns ``(1.0, False)``.
    """

    if cp_group is None:
        return 1.0, False
    import torch.distributed as dist

    cp_size = float(dist.get_world_size(cp_group))
    if is_fsdp_module(train_module):
        return cp_size, True
    return 1.0, False
