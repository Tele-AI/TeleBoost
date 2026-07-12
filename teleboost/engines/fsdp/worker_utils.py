# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light helpers shared by FSDP family workers."""

from __future__ import annotations

from typing import Any, Iterable

from teleboost.config.access import select

__all__ = [
    "actor_config",
    "actor_strategy",
    "backend_value",
    "build_identity_sharding_manager",
    "build_optimizer",
    "build_unit_lr_scheduler",
    "family_select",
    "fsdp_enabled_for_family",
    "require_single_rank_when_fsdp_off",
]


def backend_value(config: Any) -> str:
    for key in ("backend.name", "actor_rollout_ref.type", "trainer.type", "data.type"):
        value = select(config, key, None)
        if value is not None:
            return str(value).strip().lower()
    return ""


def family_select(config: Any, family: str, key: str, default: Any = None) -> Any:
    value = select(config, f"{family}.{key}", None)
    if value is None:
        value = select(config, f"actor_rollout_ref.{family}.{key}", None)
    return default if value is None else value


def actor_config(config: Any) -> Any:
    root_actor = select(config, "actor", None)
    return root_actor if root_actor is not None else select(config, "actor_rollout_ref.actor", None)


def actor_strategy(config: Any) -> str:
    return str(select(config, "actor.strategy", select(config, "actor_rollout_ref.actor.strategy", "")) or "").strip().lower()


def fsdp_enabled_for_family(config: Any, family: str, *, default_from_strategy: bool = False) -> bool:
    explicit = family_select(config, family, "fsdp.enable", None)
    if explicit is None:
        return actor_strategy(config) == "fsdp" if default_from_strategy else True
    if isinstance(explicit, str):
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return bool(explicit)


def require_single_rank_when_fsdp_off(config: Any, *, family: str, exc_type: type[Exception]) -> None:
    if actor_strategy(config) != "fsdp":
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        raise exc_type(f"actor.strategy=fsdp with {family}.fsdp.enable=false on world_size>1: data-parallel ranks would train unsynchronized copies. Drop the explicit disable or run single-rank.")


def build_optimizer(parameters: Iterable[Any], config: Any) -> Any | None:
    import torch

    optim_config = select(config, "actor.optim", select(config, "actor_rollout_ref.actor.optim", None))
    lr = float(select(optim_config, "lr", select(config, "actor_rollout_ref.actor.lr", 1e-6)))
    weight_decay = float(select(optim_config, "weight_decay", 0.0))
    betas = tuple(select(optim_config, "betas", (0.9, 0.999)))
    return torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)


def build_unit_lr_scheduler(optimizer: Any) -> Any | None:
    if optimizer is None:
        return None
    import torch

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def build_identity_sharding_manager() -> Any:
    from teleboost.engines.fsdp.sharding.identity import IdentityShardingManager

    return IdentityShardingManager()
