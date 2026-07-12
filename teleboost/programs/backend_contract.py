# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contract implemented by model-family backends.

This module is intentionally limited to :mod:`typing`.  In particular, the
public plugin contract must be importable without importing a training stack,
model implementation, or accelerator extension.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class BackendSpec(Protocol):
    """Construction hooks required from every model-family backend."""

    name: str

    def validate_capabilities(self, config: Any) -> None:
        """Validate model-family runtime and algorithm capabilities."""

    def validate_reward(self, config: Any) -> None:
        """Validate reward types and adapters supported by the family."""

    def prepare_tokenizer(self, config: Any) -> tuple[Any, Any]:
        """Construct the tokenizer and processor required by the trainer."""

    def resolve_worker_and_group(self, config: Any) -> tuple[type[Any], type[Any]]:
        """Resolve the worker-group and actor-worker classes."""

    def register_reward_workers(
        self,
        config: Any,
        role_worker_mapping: dict[Any, Any],
        mapping: dict[Any, Any],
        global_pool_id: str,
    ) -> None:
        """Register or delegate the backend's reward execution path."""

    def collate_fn(self, config: Any) -> Callable[..., Any]:
        """Return the backend's dataset collation function."""

    def trainer_cls(self, config: Any) -> type[Any]:
        """Return the trainer class for the validated algorithm selection."""


BackendFactory: TypeAlias = Callable[[], BackendSpec]

__all__ = ["BackendFactory", "BackendSpec"]
