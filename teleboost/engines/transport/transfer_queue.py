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
"""TransferQueue bootstrap for TeleBoost.

Phase 2 of the v0.7.1 upgrade introduces TransferQueue
(https://github.com/Ascend/TransferQueue) so generated video latents and
trajectory tensors flow peer-to-peer between rollout / reward / actor
workers instead of going through the Ray driver via ``DataProto`` pickle.
For large video frames this turns a many-hop ``ray.get`` into a single
zero-copy pull.

This module owns the one-time bootstrap: a ``TransferQueueController``
Ray actor + each worker grabbing a client handle. Producers and consumers
that move bulk tensors are wired up incrementally in follow-up commits;
this file is intentionally side-effect-free until ``bootstrap()`` is
explicitly called.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Initialize TransferQueue once per training run.

    Idempotent: subsequent calls are no-ops. Safe to call from the driver
    after ``ray.init()`` — TransferQueue itself uses Ray actors for the
    controller and storage. If the ``transfer_queue`` package isn't
    importable (e.g. user opted out of Phase 2 deps) this logs a warning
    and skips; the rollout / reward path still works through the
    DataProto channel.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    try:
        import transfer_queue as _tq
    except ImportError:
        logger.warning("TransferQueue not installed; falling back to DataProto for rollout→reward→actor data movement. `pip install TransferQueue` to enable the Phase 2 fast path.")
        return
    try:
        _tq.init()
    except Exception as e:
        logger.warning("TransferQueue init failed (%s); falling back.", e)
        return
    _BOOTSTRAPPED = True
    logger.info("TransferQueue bootstrapped (backend = SimpleStorage default)")


# ---------------------------------------------------------------------------
# Cross-process helpers: the ONE home for the gate, the partition id, and the
# video-frames kv schema. Rollout workers put; the trainer driver gets. Both
# read TQ state from env vars propagated through Ray runtime_env (see
# ``main_teleboost._init_ray``), so every actor sees the same toggle.
# ---------------------------------------------------------------------------

_DEFAULT_PARTITION = "teleboost.rollout.video_frames"


def enabled() -> bool:
    """Upstream-verl-aligned gate: TRANSFER_QUEUE_ENABLE env var."""
    return os.environ.get("TRANSFER_QUEUE_ENABLE") == "1"


def partition_id() -> str:
    return os.environ.get("TELEBOOST_TQ_PARTITION", _DEFAULT_PARTITION)


def put_video_frames(frames) -> list:
    """kv_put one video tensor per key; returns the generated keys.

    Raises on any failure — callers keep their own degrade-to-DataProto
    try/except so the fallback policy stays at the call site.
    """
    import uuid

    import transfer_queue as _tq

    keys = [str(uuid.uuid4()) for _ in range(len(frames))]
    for i, k in enumerate(keys):
        # TransferQueue's public API is module-level kv_put(key, partition_id, fields).
        _tq.kv_put(k, partition_id=partition_id(), fields={"frames": frames[i]})
    return keys


def get_video_frames(keys):
    """kv_batch_get the frames for ``keys``; returns the (B, *frame) tensor."""
    import transfer_queue as _tq

    td = _tq.kv_batch_get(list(keys), partition_id=partition_id(), select_fields=["frames"])
    return td["frames"]
