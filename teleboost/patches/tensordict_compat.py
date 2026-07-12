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
"""Globally bypass ``tensordict>=0.10`` lock checks.

Upstream ``verl.protocol.DataProto.pop()`` and assorted recipes call
sites freely mutate ``self.batch`` (assignment, deletion, pop) on
TDs that arrive locked through Ray serialization. ``tensordict`` 0.10
made these mutations raise ``RuntimeError(_LOCK_ERROR)``.

Patch the class-level ``_set_str`` to force ``ignore_lock=True``
unconditionally. Single-writer semantics in verl already ensure
safety, so the lock check is just noise here. Idempotent.

Applied explicitly at runtime startup via ``teleboost.patches.apply()``.
"""

from __future__ import annotations


def apply() -> None:
    # Unguarded on purpose: the caller (patches/__init__.apply) only runs this
    # when verl is installed, so a failure here is tensordict layout drift and
    # must be loud — a silently skipped lock bypass surfaces later as cryptic
    # "TensorDict is locked" errors mid-training.
    from tensordict import TensorDictBase as _TDB
    from tensordict._td import TensorDict as _TD

    if getattr(_TDB, "_teleboost_lock_bypassed", False):
        return

    def _noop_lock(self, *a, **kw):
        return self

    _TDB.lock_ = _noop_lock
    _TDB.lock = _noop_lock

    _orig_set_str = _TD._set_str

    def _patched_set_str(self, key, value, *, inplace=False, validated=False, non_blocking=False, ignore_lock=False):
        return _orig_set_str(
            self,
            key,
            value,
            inplace=inplace,
            validated=validated,
            non_blocking=non_blocking,
            ignore_lock=True,
        )

    _TD._set_str = _patched_set_str
    _TDB._teleboost_lock_bypassed = True
