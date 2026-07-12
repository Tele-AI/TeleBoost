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
"""P0 regression: the Wan Ulysses head-gather patch must be idempotent at the
class level, so the wan2.2 dual-model path (which calls apply_wan_ulysses_patches
twice on the shared `Head` class) does NOT stack wrappers and double the seq-dim
all-gather under SP>1. See teleboost/patches/ulysses.py:154 and
docs/teleboost_arch_review.md §3 (P0).

Requires `verl` (the patch's inner `from verl.utils.ulysses import ...`); the
training profile validates that dependency before collecting this module.
"""

import pytest


@pytest.mark.training_env
def test_head_gather_wrap_is_idempotent():
    from teleboost.patches.ulysses import patch_diffusion_for_ulysses_head_gather

    class DummyHead:
        def forward(self, *args, **kwargs):
            return kwargs.get("x")

    # First application wraps and stamps the sentinel.
    patch_diffusion_for_ulysses_head_gather(DummyHead)
    first = DummyHead.forward
    assert getattr(first, "_tb_head_wrapped", False) is True

    # Second application (the wan2.2 low+high double-call) must be a no-op:
    # the sentinel returns early, so forward is NOT re-wrapped (no double gather).
    patch_diffusion_for_ulysses_head_gather(DummyHead)
    second = DummyHead.forward
    assert second is first, "sentinel must prevent re-wrapping (double seq-dim gather)"
