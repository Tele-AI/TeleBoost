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
"""Test uid broadcast logic (fix for C9).

Mirrors the broadcast that ``ray_trainer.fit`` does on
``gen_batch_output.non_tensor_batch["uid"]``: generate one fresh UUID
per prompt, then ``np.repeat(uids, n_resp, axis=0)`` so each sample
in the same prompt's n_resp rollouts shares the prompt's uid.
"""

from __future__ import annotations

import numpy as np


def _broadcast_prompt_uids(prompt_uids: np.ndarray, n_resp: int) -> np.ndarray:
    """Same call ray_trainer.fit makes."""
    return np.repeat(prompt_uids, n_resp, axis=0)


def test_broadcast_length_correct():
    prompt_uids = np.array(["u0", "u1", "u2"], dtype=object)
    n = 4
    out = _broadcast_prompt_uids(prompt_uids, n)
    assert len(out) == 3 * 4


def test_broadcast_each_prompt_n_consecutive_samples():
    prompt_uids = np.array(["u0", "u1", "u2"], dtype=object)
    n = 4
    out = _broadcast_prompt_uids(prompt_uids, n)
    # samples 0..3 are prompt 0, 4..7 are prompt 1, 8..11 are prompt 2
    assert all(out[i] == "u0" for i in range(0, 4))
    assert all(out[i] == "u1" for i in range(4, 8))
    assert all(out[i] == "u2" for i in range(8, 12))


def test_broadcast_matches_repeat_interleave_layout():
    """The broadcast layout must match the rollout's DataProto.repeat(
    interleave=True) layout — which is also np.repeat axis=0.
    """
    prompt_uids = np.array(["u0", "u1"], dtype=object)
    n = 3
    out = _broadcast_prompt_uids(prompt_uids, n)
    # interleave layout: u0,u0,u0,u1,u1,u1
    expected = np.array(["u0", "u0", "u0", "u1", "u1", "u1"], dtype=object)
    np.testing.assert_array_equal(out, expected)


def test_broadcast_n_resp_one_is_identity():
    """n_resp=1 → broadcast is the identity (each prompt = one sample)."""
    prompt_uids = np.array(["u0", "u1", "u2", "u3"], dtype=object)
    out = _broadcast_prompt_uids(prompt_uids, 1)
    np.testing.assert_array_equal(out, prompt_uids)


def test_broadcast_uniqueness_across_prompts():
    """Distinct prompt UUIDs stay distinct after broadcast."""
    import uuid

    n_prompts = 5
    prompt_uids = np.array([str(uuid.uuid4()) for _ in range(n_prompts)], dtype=object)
    n = 3
    out = _broadcast_prompt_uids(prompt_uids, n)
    # n_prompts * n samples, n_prompts distinct uids
    assert len(set(out.tolist())) == n_prompts
