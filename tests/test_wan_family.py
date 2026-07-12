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
"""wan_family: the checked mirror of Wan model facts.

``wan_seq_len`` replaced three verbatim copies of the token-count formula
(actor recompute + both rollout loops); this pins it to the exact original
expression so the consolidation stays byte-identical.
"""

import math

import pytest

from teleboost.models.wan.family import (
    LATENT_CHANNELS,
    PATCH_SIZE,
    TOKENIZER_SUBPATH,
    VAE_STRIDE,
    resolve_wan22_dual_paths,
    wan_seq_len,
)


def _original_formula(t, h, w):
    # verbatim from the pre-consolidation call sites
    patch_size = [1, 2, 2]
    return math.ceil((h * w) / (patch_size[1] * patch_size[2]) * t)


def test_seq_len_matches_all_three_original_copies():
    # grid covers smoke (60x104), 480p (60x104x13), 720p-ish, and odd dims
    for t in (1, 7, 13, 21):
        for h in (30, 60, 61, 90):
            for w in (52, 104, 105, 160):
                assert wan_seq_len(t, h, w) == _original_formula(t, h, w), (t, h, w)


def test_seq_len_ceil_wraps_whole_product():
    # the ceil is over the FULL product (h*w/4*t), not per-frame:
    # odd h*w makes the two orderings differ — pin the original semantics.
    assert wan_seq_len(3, 61, 105) == math.ceil((61 * 105) / 4 * 3)
    assert wan_seq_len(3, 61, 105) != math.ceil((61 * 105) / 4) * 3


def test_mirror_constants():
    assert PATCH_SIZE == (1, 2, 2)
    assert VAE_STRIDE == (4, 8, 8)
    assert LATENT_CHANNELS == 16
    assert TOKENIZER_SUBPATH == "google/umt5-xxl"


def test_wan22_paths_both_explicit_returned_verbatim():
    cfg = {"high_noise_path": "/ckpt/high", "low_noise_path": "/ckpt/low"}
    assert resolve_wan22_dual_paths(cfg, "/unused") == ("/ckpt/high", "/ckpt/low")


@pytest.mark.parametrize("present", ["high_noise_path", "low_noise_path"])
def test_wan22_paths_single_key_fails_fast(present):
    with pytest.raises(ValueError, match="set together"):
        resolve_wan22_dual_paths({present: "/ckpt/one"}, "/unused")


def test_wan22_paths_derived_from_local_path(tmp_path):
    (tmp_path / "high_noise_model").mkdir()
    (tmp_path / "low_noise_model").mkdir()
    high, low = resolve_wan22_dual_paths({}, str(tmp_path))
    assert high == str(tmp_path / "high_noise_model")
    assert low == str(tmp_path / "low_noise_model")


def test_wan22_paths_missing_subdirs_raise(tmp_path):
    with pytest.raises(ValueError, match="near model.path"):
        resolve_wan22_dual_paths({}, str(tmp_path))


def test_wan22_paths_empty_string_treated_as_unset(tmp_path):
    (tmp_path / "high_noise_model").mkdir()
    (tmp_path / "low_noise_model").mkdir()
    cfg = {"high_noise_path": "", "low_noise_path": ""}
    assert resolve_wan22_dual_paths(cfg, str(tmp_path))[0] == str(tmp_path / "high_noise_model")
