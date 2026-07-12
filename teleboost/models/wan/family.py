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
"""Wan-family facts — TeleBoost's checked mirror of upstream Wan configs.

Upstream truth lives in the installed ``wan.configs`` modules
(``t5_tokenizer``, ``patch_size``, ``vae_stride`` per checkpoint) and on the
loaded ``WanModel`` instance itself (``model.patch_size``). This module mirrors
the values the TeleBoost layers need so each fact is written once on our side —
they used to be copy-pasted per call site (three seq-len formulas, a hardcoded
tokenizer subdir that ignored its own config key, a bare ``16``).
``fsdp_worker`` asserts the loaded model's ``patch_size`` against this mirror
at build time, so upstream drift fails loudly instead of silently diverging.

Scope (per the adversarial design review): ``PATCH_SIZE`` and
``TOKENIZER_SUBPATH`` are invariant across every checkpoint this tree supports
(wan21 dense + wan22 dual-A14B). ``VAE_STRIDE`` / ``LATENT_CHANNELS`` are the
*defaults for the config keys* ``actor_rollout_ref.vae_stride`` /
``latent_channels`` — they describe the Wan2.1-VAE lineage, NOT the whole Wan
family (Wan2.2 TI2V-5B ships a 48-channel ``(4, 16, 16)`` VAE); runtime truth
for those two is the config.
"""

from __future__ import annotations
import math
import os
import torch


#: DiT patchification (t, h, w) — invariant across supported checkpoints.
PATCH_SIZE = (1, 2, 2)

#: Wan2.1-VAE lineage defaults; ``actor_rollout_ref.{vae_stride,latent_channels}`` override.
VAE_STRIDE = (4, 8, 8)
LATENT_CHANNELS = 16

#: T5 tokenizer subdir inside a Wan HF checkpoint dir;
#: ``actor_rollout_ref.tokenizer_subpath`` overrides.
TOKENIZER_SUBPATH = "google/umt5-xxl"


def wan_seq_len(t: int, h: int, w: int) -> int:
    """Token sequence length of a (T, H, W) latent under ``PATCH_SIZE``.

    The one copy of the formula shared by the actor recompute and both rollout
    loops (callers pass dims explicitly — their tensors differ in rank).
    """
    return math.ceil((h * w) / (PATCH_SIZE[1] * PATCH_SIZE[2]) * t)


def resolve_wan22_dual_paths(model_config, local_path: str) -> tuple[str, str]:
    """Resolve (high_noise_path, low_noise_path): both keys together, or both derived from local_path."""
    high = model_config.get("high_noise_path", None) or None
    low = model_config.get("low_noise_path", None) or None
    if (high is None) != (low is None):
        raise ValueError(f"wan_version=wan22 requires model.high_noise_path and model.low_noise_path to be set together (got high_noise_path={high!r}, low_noise_path={low!r})")
    if high is None:
        high = os.path.join(local_path, "high_noise_model")
        low = os.path.join(local_path, "low_noise_model")
        if not os.path.exists(high):
            raise ValueError("wan_version=wan22 requires model.high_noise_path/model.low_noise_path or existing low_noise_model and high_noise_model near model.path")
    return high, low


# =========================================================================
# wan2.2 dual-model selectors (were verbatim copies in rollout AND actor;
# the GRPO ratio needs both sides on the SAME per-step scale, so ONE home)
# =========================================================================

"""wan2.2 dual-model guide-scale selection — the ONE shared copy.

The wan2.2 T2V pipeline pairs a high-noise and a low-noise transformer split
at a sigma boundary, and ships a (low, high) guide-scale pair selected per
step by that same boundary. Rollout sampling, the actor's log-prob recompute,
and the dual-model wrapper must all pick the SAME scale for the same step, or
the GRPO ratio compares two different effective policies — so the selector
lives here and both sides import it (they used to carry verbatim copies).

The boundary constant itself is the single config key
``actor_rollout_ref.wan22_boundary`` (propagated to the actor sub-config by
``fsdp_worker``'s whitelist).
"""


def normalize_wan22_timestep(t, sigma):
    if sigma is not None:
        if torch.is_tensor(sigma):
            return sigma.detach().flatten()[0].float().item()
        return float(sigma)
    if t is None:
        return None
    if torch.is_tensor(t):
        t_val = t.detach().flatten()[0].float().item()
    else:
        t_val = float(t)
    if t_val > 1.0:
        t_val = t_val / 1000.0
    return t_val


def select_wan22_guide_scale(guide_scale, t, sigma, boundary):
    if isinstance(guide_scale, list | tuple) and len(guide_scale) >= 2:
        t_val = normalize_wan22_timestep(t, sigma)
        if t_val is None:
            return guide_scale[0]
        return guide_scale[1] if t_val >= boundary else guide_scale[0]
    return guide_scale


def should_skip_uncond(skip_uncond, sample_guide_scale) -> bool:
    """Return true only when a caller opted in and resolved CFG scale is exactly 1."""
    if not skip_uncond:
        return False
    if torch.is_tensor(sample_guide_scale):
        return bool((sample_guide_scale.detach().float() == 1.0).all())
    return float(sample_guide_scale) == 1.0
