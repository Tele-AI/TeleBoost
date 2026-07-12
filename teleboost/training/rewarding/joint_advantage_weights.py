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
"""Joint reward advantage-weight helpers.

The convex weight scheme implemented here is not part of the BGPO paper
(arxiv 2511.18919). It is an in-house rule used only by
``JointRewardMixin`` to compute per-sample reward weights from per-reward
advantages (aesthetic + RAFT + VideoCLIP-XL + VideoPhy, etc.).
"""

from __future__ import annotations

import numpy as np
import torch


def compute_joint_advantage_weights(advantages: torch.Tensor) -> torch.Tensor:
    """Compute per-sample convex weights from multi-reward advantages.

    Each row of ``advantages`` (shape ``(B, K)`` for ``B`` samples and
    ``K`` reward heads) is one sample's per-reward advantage vector. The
    function returns a ``(B, K)`` matrix where each row is a convex
    weight vector (non-negative entries that sum to 1).

    Per-row rule:

    * Mixed signs (``min(a) <= 0 <= max(a)``): pick the argmin/argmax
      pair and choose the zero-bracketing convex combination.
    * All positive: full weight on the least enthusiastic head
      (``argmin``).
    * All negative: full weight on the least negative head (``argmax``).
    """
    if advantages.numel() == 0:
        return torch.zeros_like(advantages)
    if advantages.dim() != 2:
        raise ValueError(f"advantages must be 2D, got shape {tuple(advantages.shape)}")

    weights = torch.zeros_like(advantages)
    for i in range(advantages.shape[0]):
        a = advantages[i].detach().cpu().numpy().astype(np.float32)
        n = int(a.shape[0])
        if n == 0:
            continue

        if a.min() <= 0 <= a.max():
            idx_lo, idx_hi = int(np.argmin(a)), int(np.argmax(a))
            if np.isclose(a[idx_hi], a[idx_lo]):
                c = np.ones(n, dtype=np.float32) / n
            else:
                t = -a[idx_lo] / (a[idx_hi] - a[idx_lo])
                c = np.zeros(n, dtype=np.float32)
                c[idx_lo] = 1.0 - t
                c[idx_hi] = t
        elif a.min() > 0:
            c = np.zeros(n, dtype=np.float32)
            c[int(np.argmin(a))] = 1.0
        else:
            c = np.zeros(n, dtype=np.float32)
            c[int(np.argmax(a))] = 1.0

        weights[i] = torch.from_numpy(c).to(device=advantages.device, dtype=advantages.dtype)

    return weights


__all__ = ["compute_joint_advantage_weights"]
