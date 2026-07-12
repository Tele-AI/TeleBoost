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
"""Per-prompt group-relative GRPO advantage.

Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
Open Language Models", arXiv 2402.03300 (Shao et al., 2024), §4.1.2.
Quoted from the paper:

    Â_{i,t} = (r_i − mean(r)) / std(r)
    where r is the group of G outputs sampled from the same question q.

DanceGRPO (arxiv 2505.07818, Eq. 10 + Algorithm 1) applies the same
per-prompt grouping to visual generation: for each prompt
``c ∈ D_b``, generate ``G`` samples with shared initialization noise,
then compute advantages within that group.

**Pre-fix bug.**  The local `compute_advantage` and `joint_reward_trainer.py`'s
per-task advantage code computed mean/std over the **whole batch**,
not per prompt — see the historical TODO comment at
`ray_trainer.py:106` ("TODO: when batchsize not equal to
1") which acknowledged the placeholder.  When ``prompt_batch_size *
n_resp_per_prompt = whole batch`` (i.e. only one prompt per batch)
the whole-batch z-score happens to coincide with the per-prompt one,
masking the bug.  Once you train with prompt_batch_size > 1, the
z-score is taken across **different prompts' rewards mixed
together**, which is not the GRPO algorithm and silently degrades
the policy gradient signal.

**Layout assumption.**  This helper assumes ``rewards`` arrives in
**interleaved per-prompt order** — i.e. samples 0..n-1 belong to
prompt_0, samples n..2n-1 belong to prompt_1, etc.  This is the
contract of ``DataProto.repeat(num_repeat, interleave=True)`` in
DanceGRPO's rollout path (see ``ray_trainer.py`` rollout
side; verl-upstream ``RayPPOTrainer._balance_batch`` is **not**
called in DanceGRPO's ``fit()``, so contiguous prompt-grouping is
preserved end-to-end).  If a future change re-introduces a
load-balancing reshuffle between rollout and advantage compute,
this assumption breaks and the helper must switch to ``uid``-based
grouping (see ``verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage``
which groups by ``index`` instead of relying on layout).
"""

from __future__ import annotations

import torch


def per_prompt_zscore_advantage(
    rewards: torch.Tensor,
    num_repeat: int,
    eps: float = 1e-4,
    min_group_std: float = 0.0,
) -> torch.Tensor:
    """Compute group-relative GRPO advantage with per-prompt grouping.

    Args:
        rewards: 1-D tensor of length ``B = num_prompts * num_repeat``.
            Layout assumption: samples 0..num_repeat-1 are prompt_0,
            num_repeat..2*num_repeat-1 are prompt_1, etc.
        num_repeat: ``n_resp_per_prompt`` from the rollout config.
        eps: numerical guard added to std before division.  Default
            ``1e-4`` matches flow_grpo's ``PerPromptStatTracker``
            (``np.std(...) + 1e-4``): a soft floor that *damps* (does not
            eliminate) advantage blow-up once a group's reward std drops
            below ~``1e-4``.  For a hard cutoff that zeros such groups,
            use ``min_group_std``.  The ``+ eps`` is applied outside any
            finite-positive check (matches the pre-fix ``compute_advantage``
            numerical behaviour).
        min_group_std: groups whose reward std is BELOW this floor get
            zero advantages instead of z-scored ones.  The z-score
            divides the group std out, so a group whose rewards differ
            only by float noise (e.g. a saturated judge with
            within-group std ~1e-3) is re-inflated to unit-scale
            advantages of RANDOM sign — pure gradient noise.  Default
            0.0 keeps the legacy behaviour exactly.

    Returns:
        1-D tensor of advantages, same length as ``rewards``.

    Raises:
        ValueError: if ``rewards.numel() % num_repeat != 0``.
    """
    if num_repeat <= 0:
        raise ValueError(f"num_repeat must be >= 1, got {num_repeat}")
    if num_repeat == 1:
        # Single-sample groups have no within-group variance — Bessel-
        # corrected ``std()`` returns NaN, and the post-hoc ``+ eps``
        # cannot rescue it.  Return zeros: under group-relative GRPO a
        # group-of-one carries no relative signal anyway.  Production
        # configs always set num_repeat >= 2, so this branch is a
        # tripwire for misconfiguration, not a normal path.
        return torch.zeros_like(rewards)
    n_total = rewards.numel()
    if n_total % num_repeat != 0:
        raise ValueError(
            f"rewards length ({n_total}) is not a multiple of num_repeat ({num_repeat}); cannot reshape to (num_groups, num_repeat). This usually means the rollout used a different interleave-repeat factor than what reaches advantage computation, or _balance_batch reshuffled the batch."
        )
    num_groups = n_total // num_repeat
    grouped = rewards.view(num_groups, num_repeat)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std_raw = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - group_mean) / (group_std_raw + eps)
    if min_group_std > 0.0:
        advantages = torch.where(
            group_std_raw < min_group_std,
            torch.zeros_like(advantages),
            advantages,
        )
    return advantages.view(-1)


__all__ = ["per_prompt_zscore_advantage"]
