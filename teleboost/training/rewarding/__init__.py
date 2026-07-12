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
"""Trainer-driver reward orchestration.

Not reward functions and not reward workers: the pieces of the reward
flow that live on the DRIVER (know ``self.rm_wg``, DataProto, metrics,
advantage precompute).

Modules:

* ``joint_reward_trainer.py``: driver-side logic for
  ``reward.reward_model.type=joint``; calls reward workers, then computes the
  advantage-space joint reward/advantage.
* ``joint_advantage_weights.py``: pure dynamic-weight math for combining
  per-reward advantages.

Reward functions and worker-side model lifecycle live under
``teleboost/reward/execution``.
"""
