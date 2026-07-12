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
"""TeleBoost: video-generation post-training stack on top of upstream verl.

Package import is deliberately side-effect free. Runtime entrypoints call
``apply_runtime_patches()`` before importing patched verl symbols; importing a
utility such as ``teleboost.models`` must not mutate process-global upstream
classes. The Megatron/TeleTron adapter remains a separate explicit lifecycle
owned by the DPO entrypoint.
"""


def apply_runtime_patches(*, require_verl: bool = True) -> bool:
    """Install TeleBoost's idempotent verl compatibility layer explicitly."""
    from teleboost import patches

    return patches.apply(require_verl=require_verl)
