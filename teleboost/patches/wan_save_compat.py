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
"""Wan-aware save_checkpoint compatibility shim.

Upstream verl 0.4.0 `FSDPCheckpointManager.save_checkpoint` unconditionally calls
`model_config.save_pretrained(local_path)` to write `config.json` next to the
weights. For HF transformers configs this works; for Wan models, `model.config`
is a `diffusers.configuration_utils.FrozenDict` which has no `save_pretrained`,
and the save crashes with `AttributeError`.

Pre-X3's in-tree verl fork avoided this by commenting out the line. Same idea
here: install a no-op `save_pretrained` on `FrozenDict` so the call short-circuits
without touching upstream code or recipes Workers.

The HF generation_config save path is gated by `unwrap_model.can_generate()`;
Wan22DualModel defines it and the worker pins `can_generate = lambda: False`
onto the loaded upstream WanModel (stock diffusers ModelMixin lacks it).
"""

from __future__ import annotations


def apply() -> None:
    try:
        from diffusers.configuration_utils import FrozenDict  # type: ignore
    except ImportError:
        return

    if hasattr(FrozenDict, "save_pretrained"):
        return

    def _save_pretrained_noop(self, *args, **kwargs):
        # Wan diffusion configs are not HF transformers configs; skip the HF dump.
        return None

    FrozenDict.save_pretrained = _save_pretrained_noop
