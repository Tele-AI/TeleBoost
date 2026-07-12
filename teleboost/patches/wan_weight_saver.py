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
"""Register a no-op Wan weight saver in verl's checkpoint registry.

verl's ``MegatronCheckpointManager.__init__`` (called from
``MegatronEngine.initialize()`` at line 349) looks up
``get_weight_saver(self.arch)`` to install a weight-export function
for HF-format checkpoint dumps + vLLM rollout weight sync. Wan video
diffusion has neither path:

* HF-format dumps don't make sense (Wan isn't HF-registered;
  ``teleboost-convert-wan-to-teletron`` handles the Wan↔teletron
  conversion offline).
* DPO is offline preference-pair training — no actor↔rollout weight
  sync, so the saver is never actually called.

But the **lookup** has to succeed. verl's ``get_weight_saver`` rebuilds
an in-function dict on every call (no module-level register API), so
we wrap the function and pre-empt for the Wan arch.

Patch surface
-------------
``verl.utils.checkpoint.megatron_checkpoint_manager`` does
``from verl.models.weight_loader_registry import get_weight_saver``
at module-load time, capturing the original function reference. We
have to patch BOTH the registry module's attribute AND the captured
binding in ``megatron_checkpoint_manager``; patching only the registry
would leave the ckpt manager's import-time reference dangling.
"""

from __future__ import annotations


def _noop_wan_weight_saver(*args, **kwargs):
    """No-op weight saver for Wan video diffusion.

    Called by verl's ``MegatronCheckpointManager`` if it ever tries
    to export model weights to HF safetensors (e.g. rollout sync).
    The DPO recipes never triggers that path — but raising here would
    mask real misconfigurations, so we silently return None to let any
    accidental caller continue. (If a caller depends on the return
    value being a non-trivial state-dict iterator, surface that as a
    real bug rather than papering over it.)
    """
    return None


def apply() -> None:
    """Wrap verl's ``get_weight_saver`` to handle the Wan architecture.

    Idempotent — re-applying is a no-op because we check whether the
    wrapped function is already in place.
    """
    import importlib.util

    # This patch only serves the megatron/DPO backend: verl's
    # MegatronCheckpointManager is the sole consumer of get_weight_saver,
    # and importing verl.utils.checkpoint.megatron_checkpoint_manager pulls
    # megatron at module level. On FSDP-only installs (verl without
    # megatron) there is nothing to patch — same gate contract as
    # teleboost/__init__.
    if importlib.util.find_spec("megatron") is None:
        return

    import verl.models.weight_loader_registry as _registry
    import verl.utils.checkpoint.megatron_checkpoint_manager as _cm

    from teleboost.models import WAN_ARCH

    _SENTINEL_ATTR = "_teleboost_wan_aware"
    if getattr(_registry.get_weight_saver, _SENTINEL_ATTR, False):
        return

    _orig_get_weight_saver = _registry.get_weight_saver

    def _wan_aware_get_weight_saver(arch: str):
        if arch == WAN_ARCH:
            return _noop_wan_weight_saver
        return _orig_get_weight_saver(arch)

    _wan_aware_get_weight_saver._teleboost_wan_aware = True

    _registry.get_weight_saver = _wan_aware_get_weight_saver
    _cm.get_weight_saver = _wan_aware_get_weight_saver
