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
"""Runtime patches that overlay TeleBoost-specific fixes onto upstream verl.

Installed explicitly by TeleBoost runtime entrypoints before they import the
patched verl symbols. Every patch is idempotent.

Scope: verl / vllm / tensordict **drift shims** only — version-pinned
workarounds expected to disappear as upstream moves. The megatron-side
counterpart (``teleboost/engines/teletron/megatron_adaptor.py``)
is a permanent extension mechanism and deliberately lives elsewhere;
see ``teleboost/engines/teletron/__init__`` for the distinction.
"""

_APPLIED = False


def _verl_available() -> bool:
    """Gate for the whole patch layer: no verl ⇒ nothing to patch (pure-CPU
    environments legitimately run only the algorithm layer + its tests).
    When verl IS present, the patch bodies import their exact targets WITHOUT
    guards — an ImportError/AttributeError there means the upstream layout
    drifted under our 0.7.1 pin and must fail LOUDLY at startup, not silently
    skip a correctness patch."""
    try:
        import verl
    except ImportError:
        return False
    # A few CPU contract tests install a minimal ``verl.utils.ulysses`` shim.
    # Such a module is intentionally not the runtime package and has no
    # package search path.  Treat it like an absent dependency; otherwise the
    # patch layer tries to import unrelated verl managers from the shim.
    return hasattr(verl, "__path__")


def apply(*, require_verl: bool = False) -> bool:
    """Apply every TeleBoost patch over upstream verl. Idempotent.

    Returns whether the runtime patch set is installed.  A dependency-light
    caller may probe with ``require_verl=False``; importantly, an absent verl
    package does not poison the idempotency sentinel and a later call can
    still install the patches.
    """
    global _APPLIED
    if _APPLIED:
        return True
    if not _verl_available():
        if require_verl:
            raise RuntimeError("TeleBoost runtime patches require the pinned verl package; install the version in constraints/upstreams/verl.txt first.")
        return False

    # Keep importing ``teleboost`` dependency-light when verl is absent.  The
    # patch modules intentionally import their exact upstream targets so API
    # drift fails loudly, but that should happen only in a real verl runtime,
    # not while using TeleBoost's standalone algorithm/data utilities.
    from teleboost.patches.dataset_compat import apply as _apply_dataset_compat
    from teleboost.patches.diffusion_async_bypass import apply as _apply_async_bypass
    from teleboost.patches.tensordict_compat import apply as _apply_tensordict_compat
    from teleboost.patches.ulysses import apply_api as _apply_wan_ulysses
    from teleboost.patches.ulysses import apply_cp_grad_fix as _apply_cp_fix
    from teleboost.patches.wan_save_compat import apply as _apply_wan_save_compat
    from teleboost.patches.wan_weight_saver import apply as _apply_wan_weight_saver

    _apply_tensordict_compat()  # run first — DataProto mutations come early.
    _apply_dataset_compat()
    _apply_async_bypass()
    _apply_cp_fix()
    _apply_wan_save_compat()
    _apply_wan_ulysses()
    _apply_wan_weight_saver()  # DPO: Wan arch no-op weight saver for MegatronCheckpointManager init.
    _APPLIED = True
    return True
