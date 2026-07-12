# Copyright (c) 2025 TeleAI-infra Team (TeleTron)
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

"""Install teletron's parallel-state extensions into ``megatron.core``.

Applied EXPLICITLY: the DPO entrypoint calls :func:`install` at startup
(before any megatron parallel-state call); nothing happens on import,
and the GRPO line never touches this module. Idempotent. This is a
permanent extension of megatron's lifecycle, not a drift shim — see
``teleboost/engines/teletron/__init__`` for how this differs from
``teleboost/patches/``.
"""

_INSTALLED = False


def _is_installed(parallel_state) -> bool:
    return bool(
        getattr(
            parallel_state.initialize_model_parallel,
            "_teleboost_initialize_wrapper",
            False,
        )
        and getattr(
            parallel_state.destroy_model_parallel,
            "_teleboost_destroy_wrapper",
            False,
        )
    )


def install():
    """Wrap megatron.core.parallel_state with teletron's TCP extensions."""
    global _INSTALLED
    import megatron.core

    parallel_state = megatron.core.parallel_state
    # Do not trust only our module boolean: test isolation, module reloads, or
    # another integration can replace Megatron's functions while leaving the
    # flag behind. Conversely, wrapper sentinels prevent stacking if this
    # adaptor module itself was reloaded.
    if _INSTALLED and _is_installed(parallel_state):
        return

    from teleboost.engines.teletron.parallel_state import (
        destroy_model_parallel_wrapper,
        get_tensor_and_context_parallel_src_rank,
        get_tensor_context_parallel_group,
        get_tensor_context_parallel_rank,
        get_tensor_context_parallel_src_rank,
        get_tensor_context_parallel_world_size,
        initialize_model_parallel_decorators,
    )

    initialize = parallel_state.initialize_model_parallel
    if not getattr(initialize, "_teleboost_initialize_wrapper", False):
        initialize = initialize_model_parallel_decorators(initialize)

    destroy = parallel_state.destroy_model_parallel
    if not getattr(destroy, "_teleboost_destroy_wrapper", False):
        destroy = destroy_model_parallel_wrapper(destroy)

    # Build/validate both wrappers before mutating Megatron so a contract drift
    # cannot leave the adaptor half-installed.
    if not callable(initialize) or not callable(destroy):
        raise RuntimeError("TeleTron failed to construct callable Megatron lifecycle wrappers")

    parallel_state.initialize_model_parallel = initialize
    parallel_state.destroy_model_parallel = destroy
    parallel_state.get_tensor_context_parallel_group = get_tensor_context_parallel_group
    parallel_state.get_tensor_context_parallel_rank = get_tensor_context_parallel_rank
    parallel_state.get_tensor_context_parallel_world_size = get_tensor_context_parallel_world_size
    parallel_state.get_tensor_context_parallel_src_rank = get_tensor_context_parallel_src_rank
    parallel_state.get_tensor_and_context_parallel_src_rank = get_tensor_and_context_parallel_src_rank
    megatron.core.mpu = parallel_state
    _INSTALLED = True
