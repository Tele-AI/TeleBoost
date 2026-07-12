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
"""TeleBoost first-party parallel/distributed library (TeleTron lineage).

Serves the megatron/DPO line: context-parallel and tensor-parallel
primitives (``context_parallel/``, ``tensor_parallel/``), the extended
parallel-state groups (``parallel_state.py``), the distributed-VAE
producer/consumer encoders (``distributed/``), and activation-recompute
helpers (``transformer/``). Consumers: ``teleboost/models/wan/teletron``,
``teleboost/training``, ``teleboost/programs/wan/dpo``, and ``tests/unit_tests/teletron``.

Relationship to the two patch surfaces (deliberately separate, zero
mutual imports):

* ``teleboost/patches/`` — verl/vllm/tensordict **drift shims**:
  version-pinned workarounds explicitly installed by runtime entrypoints,
  expected to disappear as upstream moves.
* ``teleboost/engines/teletron/megatron_adaptor.py`` — installs THIS package's
  parallel-state extensions into ``megatron.core`` (via an explicit
  ``install()`` called by the DPO entrypoint; no import side effects). Together with
  ``parallel_state.apply_distributed_op_patches`` (which redirects
  ``torch.distributed`` default groups in multi-model-group mode) it is
  a **permanent extension mechanism**, not a drift shim — which is why
  it lives here and not in ``patches/``.
"""

# TeleTron runtime API used by Megatron/DPO modules.
from teleboost.engines.teletron.rank_print import print_rank_0  # noqa: F401
from teleboost.engines.teletron.wrapped_model import (  # noqa: F401
    get_attr_wrapped_model,
    get_model_config,
)
from teleboost.engines.teletron.config import set_config  # noqa: F401
from teleboost.engines.teletron.runtime_state import (  # noqa: F401
    get_args,
    get_current_global_batch_size,
    get_num_microbatches,
    get_timers,
    set_args,
    set_global_args,
    update_num_microbatches,
)
