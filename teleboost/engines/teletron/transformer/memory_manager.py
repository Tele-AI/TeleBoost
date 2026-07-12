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
from collections import defaultdict, deque

import torch

from teleboost.engines.teletron import get_args


class MemoryManager:
    """
    Manages pinned-memory buffers and a dedicated CUDA data-transfer stream.
    This is essential for efficient asynchronous data transfers.
    """

    def __init__(self):
        self.device = torch.cuda.current_device
        self.memory_pool = defaultdict(deque)
        args = get_args()
        self.num_layers = args.num_layers
        self._warmup()

    def _warmup(self):
        pass

    def get_buffer(self, shape, dtype):
        """Get a sufficiently large buffer from the pool."""
        key = (shape, dtype)
        if self.memory_pool[key]:
            # Pop an available buffer from the pool
            return self.memory_pool[key].popleft()
        else:
            return torch.empty(shape, dtype=dtype)

    def return_buffer(self, buffer):
        """
        Return a used pinned-memory buffer to the pool.
        """
        key = (buffer.shape, buffer.dtype)
        self.memory_pool[key].append(buffer)


# Global manager instance
_memory_manager = None


def get_memory_manager():
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
