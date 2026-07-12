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
"""Identity sharding-manager adapter for verl worker paths."""

from __future__ import annotations

from typing import Any

__all__ = ["IdentityShardingManager"]


class IdentityShardingManager:
    """Context manager matching verl sharding-manager hooks without mutation."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def preprocess_data(self, data: Any = None, **kwargs: Any) -> Any:
        return data if data is not None else kwargs.get("prompts")

    def postprocess_data(self, data: Any = None, **kwargs: Any) -> Any:
        return data if data is not None else kwargs.get("prompts")
