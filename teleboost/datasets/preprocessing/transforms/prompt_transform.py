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
import logging

logger = logging.getLogger(__name__)


class InjectPromptToTopLevel:
    def __init__(self, prompt_key="prompt", target_keys=("struct_prompt", "short_prompt", "dense_prompt"), force_list=True):
        self.prompt_key = prompt_key
        self.target_keys = target_keys
        self.force_list = force_list

    def __call__(self, data_dict: dict):
        p = data_dict.get(self.prompt_key, None)
        if p is None:
            return data_dict

        # Match the dump format: use List[str] even when batch=1.
        if self.force_list and isinstance(p, str):
            p = [p]

        for k in self.target_keys:
            data_dict[k] = p

        return data_dict
