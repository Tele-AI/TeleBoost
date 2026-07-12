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
from .registry import ModelRegistry
from .wan.teletron.parallel_wan_teletron_model import ParallelWanTeletronModel

MODEL_REGISTRY = ModelRegistry("model")
MODEL_REGISTRY.register(ParallelWanTeletronModel)


def build_model(name, config=None):
    if config is None:
        return MODEL_REGISTRY.build(name)
    return MODEL_REGISTRY.build(name, config)
