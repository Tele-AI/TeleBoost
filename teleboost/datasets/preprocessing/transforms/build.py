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
from teleboost.datasets.registry import Registry, build_module

from .formatting import PackInputs, PackInputsNoResize
from .prompt_transform import InjectPromptToTopLevel
from .video_transform import (
    InjectImagesFromVideoTensor,
    InjectRawFirstImageFromVideo,
    PreprocessVideoToTensor,
)

TRANSFORMS = Registry()
TRANSFORMS.register_module(InjectRawFirstImageFromVideo)
TRANSFORMS.register_module(PreprocessVideoToTensor)
TRANSFORMS.register_module(InjectImagesFromVideoTensor)
TRANSFORMS.register_module(InjectPromptToTopLevel)
TRANSFORMS.register_module(PackInputs)
TRANSFORMS.register_module(PackInputsNoResize)


def build_transform(params_or_type, *args, **kwargs):
    return build_module(TRANSFORMS, params_or_type, *args, **kwargs)
