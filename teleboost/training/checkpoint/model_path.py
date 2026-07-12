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
import os


def get_model_dir():
    model_dir = os.environ.get("TELEBOOST_MODEL_DIR") or os.environ.get("MODEL_DIR")
    if model_dir:
        return model_dir
    raise RuntimeError("model directory is not configured")


def get_huggingface_model_path(model_name_or_path):
    return model_name_or_path


def get_model_path(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        raise ValueError(f"{model_name_or_path} does not exist")
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, model_name_or_path)
    if os.path.exists(model_path):
        return model_path
    return get_huggingface_model_path(model_name_or_path)
