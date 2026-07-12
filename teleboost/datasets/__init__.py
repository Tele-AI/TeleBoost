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
# Public surface for OSS users.
#
# `DPODatasetBase` is self-contained (only depends on torch) so we eager-import
# it. The runtime-backed datasets stay lazy because their constructors depend
# on initialized TeleTron state and the dataset registry still participates in
# the training-loader import graph. Config serialization no longer lives in
# this package.
from teleboost.datasets.dpo import DPODatasetBase

__all__ = [
    "CSVPreferenceDPODataset",
    "DPODatasetBase",
    "FakeDataset",
    "FakeDPODataset",
    "DATASETS",
    "build_dataset",
]


def __getattr__(name):
    """PEP 562 lazy attribute access — resolves once teleboost.engines.teletron is ready."""
    if name == "FakeDataset":
        from teleboost.datasets.rlhf import FakeDataset

        return FakeDataset
    if name == "FakeDPODataset":
        from teleboost.datasets.rlhf import FakeDPODataset

        return FakeDPODataset
    if name == "CSVPreferenceDPODataset":
        from teleboost.datasets.dpo import CSVPreferenceDPODataset

        return CSVPreferenceDPODataset
    if name in ("DATASETS", "build_dataset"):
        from .build import DATASETS, build_dataset

        return {"DATASETS": DATASETS, "build_dataset": build_dataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
