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
"""Inject the diffusion-aware RLHFDataset onto upstream verl.

Upstream `verl.utils.dataset.rl_dataset.RLHFDataset` only supports parquet
input with chat-template tokenized prompts. TeleBoost rolls out from
prompts whose umT5 context embeddings have been pre-encoded to .npy files
and indexed in a JSON manifest (`context_path` + optional
`context_null_path`).

Rather than fork verl, we swap the class on `verl.utils.dataset.rl_dataset`
at import time. Recipe code that imports `from verl.utils.dataset.rl_dataset
import RLHFDataset` then transparently gets TeleBoost's diffusion-aware
implementation.

Also exposes the matching collate function so the recipes trainer can wire
it onto the DataLoader.
"""

from __future__ import annotations


def apply() -> None:
    import verl.utils.dataset.rl_dataset as _vrl_ds

    from teleboost.patches._rlhf_dataset import RLHFDataset as _DanceGRPORLHFDataset
    from teleboost.patches._rlhf_dataset import wan_preprocessed_collate_function

    # RLHFDataset is an OVERRIDE of verl's class — assert it exists so an
    # upstream rename fails loud instead of shadowing. (wan_preprocessed_collate_
    # function is a teleboost-added attribute, nothing to assert.)
    assert hasattr(_vrl_ds, "RLHFDataset"), "verl drift: rl_dataset.RLHFDataset gone"
    _vrl_ds.wan_preprocessed_collate_function = wan_preprocessed_collate_function
    _vrl_ds.RLHFDataset = _DanceGRPORLHFDataset
