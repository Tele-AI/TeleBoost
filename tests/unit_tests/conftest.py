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
"""Collection support for the environment-heavy ``unit_tests`` tree.

A bare ``pytest tests/`` uses the root core profile, which excludes heavy
modules before import.  The training and heavy profiles instead fail their
dependency preflight when a required runtime is unavailable; this file must
not convert those failures into passing skips.  It only exposes the shared
``unit_tests`` helpers and keeps torchrun executables out of normal discovery.
"""

import os
import sys
from pathlib import Path

# The suites in this tree import shared helpers as ``unit_tests.*`` —
# resolve that package against the tests/ directory.
_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

collect_ignore_glob = []

# torchrun-only suites (filled in per-file below): they initialize
# torch.distributed at import time and can only run under a torchrun
# rendezvous (RANK set by the launcher).
_TORCHRUN_ONLY = [
    "teletron/context_parallel/test_forward_attn_precision.py",
    "teletron/context_parallel/test_context_parallel_mixin_stateless.py",
]
if "RANK" not in os.environ:
    collect_ignore_glob += _TORCHRUN_ONLY
