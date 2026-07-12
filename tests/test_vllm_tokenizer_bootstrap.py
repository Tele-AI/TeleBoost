# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Spawn-safe vLLM tokenizer compatibility bootstrap contract."""

import os
import subprocess
import sys
from pathlib import Path


def test_spawned_interpreter_installs_tokenizer_and_processor_fix():
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap = repo_root / "teleboost" / "patches" / "vllm"
    env = os.environ.copy()
    env["TELEBOOST_VLLM_TOKENIZER_REGEX_FIX"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(bootstrap), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    code = """
from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin
assert AutoTokenizer._teleboost_regex_fix is True
assert ProcessorMixin._teleboost_regex_fix is True
print('bootstrap-ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "bootstrap-ok"
