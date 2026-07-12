# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Scoped Python-startup compatibility for spawned vLLM processes.

vLLM 0.14 does not expose arbitrary Hugging Face tokenizer kwargs through
EngineArgs, and Ray forces its vLLM children to use multiprocessing ``spawn``.
Runtime monkeypatches in the parent therefore do not reach the model worker.
The launchers put this directory (not the whole module as ``sitecustomize``)
on PYTHONPATH and set ``TELEBOOST_VLLM_TOKENIZER_REGEX_FIX=1`` so each spawned
interpreter installs the compatibility flag before importing vLLM.
"""

import os


def _install() -> None:
    from transformers import AutoTokenizer
    from transformers.processing_utils import ProcessorMixin

    if not getattr(AutoTokenizer, "_teleboost_regex_fix", False):
        original_tokenizer_loader = AutoTokenizer.from_pretrained.__func__

        @classmethod
        def tokenizer_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
            kwargs.setdefault("fix_mistral_regex", True)
            return original_tokenizer_loader(
                cls,
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )

        AutoTokenizer.from_pretrained = tokenizer_loader
        AutoTokenizer._teleboost_regex_fix = True

    if not getattr(ProcessorMixin, "_teleboost_regex_fix", False):
        original_processor_loader = ProcessorMixin.from_pretrained.__func__

        @classmethod
        def processor_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
            kwargs.setdefault("fix_mistral_regex", True)
            return original_processor_loader(
                cls,
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )

        ProcessorMixin.from_pretrained = processor_loader
        ProcessorMixin._teleboost_regex_fix = True


if os.environ.get("TELEBOOST_VLLM_TOKENIZER_REGEX_FIX", "0") == "1":
    _install()
