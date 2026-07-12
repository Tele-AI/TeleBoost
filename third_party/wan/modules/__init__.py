# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified by TeleBoost contributors in 2026 to make module exports lazy.
"""Lazy model-component exports for the vendored Wan package."""

from importlib import import_module

_EXPORTS = {
    "WanVAE": (".vae", "WanVAE"),
    "WanModel": (".model", "WanModel"),
    "T5Model": (".t5", "T5Model"),
    "T5Encoder": (".t5", "T5Encoder"),
    "T5Decoder": (".t5", "T5Decoder"),
    "T5EncoderModel": (".t5", "T5EncoderModel"),
    "HuggingfaceTokenizer": (".tokenizers", "HuggingfaceTokenizer"),
    "flash_attention": (".attention", "flash_attention"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
