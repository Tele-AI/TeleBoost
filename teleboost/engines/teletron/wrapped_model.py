# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025 TeleAI-infra Team.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
# TeleAI modifications are licensed under Apache-2.0; see LICENSE at the root.
"""Reach attributes through model wrappers (DDP / Float16Module) — Megatron-LM port."""


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    """Get an attribute from a wrapped model.
    If return_model_obj is true, return the object that has the 'attr' attribute;
    otherwise, return the attribute directly."""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else:

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if hasattr(model, "module"):
            model = model.module
        else:
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

    if return_model_obj:
        return model
    return getattr(model, attr)


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)
