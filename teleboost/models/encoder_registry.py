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
from collections.abc import Callable
from typing import Any

import torch

from teleboost.engines.teletron.distributed.base_encoder import BaseEncoder

_EncoderFactory = Callable[[torch.device], BaseEncoder]

_ENCODER_REGISTRY: dict[str, _EncoderFactory] = {}


def _build_wan_teletron_encoder(device: torch.device, **kwargs: Any) -> BaseEncoder:
    from teleboost.models.wan.teletron.wan_teletron_encoder import WanTeletronEncoder

    return WanTeletronEncoder(device=device, **kwargs)


_ENCODER_REGISTRY["wan_teletron_encoder"] = _build_wan_teletron_encoder


def register_encoder(name: str):
    """
    Decorator that registers an encoder class in the global registry.

    Args:
        name (str): Unique name of the encoder.
    """

    def decorator(cls: type[BaseEncoder]):
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Error: encoder '{name}' is already registered.")
        if not issubclass(cls, BaseEncoder):
            raise TypeError(f"Error: registered class '{cls.__name__}' must be a subclass of BaseEncoder.")

        def _factory(device: torch.device, **kwargs: Any) -> BaseEncoder:
            return cls(device=device, **kwargs)

        _ENCODER_REGISTRY[name] = _factory
        return cls

    return decorator


def get_encoder(name: str, device: torch.device, **kwargs: Any) -> BaseEncoder:
    """
    Look up an encoder by name in the registry and instantiate it.

    Args:
        name (str): Name of the encoder to fetch.
        device (torch.device): Device on which the encoder will be initialized.
        **kwargs: Additional arguments passed to the encoder constructor.

    Returns:
        BaseEncoder: An instance of the requested encoder.

    Raises:
        ValueError: If the requested name is not found in the registry.
    """
    if name not in _ENCODER_REGISTRY:
        raise ValueError(f"Error: encoder '{name}' not found in the registry. Available options: {list(_ENCODER_REGISTRY.keys())}")

    encoder_factory = _ENCODER_REGISTRY[name]
    return encoder_factory(device=device, **kwargs)
