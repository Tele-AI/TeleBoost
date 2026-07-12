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
from abc import ABC, abstractmethod
from typing import Any

import torch

from teleboost.engines.teletron import get_args


def get_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported options are 'float32', 'float16', 'bfloat16'.")


class BaseEncoder(ABC):
    def __init__(self, device: torch.device, **kwargs: Any):
        args = get_args()
        self.device = device
        self.moe = args.consumer_models_num > 1
        self.dtype = get_dtype(args.encoder_dtype)

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """
        init models
        """
        pass

    @abstractmethod
    def encode(self, raw_batch: dict[str, Any]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """

        Args:
            raw_batch (Dict[str, Any])

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]:
            - Tensor List (List[torch.Tensor])
            - Sizes of Tensors (torch.Tensor)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_output_schema() -> list[str]:
        """
        Return an ordered list with the names of all tensors the encoder outputs.
        The order must strictly match the order of the tensor list returned by encode().
        This is a static method, so it can be called without an instance.

        Returns:
            List[str]: a list of tensor names, e.g. ['context', 'clip,'image_feature', 'latents'].
        """
        pass

    @staticmethod
    def _pack_tensors(tensors_to_pack: list[torch.Tensor], dtype=torch.bfloat16) -> torch.Tensor:
        """
        Flatten a list of tensors and concatenate them into a single flat tensor.
        """
        if not tensors_to_pack:
            return torch.tensor([])

        flattened_tensors = [t.flatten().to(dtype) for t in tensors_to_pack]
        return torch.cat(flattened_tensors, dim=0)
