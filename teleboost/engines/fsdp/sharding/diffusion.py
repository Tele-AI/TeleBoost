# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP


from typing import Any

from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_device_is_available

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusionBaseShardingManager:
    @check_device_is_available()
    def __init__(self, module: FSDP, inference_engine: Any, model_config, full_params: bool = False, device_mesh: DeviceMesh = None, offload_param: bool = False, load_format: str = "dummy_hf", layered_summon: bool = True):
        # ``inference_engine`` is typed as ``Any`` because diffusion rollouts
        # don't hold a vLLM ``LLM`` handle — the type hint was a holdover
        # from the fork-only ``verl.third_party.vllm.LLM``. The legacy body
        # (lines 56-101) that reached into ``llm_engine.model_executor.driver_worker.…``
        # is commented out; once the wan rollout switches to the upstream
        # ``vLLMReplica`` HTTP path this whole class can be deleted.
        self.module = module
        self.offload_param = offload_param

    def __enter__(self):
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)

        # add empty cache after each compute
        get_torch_device().empty_cache()

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data
