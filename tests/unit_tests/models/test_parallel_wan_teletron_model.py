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
import math
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from dataclasses import asdict, dataclass
from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core import mpu

# Env-sensitive suite: needs a matched flash-attn build / CP-capable env.
pytestmark = pytest.mark.heavy_env

TELEAI_MODEL_FWD_SUCCESS = "Parallel Wan model forward test success"
TELEAI_MODEL_FWD_FAIL = "Parallel Wan model forward test fail"
TELEAI_MODEL_BWD_SUCCESS = "Parallel Wan model backward test success"
TELEAI_MODEL_BWD_FAIL = "Parallel Wan model backward test fail"

CUDA_DEVICES = [0, 1, 2, 3]


@dataclass
class WanTeletronParams:
    """Small but structurally valid Wan config for TP/CP contract testing."""

    dim: int = 128
    in_dim: int = 36
    out_dim: int = 16
    text_dim: int = 64
    freq_dim: int = 32
    ffn_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_heads: int = 4
    num_layers: int = 1
    has_image_input: bool = True
    has_image_pos_emb: bool = False


_TEXT_CONTEXT_TOKENS = 512
_CHILD_TIMEOUT_SECONDS = 180.0
_FORWARD_TOLERANCE = 2e-3
_BACKWARD_TOLERANCE = 2.5e-2


def _synthetic_inputs(device: torch.device, params: WanTeletronParams):
    """Build deterministic I2V inputs with three latent patches.

    Wan's I2V path concatenates 16 noisy-latent channels with 20 image-latent
    channels. Three CLIP tokens plus the fixed 512 T5 tokens exercise both
    cross-attention branches while keeping the test small. Three patches make
    self-attention query/key gradients meaningful and exercise CP padding at
    world size two.
    """
    generator = torch.Generator(device="cpu").manual_seed(2026)

    def randn(*shape):
        return torch.randn(shape, generator=generator).to(
            device=device,
            dtype=torch.bfloat16,
        )

    noisy_channels = params.out_dim
    image_channels = params.in_dim - noisy_channels
    assert image_channels > 0
    patch_f, patch_h, patch_w = params.patch_size
    latent_width = 3 * patch_w
    return {
        "noisy_latents": randn(
            1,
            noisy_channels,
            patch_f,
            patch_h,
            latent_width,
        ),
        "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
        "context": randn(1, _TEXT_CONTEXT_TOKENS, params.text_dim),
        "clip_feature": randn(1, 3, 1280),
        "y": randn(1, image_channels, patch_f, patch_h, latent_width),
    }


@patch("teleboost.engines.teletron.set_config")
@patch("teleboost.engines.teletron.get_args")
def parallel_wan_teletron_model_testing(
    rank,
    world_size,
    q,
    tp_size,
    cp_size,
    master_port,
    mock_get_args,
    mock_set_config,
):
    process_group_initialized = False
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)

        from megatron.core.transformer import TransformerConfig

        from teleboost.models.wan.teletron import (
            ParallelWanTeletronModel,
            WanTeletronModel,
        )
        from teleboost.engines.teletron.parallel_state import initialize_model_parallel_base

        params = WanTeletronParams()
        args = Mock()
        args.recompute_method = "block"
        args.recompute_granularity = "full"
        args.recompute_num_layers = 1
        args.activation_offload = True
        args.num_layers = params.num_layers
        args.num_attention_heads = params.num_heads
        args.distributed_vae = False
        args.consumer_models_num = 1
        mock_get_args.return_value = args
        mock_set_config.return_value = {
            "model_config": {
                "dit": {
                    "type": "ParallelWanTeletronModel",
                    "config": asdict(params),
                }
            }
        }

        cfg = Mock(spec=TransformerConfig)
        cfg._cpu_offloading_context = None
        cfg.perform_initialization = True
        cfg.use_cpu_initialization = True
        cfg.params_dtype = torch.bfloat16
        cfg.gradient_accumulation_fusion = False
        cfg.expert_model_parallel_size = 1
        cfg.defer_embedding_wgrad_compute = False
        cfg.async_tensor_model_parallel_allreduce = False
        cfg.num_layers = args.num_layers
        cfg.sequence_parallel = False

        assert len(CUDA_DEVICES) >= world_size, "GPU number is not enough"
        cuda_rank = CUDA_DEVICES[rank]
        torch.cuda.set_device(cuda_rank)
        device = torch.device("cuda", cuda_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
        )
        process_group_initialized = True

        initialize_model_parallel_base(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
            nccl_communicator_config_path=None,
            distributed_timeout_minutes=3,
        )

        torch.manual_seed(1234)
        wan_teletron_model = WanTeletronModel(**asdict(params)).to(
            device=device,
            dtype=torch.bfloat16,
        )
        torch.manual_seed(1234)
        parallel_wan_teletron_model = ParallelWanTeletronModel(cfg).to(
            device=device,
            dtype=torch.bfloat16,
        )
        parallel_wan_teletron_model.load_state_dict(tp_load_state_dict(wan_teletron_model))
        inputs = _synthetic_inputs(device, params)

        wan_output = wan_teletron_model(
            x=inputs["noisy_latents"],
            timestep=inputs["timestep"],
            context=inputs["context"],
            clip_feature=inputs["clip_feature"],
            y=inputs["y"],
        )
        parallel_output = parallel_wan_teletron_model(
            x=inputs["noisy_latents"],
            timestep=inputs["timestep"],
            context=inputs["context"],
            clip_feature=inputs["clip_feature"],
            y=inputs["y"],
        )
        forward_distance = normalized_euclid_dist(wan_output, parallel_output)
        if forward_distance >= _FORWARD_TOLERANCE:
            raise AssertionError(f"{TELEAI_MODEL_FWD_FAIL} rank{rank}: normalized distance={forward_distance:.6g}")
        q.put(f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}")

        wan_output.backward(torch.ones_like(wan_output))
        parallel_output.backward(torch.ones_like(parallel_output))
        model_grads = {name: param.grad for name, param in wan_teletron_model.named_parameters() if param.grad is not None}
        parallel_model_grads = {name: param.grad for name, param in parallel_wan_teletron_model.named_parameters() if param.grad is not None}
        missing = sorted(set(model_grads) - set(parallel_model_grads))
        if missing:
            raise AssertionError(f"parallel model is missing gradients on rank {rank}: {missing}")

        tp_rank = mpu.get_tensor_model_parallel_rank()
        mismatches = []
        for name, model_grad in model_grads.items():
            parallel_grad = parallel_model_grads[name]
            reference_grad = tp_reference_grad(
                tp_rank,
                name,
                model_grad,
                parallel_grad,
            )
            distance = normalized_euclid_dist(reference_grad, parallel_grad)
            if not math.isfinite(distance) or distance >= _BACKWARD_TOLERANCE:
                mismatches.append(
                    (
                        name,
                        distance,
                        float(reference_grad.float().norm()),
                        float(parallel_grad.float().norm()),
                        float((reference_grad.float() - parallel_grad.float()).norm()),
                    )
                )
        if mismatches:
            raise AssertionError(f"{TELEAI_MODEL_BWD_FAIL} rank{rank}: {mismatches[:5]}")
        q.put(f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}")
    except BaseException:
        q.put(("ERROR", rank, traceback.format_exc()))
        raise
    finally:
        if process_group_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def tp_reference_grad(rank, name, output, parallel_output):
    col_w = ["self_attn.query.weight", "self_attn.key.weight", "self_attn.value.weight", "ffn.0.weight", "cross_attn.query.weight", "cross_attn.key.weight", "cross_attn.value.weight", "cross_attn.img_key.weight", "cross_attn.img_value.weight"]

    col_b = ["self_attn.query.bias", "self_attn.key.bias", "self_attn.value.bias", "ffn.0.bias", "cross_attn.query.bias", "cross_attn.key.bias", "cross_attn.value.bias", "cross_attn.img_key.bias", "cross_attn.img_value.bias"]

    row_w = ["ffn.2.weight", "self_attn.out_proj.weight", "cross_attn.out_proj.weight"]

    norm_w = ["self_attn.norm_query.weight", "self_attn.norm_key.weight", "cross_attn.norm_query.weight", "cross_attn.norm_key.weight", "cross_attn.norm_image_key.weight"]

    if any(cw in name for cw in col_w):
        size = parallel_output.shape[0]
        return output[rank * size : (rank + 1) * size, :]
    elif any(cb in name for cb in col_b):
        size = parallel_output.shape[0]
        return output[rank * size : (rank + 1) * size]
    elif any(rw in name for rw in row_w):
        size = parallel_output.shape[1]
        return output[:, rank * size : (rank + 1) * size]
    elif any(nw in name for nw in norm_w):
        size = parallel_output.shape[0]
        return output[rank * size : (rank + 1) * size]
    else:
        return output


def normalized_euclid_dist(output, parallel_output) -> float:
    wan_teletron_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    denominator = wan_teletron_norm + parallel_norm
    if denominator == 0.0:
        return 0.0 if euclid_dist.item() == 0.0 else float("inf")
    return float((0.5 * euclid_dist / denominator).detach().item())


def tp_load_state_dict(base_model):
    base_dict = base_model.state_dict()
    tp_dict = {}

    col_w = ["self_attn.query.weight", "self_attn.key.weight", "self_attn.value.weight", "ffn.0.weight", "cross_attn.query.weight", "cross_attn.key.weight", "cross_attn.value.weight", "cross_attn.img_key.weight", "cross_attn.img_value.weight"]

    col_b = ["self_attn.query.bias", "self_attn.key.bias", "self_attn.value.bias", "ffn.0.bias", "cross_attn.query.bias", "cross_attn.key.bias", "cross_attn.value.bias", "cross_attn.img_key.bias", "cross_attn.img_value.bias"]

    row_w = ["ffn.2.weight", "self_attn.out_proj.weight", "cross_attn.out_proj.weight"]

    norm_w = ["self_attn.norm_query.weight", "self_attn.norm_key.weight", "cross_attn.norm_query.weight", "cross_attn.norm_key.weight", "cross_attn.norm_image_key.weight"]

    def tp_col_weight_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        size = param.shape[0] // tp_size
        tp_dict[name] = param[rank * size : (rank + 1) * size, :]

    def tp_col_bias_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        size = param.shape[0] // tp_size
        tp_dict[name] = param[rank * size : (rank + 1) * size]

    def tp_row_weight_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        size = param.shape[1] // tp_size
        tp_dict[name] = param[:, rank * size : (rank + 1) * size]

    def tp_row_bias_load(tp_dict, name, param):
        tp_dict[name] = param

    def tp_norm_weight_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        size = param.shape[0] // tp_size
        tp_dict[name] = param[rank * size : (rank + 1) * size]

    for name, param in base_dict.items():
        if any(cw in name for cw in col_w):
            tp_col_weight_load(tp_dict, name, param)
        elif any(cb in name for cb in col_b):
            tp_col_bias_load(tp_dict, name, param)
        elif any(rw in name for rw in row_w):
            tp_row_weight_load(tp_dict, name, param)
        elif any(nw in name for nw in norm_w):
            tp_norm_weight_load(tp_dict, name, param)
        else:
            tp_dict[name] = param
    return tp_dict


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drain_queue(result_queue) -> list:
    messages = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            return messages


def _stop_processes(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def launch_multiprocess_testing(world_size, tp_size, cp_size):
    assert world_size == tp_size * cp_size
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    master_port = _free_tcp_port()
    processes = [
        context.Process(
            target=parallel_wan_teletron_model_testing,
            args=(rank, world_size, result_queue, tp_size, cp_size, master_port),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    messages = []
    deadline = time.monotonic() + _CHILD_TIMEOUT_SECONDS
    timed_out = False
    child_failed = False
    while True:
        messages.extend(_drain_queue(result_queue))
        child_failed = any(isinstance(message, tuple) and message[:1] == ("ERROR",) for message in messages) or any(process.exitcode not in (None, 0) for process in processes)
        if child_failed or all(process.exitcode is not None for process in processes):
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.05)

    if timed_out or child_failed:
        _stop_processes(processes)
    else:
        for process in processes:
            process.join(timeout=5)
    messages.extend(_drain_queue(result_queue))
    result_queue.close()
    result_queue.join_thread()

    if timed_out:
        raise AssertionError(f"Wan TP/CP child processes exceeded {_CHILD_TIMEOUT_SECONDS:.0f}s")
    errors = [message for message in messages if isinstance(message, tuple) and message[:1] == ("ERROR",)]
    if errors:
        details = "\n".join(f"rank {rank}:\n{child_traceback}" for _, rank, child_traceback in errors)
        raise AssertionError(f"Wan TP/CP child failed:\n{details}")
    bad_exitcodes = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if bad_exitcodes:
        raise AssertionError(f"Wan TP/CP child exited without a traceback: {bad_exitcodes}")
    return [message for message in messages if isinstance(message, str)]


class testParallelWanModel(TestCase):
    def test_single(self):
        world_size = 1
        responses = launch_multiprocess_testing(world_size, 1, 1)

        correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size)]
        correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size)]

        self.assertEqual(sorted(responses), correct_responses)

    def test_tp(self):
        world_size = tensor_model_parallel_world_size = 2
        responses = launch_multiprocess_testing(world_size, tensor_model_parallel_world_size, 1)

        correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size)]
        correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size)]

        self.assertEqual(sorted(responses), correct_responses)

    def test_cp(self):
        world_size = cp_size = 2
        responses = launch_multiprocess_testing(world_size, 1, cp_size)

        correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size)]
        correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size)]

        self.assertEqual(sorted(responses), correct_responses)

    # def test_tp_cp(self):
    #     cp_size = 2
    #     tp_size = 2
    #     world_size = cp_size * tp_size
    #     responses = launch_multiprocess_testing(world_size, tp_size, cp_size)

    #     correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size )]
    #     correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size )]

    #     self.assertEqual(sorted(responses), correct_responses)


if __name__ == "__main__":
    tensor_model_parallel_world_size = 1
    world_size = 1
    cp_size = world_size // tensor_model_parallel_world_size
    responses = launch_multiprocess_testing(world_size, tensor_model_parallel_world_size, cp_size)

    correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size)]
    correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size)]
    print(f"test_result: {responses}")
    assert sorted(responses) == correct_responses
    print("test success!")
