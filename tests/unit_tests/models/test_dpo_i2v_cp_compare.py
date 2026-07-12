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
import os
from functools import wraps
from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core import mpu
from unit_tests.test_utils import spawn

from teleboost.models.flow_match import FlowMatchScheduler
from teleboost.engines.teletron.parallel_state import get_tensor_and_context_parallel_src_rank

# Env-sensitive suite: needs a matched flash-attn build / CP-capable env.
pytestmark = pytest.mark.heavy_env

DPO_CP_FWD_SUCCESS = "DPO i2v forward CP compare success"
DPO_CP_FWD_FAIL = "DPO i2v forward CP compare fail"
DPO_CP_BWD_SUCCESS = "DPO i2v backward CP compare success"
DPO_CP_BWD_FAIL = "DPO i2v backward CP compare fail"

CUDA_DEVICES = [0, 1, 2, 3]


def _normalized_euclid_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a.norm().item()
    b_norm = b.norm().item()
    denom = a_norm + b_norm
    if denom == 0:
        return 0.0
    return 0.5 * torch.norm(a - b).item() / denom


def tp_normalized_euclid_dist(rank, name, output, parallel_output):
    col_w = [
        "self_attn.query.weight",
        "self_attn.key.weight",
        "self_attn.value.weight",
        "ffn.0.weight",
        "cross_attn.query.weight",
        "cross_attn.key.weight",
        "cross_attn.value.weight",
        "cross_attn.img_key.weight",
        "cross_attn.img_value.weight",
    ]

    col_b = [
        "self_attn.query.bias",
        "self_attn.key.bias",
        "self_attn.value.bias",
        "ffn.0.bias",
        "cross_attn.query.bias",
        "cross_attn.key.bias",
        "cross_attn.value.bias",
        "cross_attn.img_key.bias",
        "cross_attn.img_value.bias",
    ]

    row_w = [
        "ffn.2.weight",
        "self_attn.out_proj.weight",
        "cross_attn.out_proj.weight",
    ]

    norm_w = [
        "self_attn.norm_query.weight",
        "self_attn.norm_key.weight",
        "cross_attn.norm_query.weight",
        "cross_attn.norm_key.weight",
        "cross_attn.norm_image_key.weight",
    ]

    def normalized_euclid_dist(name, output, parallel_output):
        return _normalized_euclid_dist(output, parallel_output)

    if any(cw in name for cw in col_w):
        size = parallel_output.shape[0]
        return normalized_euclid_dist(name, output[rank * size : (rank + 1) * size, :], parallel_output)
    if any(cb in name for cb in col_b):
        size = parallel_output.shape[0]
        return normalized_euclid_dist(name, output[rank * size : (rank + 1) * size], parallel_output)
    if any(rw in name for rw in row_w):
        size = parallel_output.shape[1]
        return normalized_euclid_dist(name, output[:, rank * size : (rank + 1) * size], parallel_output)
    if any(nw in name for nw in norm_w):
        size = parallel_output.shape[0]
        return normalized_euclid_dist(name, output[rank * size : (rank + 1) * size], parallel_output)
    return normalized_euclid_dist(name, output, parallel_output)


def _broadcast_tensor(tensor: torch.Tensor):
    if mpu.get_tensor_and_context_parallel_world_size() > 1:
        dist_group = mpu.get_tensor_and_context_parallel_group()
        src_rank = get_tensor_and_context_parallel_src_rank()
        torch.distributed.broadcast(tensor, src_rank, group=dist_group)


def _make_or_broadcast(shape, dtype, device, seed, tag):
    src_rank = get_tensor_and_context_parallel_src_rank()
    if torch.distributed.get_rank() == src_rank:
        gen = torch.Generator(device=device).manual_seed(seed)
        out = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    else:
        out = torch.empty(shape, device=device, dtype=dtype)
    _broadcast_tensor(out)
    return out


def _generate_noise_pair(chosen_latents, reject_latents, base_seed, curr_iter):
    if base_seed is not None:
        seed_chosen = int(base_seed) + curr_iter * 2
        seed_reject = int(base_seed) + curr_iter * 2 + 1
        gen_chosen = torch.Generator(device=chosen_latents.device).manual_seed(seed_chosen)
        gen_reject = torch.Generator(device=reject_latents.device).manual_seed(seed_reject)
        noise_chosen = torch.randn(
            chosen_latents.shape,
            device=chosen_latents.device,
            dtype=chosen_latents.dtype,
            generator=gen_chosen,
        )
        noise_reject = torch.randn(
            reject_latents.shape,
            device=reject_latents.device,
            dtype=reject_latents.dtype,
            generator=gen_reject,
        )
    else:
        noise_chosen = torch.randn_like(chosen_latents)
        noise_reject = torch.randn_like(reject_latents)
    return noise_chosen, noise_reject


def _setup_tensorwatch(model, rank, run_tag, enable_tensorwatch=True):
    if not enable_tensorwatch:
        return None
    try:
        from tensorwatch import TensorWatch, watch_module_forward_backward
    except Exception as exc:
        print(f"[Rank {rank}] TensorWatch import failed: {exc}")
        return None

    base_dir = os.path.join(os.getcwd(), "tensorwatch_data")
    os.makedirs(base_dir, exist_ok=True)

    if hasattr(TensorWatch, "reset"):
        try:
            TensorWatch.reset()
        except Exception:
            pass
    if hasattr(TensorWatch, "set_save_dir"):
        try:
            TensorWatch.set_save_dir(base_dir)
        except Exception:
            pass
    if hasattr(TensorWatch, "set_run_name"):
        try:
            TensorWatch.set_run_name(run_tag)
        except Exception:
            pass
    elif hasattr(TensorWatch, "run_name"):
        try:
            TensorWatch.run_name = run_tag
        except Exception:
            pass

    watch_module_forward_backward(model, use_megatron=False, use_deepspeed=False)
    if hasattr(TensorWatch, "is_save_tensor"):
        TensorWatch.is_save_tensor = True
    print(f"[Rank {rank}] TensorWatch enabled: {base_dir} (run={run_tag})")
    return TensorWatch


def _compute_single_loss(
    latents,
    context,
    clip_feature,
    y,
    flow_scheduler,
    model,
    timestep,
    noise,
):
    # Keep model inputs on CUDA, but compute scheduler scalars on CPU to avoid device mismatch.
    timestep_cpu = timestep.detach().float().cpu()
    timesteps_cpu = flow_scheduler.timesteps.detach().float().cpu()
    sigmas_cpu = flow_scheduler.sigmas.detach().float().cpu()
    weights_cpu = flow_scheduler.linear_timesteps_weights.detach().float().cpu()
    timestep_id = torch.argmin((timesteps_cpu - timestep_cpu).abs())
    sigma = sigmas_cpu[timestep_id].to(device=latents.device, dtype=latents.dtype)
    loss_weight = weights_cpu[timestep_id].to(device=latents.device, dtype=latents.dtype)
    training_target = noise - latents
    noisy_latents = (1 - sigma) * latents + sigma * noise

    output_tensor = model(
        x=noisy_latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
    )
    loss = torch.nn.functional.mse_loss(output_tensor.float(), training_target.float())
    loss = loss * loss_weight
    return output_tensor, loss


def _with_process_group_cleanup(func):
    """Destroy all distributed groups even when a spawned rank fails."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    return wrapped


@_with_process_group_cleanup
@patch("teleboost.engines.teletron.set_config")
@patch("teleboost.engines.teletron.get_args")
def dpo_i2v_cp_compare_worker(rank, world_size, q, tp_size, cp_size, seed, run_tag, enable_tensorwatch, mock_get_args, mock_set_config):
    from megatron.core.transformer import TransformerConfig

    from teleboost.engines.teletron.parallel_state import initialize_model_parallel_base
    from teleboost.models.wan.teletron import ParallelWanTeletronModel

    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.activation_offload = False
    args.num_layers = 2
    args.num_attention_heads = 40
    args.distributed_vae = False
    args.consumer_models_num = 1
    args.profile_path = None
    mock_get_args.return_value = args

    model_config = dict(
        dit=dict(
            type="ParallelWanTeletronModel",
            config=dict(
                has_image_input=False,
                patch_size=[1, 2, 2],
                in_dim=36,
                dim=5120,
                ffn_dim=13824,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40,
                num_layers=2,
                eps=1e-6,
                has_image_pos_emb=False,
            ),
            train=dict(
                dpo=dict(beta=0.1),
            ),
        )
    )
    mock_set_config.return_value = {"model_config": model_config}

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

    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    print(f"[Rank {rank}] init_process_group done (world_size={world_size}, tp={tp_size}, cp={cp_size})")

    assert len(CUDA_DEVICES) >= world_size, "GPU number is not enough"
    cuda_rank = CUDA_DEVICES[rank]
    torch.cuda.set_device(cuda_rank)
    device = torch.device(f"cuda:{cuda_rank}")

    initialize_model_parallel_base(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        use_sharp=False,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        nccl_communicator_config_path=None,
        distributed_timeout_minutes=30,
    )
    print(f"[Rank {rank}] model parallel init done")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ParallelWanTeletronModel(cfg).to(device=device, dtype=torch.bfloat16)
    tensorwatch = _setup_tensorwatch(model, rank, run_tag, enable_tensorwatch=enable_tensorwatch)
    model.train()
    model.zero_grad(set_to_none=True)
    print(f"[Rank {rank}] model init done")

    # ---- build scheduler ----
    flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
    flow_scheduler.set_timesteps(1000, training=True)
    flow_scheduler.sigmas = flow_scheduler.sigmas.to(device)
    flow_scheduler.timesteps = flow_scheduler.timesteps.to(device)
    flow_scheduler.linear_timesteps_weights = flow_scheduler.linear_timesteps_weights.to(device)
    print(f"[Rank {rank}] scheduler init done")

    # ---- random inputs (broadcast within CP group) ----
    dtype = torch.bfloat16
    batch = 1
    latent_channels = 16
    mask_channels = 4
    y_channels = latent_channels + mask_channels
    latent_channels + y_channels

    raw_num_frames = 49
    compression_t = 4
    compression_hw = 8
    latent_frames = (raw_num_frames + compression_t - 1) // compression_t
    raw_height = 480
    raw_width = 832
    height = raw_height // compression_hw
    width = raw_width // compression_hw

    ctx_len = 512

    chosen_latents = _make_or_broadcast((batch, latent_channels, latent_frames, height, width), dtype, device, seed + 1, "chosen_latents")
    reject_latents = _make_or_broadcast((batch, latent_channels, latent_frames, height, width), dtype, device, seed + 2, "reject_latents")
    context = _make_or_broadcast((batch, ctx_len, 4096), dtype, device, seed + 3, "context")
    y = _make_or_broadcast((batch, y_channels, latent_frames, height, width), dtype, device, seed + 4, "y")
    print(f"[Rank {rank}] inputs ready: latents={chosen_latents.shape}, y={y.shape}, context={context.shape}")

    src_rank = get_tensor_and_context_parallel_src_rank()
    if torch.distributed.get_rank() == src_rank:
        t_idx_c = torch.tensor([123], device=device)
        t_idx_r = torch.tensor([456], device=device)
        timestep_c = flow_scheduler.timesteps[t_idx_c].to(dtype=dtype)
        timestep_r = flow_scheduler.timesteps[t_idx_r].to(dtype=dtype)
    else:
        timestep_c = torch.empty((1,), device=device, dtype=dtype)
        timestep_r = torch.empty((1,), device=device, dtype=dtype)
    _broadcast_tensor(timestep_c)
    _broadcast_tensor(timestep_r)
    if torch.distributed.get_rank() == src_rank:
        print(f"[Rank {rank}] timesteps ready: t_c={float(timestep_c.item()):.4f}, t_r={float(timestep_r.item()):.4f}")

    noise_chosen, noise_reject = _generate_noise_pair(chosen_latents, reject_latents, base_seed=seed + 10, curr_iter=0)
    _broadcast_tensor(noise_chosen)
    _broadcast_tensor(noise_reject)
    print(f"[Rank {rank}] noise ready")

    # ---- forward + loss (DPO-style) ----
    output_chosen, loss_chosen = _compute_single_loss(
        chosen_latents,
        context,
        None,
        y,
        flow_scheduler,
        model,
        timestep_c,
        noise_chosen,
    )
    output_reject, loss_reject = _compute_single_loss(
        reject_latents,
        context,
        None,
        y,
        flow_scheduler,
        model,
        timestep_r,
        noise_reject,
    )
    if torch.distributed.get_rank() == src_rank:
        print(f"[Rank {rank}] forward done: loss_chosen={float(loss_chosen.detach().float().cpu().item()):.6f}, loss_reject={float(loss_reject.detach().float().cpu().item()):.6f}")

    beta = float(model_config["dit"]["train"]["dpo"]["beta"])
    advantage = (loss_reject - loss_chosen).clamp(-20, 20)
    with torch.no_grad():
        coeff = beta * torch.sigmoid(-beta * advantage)
        coeff = coeff / coeff.numel()
    loss_reject_scaled = (coeff * loss_reject).sum()
    loss_chosen_scaled = (-coeff * loss_chosen).sum()
    dpo_loss_for_log = (-torch.nn.functional.logsigmoid(beta * advantage)).mean().detach()

    # DPO-style: backward on reject then chosen (accumulate grads)
    loss_reject_scaled.backward()
    loss_chosen_scaled.backward()
    if tensorwatch is not None and hasattr(tensorwatch, "step"):
        tensorwatch.step()
    if torch.distributed.get_rank() == src_rank:
        print(f"[Rank {rank}] backward done (reject + chosen)")

    import numpy as np

    # Convert grads to numpy so the Queue does not pass torch.Tensor (avoids storage fd issues).
    grad_payload: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_payload[name] = param.grad.detach().float().cpu().numpy()

    if torch.distributed.get_rank() == src_rank:
        payload = {
            "rank": rank,
            "loss_chosen": float(loss_chosen.detach().float().cpu().item()),
            "loss_reject": float(loss_reject.detach().float().cpu().item()),
            "dpo_loss": float(dpo_loss_for_log.detach().float().cpu().item()),
            # Convert outputs to numpy too, to avoid passing Tensors through the Queue.
            "output_chosen": output_chosen.detach().float().cpu().numpy(),
            "output_reject": output_reject.detach().float().cpu().numpy(),
            "grads": grad_payload,
        }
        q.put(payload)
        print(f"[Rank {rank}] payload queued")


def _launch_dpo_cp_compare(world_size, tp_size, cp_size, seed, port, run_tag, enable_tensorwatch):
    assert world_size == tp_size * cp_size
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    q = spawn(world_size, dpo_i2v_cp_compare_worker, tp_size, cp_size, seed, run_tag, enable_tensorwatch)
    responses = []
    while not q.empty():
        responses.append(q.get())
    return responses


class testDPOI2VCPCompare(TestCase):
    def test_dpo_i2v_cp_compare(self):
        baseline = _launch_dpo_cp_compare(world_size=1, tp_size=1, cp_size=1, seed=1234, port=12455, run_tag="cp1", enable_tensorwatch=True)
        cp_run = _launch_dpo_cp_compare(world_size=2, tp_size=1, cp_size=2, seed=1234, port=12456, run_tag="cp2", enable_tensorwatch=True)
        print(f"[Test] payloads: baseline={len(baseline)}, cp_run={len(cp_run)}")

        base_payload = next((x for x in baseline if x.get("rank") == 0), None)
        cp_payload = next((x for x in cp_run if x.get("rank") == 0), None)
        self.assertIsNotNone(base_payload, "missing baseline payload")
        self.assertIsNotNone(cp_payload, "missing cp payload")

        # forward compare
        out_c_base = torch.from_numpy(base_payload["output_chosen"])
        out_c_cp = torch.from_numpy(cp_payload["output_chosen"])
        out_r_base = torch.from_numpy(base_payload["output_reject"])
        out_r_cp = torch.from_numpy(cp_payload["output_reject"])

        dist_c = _normalized_euclid_dist(out_c_base, out_c_cp)
        dist_r = _normalized_euclid_dist(out_r_base, out_r_cp)
        print(f"[Test] forward dist: chosen={dist_c:.6f}, reject={dist_r:.6f}")

        if dist_c < 0.01 and dist_r < 0.01:
            fwd_msg = DPO_CP_FWD_SUCCESS
        else:
            fwd_msg = f"{DPO_CP_FWD_FAIL} dist_c={dist_c:.6f} dist_r={dist_r:.6f}"
        self.assertTrue(dist_c < 0.01 and dist_r < 0.01, fwd_msg)

        # loss compare
        self.assertAlmostEqual(base_payload["loss_chosen"], cp_payload["loss_chosen"], places=2)
        self.assertAlmostEqual(base_payload["loss_reject"], cp_payload["loss_reject"], places=2)
        self.assertAlmostEqual(base_payload["dpo_loss"], cp_payload["dpo_loss"], places=2)
        print(f"[Test] losses: chosen={base_payload['loss_chosen']:.6f}/{cp_payload['loss_chosen']:.6f}, reject={base_payload['loss_reject']:.6f}/{cp_payload['loss_reject']:.6f}, dpo={base_payload['dpo_loss']:.6f}/{cp_payload['dpo_loss']:.6f}")

        # backward compare
        grad_base = {k: torch.from_numpy(v) for k, v in base_payload["grads"].items()}
        grad_cp = {k: torch.from_numpy(v) for k, v in cp_payload["grads"].items()}
        grad_dists = []
        max_grad_dist = None
        max_grad_name = None
        tp_rank = 0
        for name in grad_base:
            if name not in grad_cp:
                continue
            dist = tp_normalized_euclid_dist(tp_rank, name, grad_base[name], grad_cp[name])
            grad_dists.append(dist)
            if max_grad_dist is None or dist > max_grad_dist:
                max_grad_dist = dist
                max_grad_name = name
        if grad_dists:
            print(f"[Test] grad dist: max={max_grad_dist:.6f} ({max_grad_name}), mean={sum(grad_dists) / len(grad_dists):.6f}")

        # CP is numerically exact in the FORWARD (output/loss dist == 0). The
        # only divergence is bf16 backward accumulation: bias grads whose post-
        # cancellation magnitude is tiny (e.g. ``cross_attn.key.bias``) round
        # differently when the sequence is split (cp>1 accumulates over a shard
        # via bf16 FlashAttention, then SUM-all-reduces) vs accumulated over the
        # full sequence on one device (cp=1). Measured: max ~0.023 on a single
        # bias, mean ~0.0015. fp32-ing the cross-rank all-reduce does NOT move it
        # (the drift is upstream, inside the bf16 attention backward), and an
        # fp32 reference run is blocked by FlashAttention (fp16/bf16 only). So we
        # gate on a per-param max (catches a real logic regression: an unreduced
        # param diverges ~50-100%, far above this bar) AND a mean (catches a
        # systematic error that a single-param max could miss).
        max_ok = max_grad_dist is not None and max_grad_dist < 0.03
        mean_grad_dist = (sum(grad_dists) / len(grad_dists)) if grad_dists else None
        mean_ok = mean_grad_dist is not None and mean_grad_dist < 0.005
        if grad_dists and max_ok and mean_ok:
            bwd_msg = DPO_CP_BWD_SUCCESS
        else:
            bwd_msg = f"{DPO_CP_BWD_FAIL} max_grad_dist={max_grad_dist if grad_dists else 'N/A'} param={max_grad_name if grad_dists else 'N/A'} mean_grad_dist={mean_grad_dist if grad_dists else 'N/A'}"
        self.assertTrue(grad_dists and max_ok and mean_ok, bwd_msg)
