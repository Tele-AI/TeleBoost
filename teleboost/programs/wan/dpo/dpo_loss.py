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
import glob
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu

from teleboost.models.flow_match import FlowMatchScheduler
from teleboost.training.utils import average_losses_across_data_parallel_group
from teleboost.engines.teletron import get_timers, set_config
from teleboost.engines.teletron.parallel_state import get_tensor_and_context_parallel_src_rank


def dpo_loss_func(output_tensor):
    # output_tensor: [loss_reject_scaled, loss_chosen_scaled, loss_reject, loss_chosen, dpo_loss]
    # The two scaled losses are kept as a list so backward_step can backprop each separately.
    if not isinstance(output_tensor, list | tuple):
        output_tensor = [output_tensor]

    def _to_scalar(t):
        return t if t.dim() == 0 else t.mean()

    loss_for_backward = [_to_scalar(t) for t in output_tensor[:2]]

    if len(output_tensor) >= 5:
        dpo_loss_mean = _to_scalar(output_tensor[4])
    else:
        dpo_loss_mean = sum(loss_for_backward).detach()

    if len(output_tensor) >= 4:
        loss_reject_mean = _to_scalar(output_tensor[2])
        loss_chosen_mean = _to_scalar(output_tensor[3])
        averaged = average_losses_across_data_parallel_group([dpo_loss_mean.detach(), loss_reject_mean.detach(), loss_chosen_mean.detach()])
        loss_dict = {
            "loss": averaged[0],
            "loss_reject_mean": averaged[1],
            "loss_chosen_mean": averaged[2],
        }
    else:
        averaged = average_losses_across_data_parallel_group([dpo_loss_mean.detach()])
        loss_dict = {"loss": averaged[0]}

    return loss_for_backward, loss_dict


def _compute_single_loss(
    latents,
    context,
    clip_feature,
    y,
    flow_scheduler,
    model,
    timestep,
    noise,
    return_debug=False,
):
    training_target = flow_scheduler.training_target(latents, noise, timestep)
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    loss_weight = flow_scheduler.training_weight(timestep)

    output_tensor_list = model(
        x=noisy_latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
    )
    loss = torch.nn.functional.mse_loss(output_tensor_list.float(), training_target.float())
    loss_wo_w = loss
    loss = loss * loss_weight

    first_frame_loss = loss_wo_w.new_zeros(())

    if return_debug:
        debug = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noisy_latents,
            "training_target": training_target,
            "loss_weight": loss_weight,
            "noise_pred": output_tensor_list.detach(),
        }
        return loss, loss_wo_w, first_frame_loss, debug

    return loss, loss_wo_w, first_frame_loss


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _to_cuda(obj, device=None):
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_cuda(v, device=device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cuda(v, device=device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cuda(v, device=device) for v in obj)
    return obj


def _dtype_from_string(dtype_str: str):
    name = dtype_str.split(".", 1)[-1] if dtype_str.startswith("torch.") else dtype_str
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    return getattr(torch, name)


def _broadcast_tensor_from_src(tensor, src_rank, group, device):
    if tensor is None and torch.distributed.get_rank() == src_rank:
        meta = {"is_none": True}
    elif torch.distributed.get_rank() == src_rank:
        meta = {
            "is_none": False,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    else:
        meta = {"is_none": True}

    meta_list = [meta]
    dist.broadcast_object_list(meta_list, src=src_rank, group=group)
    meta = meta_list[0]
    if meta.get("is_none"):
        return None

    dtype = _dtype_from_string(meta["dtype"])
    if torch.distributed.get_rank() == src_rank:
        out = tensor.to(device=device, non_blocking=True)
    else:
        out = torch.empty(meta["shape"], dtype=dtype, device=device)

    dist.broadcast(out, src=src_rank, group=group)
    return out


def _set_nested(root, path, value):
    node = root
    for key in path[:-1]:
        if key not in node:
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def _broadcast_saved_payload(payload, device):
    cp_world = mpu.get_tensor_and_context_parallel_world_size()
    if cp_world <= 1:
        return _to_cuda(payload, device=device)

    src_rank = get_tensor_and_context_parallel_src_rank()
    group = mpu.get_tensor_and_context_parallel_group()
    tensor_paths = [
        ("context",),
        ("chosen", "clip_feature"),
        ("chosen", "y"),
        ("chosen", "latents"),
        ("chosen", "noise"),
        ("chosen", "noisy_latents"),
        ("chosen", "noise_pred"),
        ("chosen", "training_target"),
        ("chosen", "loss_weight"),
        ("chosen", "timestep"),
        ("rejected", "timestep"),
        ("rejected", "clip_feature"),
        ("rejected", "y"),
        ("rejected", "latents"),
        ("rejected", "noise"),
        ("rejected", "noisy_latents"),
        ("rejected", "noise_pred"),
        ("rejected", "training_target"),
        ("rejected", "loss_weight"),
    ]

    out = {}
    for path in tensor_paths:
        if payload is not None:
            node = payload
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            tensor = node
        else:
            tensor = None
        broadcasted = _broadcast_tensor_from_src(tensor, src_rank, group, device)
        _set_nested(out, path, broadcasted)
    return out


def _compute_single_loss_from_saved(
    noisy_latents,
    training_target,
    loss_weight,
    context,
    clip_feature,
    y,
    model,
    timestep,
    expected_noise_pred=None,
    compare_name=None,
    rtol=1e-5,
    atol=1e-8,
):
    tp_cp_src_rank = get_tensor_and_context_parallel_src_rank()
    if dist.get_rank() == tp_cp_src_rank:
        inputs_info = {
            "x": (str(noisy_latents.dtype), list(noisy_latents.shape)),
            "timestep": (str(timestep.dtype), list(timestep.shape)),
            "context": (str(context.dtype), list(context.shape)),
            "clip_feature": (
                str(clip_feature.dtype) if torch.is_tensor(clip_feature) else str(type(clip_feature).__name__),
                list(clip_feature.shape) if torch.is_tensor(clip_feature) else None,
            ),
            "y": (
                str(y.dtype) if torch.is_tensor(y) else str(type(y).__name__),
                list(y.shape) if torch.is_tensor(y) else None,
            ),
        }
        print(f"[DPO input-dtype] rank={dist.get_rank()} {compare_name or 'noise_pred'} inputs={inputs_info}")
    output_tensor_list = model(
        x=noisy_latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
    )
    if expected_noise_pred is not None:
        mismatch = _compare_input_tensor(
            compare_name or "noise_pred",
            expected_noise_pred,
            output_tensor_list,
            rtol=rtol,
            atol=atol,
        )
        if mismatch is not None:
            print(f"[DPO output-check] rank={dist.get_rank()} mismatch={mismatch}")
        elif dist.get_rank() == tp_cp_src_rank:
            print(f"[DPO output-check] rank={dist.get_rank()} {compare_name or 'noise_pred'} ok")
    loss = torch.nn.functional.mse_loss(output_tensor_list.float(), training_target.float())
    loss_wo_w = loss
    loss = loss * loss_weight

    first_frame_loss = loss_wo_w.new_zeros(())

    return loss, loss_wo_w, first_frame_loss


def _compare_input_tensor(
    name,
    expected,
    actual,
    rtol=1e-5,
    atol=1e-8,
    cast_mode="float32",  # "none" | "float32" | "to_actual" | "to_expected"
):
    if expected is None and actual is None:
        return None
    if expected is None or actual is None:
        return {"name": name, "reason": "one is None"}

    if not torch.is_tensor(expected) or not torch.is_tensor(actual):
        if expected == actual:
            return None
        return {"name": name, "reason": "non-tensor mismatch"}

    if expected.shape != actual.shape:
        return {
            "name": name,
            "reason": "shape mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }

    dtype_mismatch = expected.dtype != actual.dtype
    device_mismatch = expected.device != actual.device

    # ---- choose comparison dtype/device ----
    e = expected
    a = actual
    # move to same device for compare (cheap)
    if device_mismatch:
        e = e.to(device=a.device)

    # cast for value-compare
    if cast_mode == "none":
        # keep original dtypes; if dtype mismatch, compare will be meaningless for allclose in many cases
        pass
    elif cast_mode == "float32":
        e = e.float()
        a = a.float()
    elif cast_mode == "to_actual":
        e = e.to(dtype=a.dtype)
    elif cast_mode == "to_expected":
        a = a.to(dtype=e.dtype)
    else:
        raise ValueError(f"Unknown cast_mode={cast_mode}")

    # If they are exactly equal after casting
    if torch.equal(e, a):
        if dtype_mismatch or device_mismatch:
            return {
                "name": name,
                "reason": "equal_after_cast",
                "expected_dtype": str(expected.dtype),
                "actual_dtype": str(actual.dtype),
                "cast_mode": cast_mode,
            }
        return None

    allclose = torch.allclose(e, a, rtol=rtol, atol=atol)
    if allclose:
        # still report dtype mismatch if you care
        if dtype_mismatch or device_mismatch:
            return {
                "name": name,
                "reason": "allclose_after_cast",
                "expected_dtype": str(expected.dtype),
                "actual_dtype": str(actual.dtype),
                "cast_mode": cast_mode,
            }
        return None

    diff = (e - a).abs()
    out = {
        "name": name,
        "reason": "value mismatch",
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rtol": rtol,
        "atol": atol,
        "cast_mode": cast_mode,
    }
    if dtype_mismatch:
        out["expected_dtype"] = str(expected.dtype)
        out["actual_dtype"] = str(actual.dtype)
    if device_mismatch:
        out["expected_device"] = str(expected.device)
        out["actual_device"] = str(actual.device)
    return out


def _summarize_scheduler(scheduler, max_values=20):
    if scheduler is None:
        return {}
    summary = {"class": scheduler.__class__.__name__}
    sigmas = getattr(scheduler, "sigmas", None)
    if isinstance(sigmas, torch.Tensor):
        summary["num_inference_steps"] = int(sigmas.numel())
    for key, value in scheduler.__dict__.items():
        if key.startswith("_"):
            continue
        if torch.is_tensor(value):
            t = value.detach().cpu()
            entry = {"shape": list(t.shape), "dtype": str(t.dtype)}
            if t.numel() <= max_values:
                entry["values"] = [float(x) for x in t.flatten().tolist()]
            elif t.numel() > 0:
                flat = t.float().flatten()
                entry["stats"] = {
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                    "mean": float(flat.mean()),
                    "std": float(flat.std()),
                }
                entry["preview"] = [float(x) for x in flat[:max_values].tolist()]
            summary[key] = entry
        elif isinstance(value, int | float | bool) or value is None:
            summary[key] = value
        elif isinstance(value, str):
            summary[key] = value
        elif isinstance(value, list | tuple):
            preview = list(value[:max_values])
            summary[key] = {"len": len(value), "preview": preview}
        elif isinstance(value, dict):
            summary[key] = {"keys": list(value.keys())}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def _find_latest_dump_dir(sample_dir: str, stage: str, branch: str):
    pat = os.path.join(sample_dir, f"*__{stage}__{branch}")
    cands = sorted([d for d in glob.glob(pat) if os.path.isdir(d)])
    return cands[-1] if cands else None


def _load_tensor_dump_root(dump_dir: str):
    pt = os.path.join(dump_dir, "data", "root.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError(f"Missing tensor root.pt: {pt}")
    return torch.load(pt, map_location="cpu", weights_only=True)


def _load_value_from_root_dict(root_dict_dir: str, key: str):
    pt = os.path.join(root_dict_dir, f"{key}.pt")
    js = os.path.join(root_dict_dir, f"{key}.json")

    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu", weights_only=True)
    if os.path.exists(js):
        with open(js, encoding="utf-8") as f:
            return json.load(f)["value"]
    raise KeyError(f"Key '{key}' not found under {root_dict_dir}")


def _load_stage_tensor(dump_root: str, pair_id: int, stage: str, branch: str):
    sample_dir = os.path.join(dump_root, f"sample_{int(pair_id)}")
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"sample_dir not found: {sample_dir}")

    d = _find_latest_dump_dir(sample_dir, stage=stage, branch=branch)
    if d is None:
        raise FileNotFoundError(f"dump not found: sample={pair_id} stage={stage} branch={branch}")
    return _load_tensor_dump_root(d)


def _load_stage_dict(dump_root: str, pair_id: int, stage: str, branch: str):
    sample_dir = os.path.join(dump_root, f"sample_{int(pair_id)}")
    d = _find_latest_dump_dir(sample_dir, stage=stage, branch=branch)
    if d is None:
        raise FileNotFoundError(f"dump not found: sample={pair_id} stage={stage} branch={branch}")
    root_dict = os.path.join(d, "data", "root__dict")
    if not os.path.isdir(root_dict):
        raise FileNotFoundError(f"Missing root__dict: {root_dict}")
    return root_dict


def _load_saved_payload(args, device):
    dump_root = getattr(args, "diffsynth_dump_root", None) or getattr(args, "save_inputs_dir", None) or getattr(args, "save_dumps_dir", None)
    if dump_root is None:
        raise ValueError("Please set args.diffsynth_dump_root, e.g. dump_dataset/pid_xxx")

    pair_id = int(getattr(args, "saved_pair_id", 0))
    dp_rank = int(mpu.get_data_parallel_rank())
    tp_cp_src_rank = get_tensor_and_context_parallel_src_rank()

    def _pick_optional(root, candidates):
        for k in candidates:
            try:
                return _load_value_from_root_dict(root, k)
            except Exception:
                pass
        return None

    if torch.distributed.get_rank() == tp_cp_src_rank:
        # 1) dict roots
        chosen_root = _load_stage_dict(dump_root, pair_id, stage="chosen_inputs", branch="chosen")
        rejected_root = _load_stage_dict(dump_root, pair_id, stage="rejected_inputs", branch="rejected")

        # context: if it is considered mandatory, do not treat it as optional.
        context = _pick_optional(chosen_root, ["context", "prompt_emb", "prompt_embedding"])

        chosen_clip_feature = _pick_optional(chosen_root, ["clip_feature", "chosen_clip_feature", "clip_feat"])
        reject_clip_feature = _pick_optional(rejected_root, ["clip_feature", "reject_clip_feature", "rejected_clip_feature", "clip_feat"])

        chosen_y = _pick_optional(chosen_root, ["y", "chosen_y"])
        reject_y = _pick_optional(rejected_root, ["y", "reject_y", "rejected_y"])

        # 2) tensors from training_loss stages (normally required; the loss cannot be reproduced without them)
        timestep_chosen = _load_stage_tensor(dump_root, pair_id, "training_loss__timesteps", "chosen")
        timestep_rejected = _load_stage_tensor(dump_root, pair_id, "training_loss__timesteps", "rejected")

        chosen_noisy_latents = _load_stage_tensor(dump_root, pair_id, "training_loss__noisy_latents", "chosen")
        reject_noisy_latents = _load_stage_tensor(dump_root, pair_id, "training_loss__noisy_latents", "rejected")

        chosen_training_target = _load_stage_tensor(dump_root, pair_id, "training_loss__training_target", "chosen")
        reject_training_target = _load_stage_tensor(dump_root, pair_id, "training_loss__training_target", "rejected")

        chosen_loss_weight = _load_stage_tensor(dump_root, pair_id, "training_loss__loss_weight", "chosen")
        reject_loss_weight = _load_stage_tensor(dump_root, pair_id, "training_loss__loss_weight", "rejected")

        # optional compare tensors
        chosen_noise_pred = None
        reject_noise_pred = None
        try:
            chosen_noise_pred = _load_stage_tensor(dump_root, pair_id, "training_loss__noise_pred", "chosen")
        except Exception:
            pass
        try:
            reject_noise_pred = _load_stage_tensor(dump_root, pair_id, "training_loss__noise_pred", "rejected")
        except Exception:
            pass

        payload = {
            "meta": {"dp_rank": dp_rank, "pair_id": pair_id, "dump_root": dump_root},
            "context": context,
            "chosen": {
                "clip_feature": chosen_clip_feature,  # may be None
                "y": chosen_y,  # may be None
                "timestep": timestep_chosen,
                "noisy_latents": chosen_noisy_latents,
                "training_target": chosen_training_target,
                "loss_weight": chosen_loss_weight,
                "noise_pred": chosen_noise_pred,  # optional
            },
            "rejected": {
                "clip_feature": reject_clip_feature,  # may be None
                "y": reject_y,  # may be None
                "timestep": timestep_rejected,
                "noisy_latents": reject_noisy_latents,
                "training_target": reject_training_target,
                "loss_weight": reject_loss_weight,
                "noise_pred": reject_noise_pred,  # optional
            },
        }

        print(f"[DPO load from dumper] dp_rank={dp_rank} pair_id={pair_id} root={dump_root}")
        print(f"[DPO load] keys={sorted(payload.keys())}")
        print(f"[DPO load] chosen keys={sorted(payload['chosen'].keys())}")
        print(f"[DPO load] rejected keys={sorted(payload['rejected'].keys())}")

    else:
        payload = None

    # broadcast to all ranks
    payload = _broadcast_saved_payload(payload, device=device)

    # move timesteps to desired dtype/device here (after broadcast)
    timestep_c = payload["chosen"]["timestep"].to(dtype=torch.bfloat16, device=device)
    timestep_r = payload["rejected"]["timestep"].to(dtype=torch.bfloat16, device=device)
    payload["chosen"]["timestep"] = timestep_c
    payload["rejected"]["timestep"] = timestep_r

    # return as you use in forward_step
    context = payload["context"]
    chosen_clip_feature = payload["chosen"]["clip_feature"]  # can be None
    reject_clip_feature = payload["rejected"]["clip_feature"]  # can be None
    chosen_y = payload["chosen"]["y"]  # can be None
    reject_y = payload["rejected"]["y"]  # can be None

    return payload, context, chosen_clip_feature, reject_clip_feature, chosen_y, reject_y, timestep_c, timestep_r


def _load_batch_inputs(batch, chosen_key, rejected_key):
    context = batch["context"]  # shared text context
    chosen_latents = batch[chosen_key]["latents"]
    reject_latents = batch[rejected_key]["latents"]
    chosen_clip_feature = batch[chosen_key].get("img_clip_feature", batch[chosen_key].get("clip_feature"))
    reject_clip_feature = batch[rejected_key].get("img_clip_feature", batch[rejected_key].get("clip_feature"))
    chosen_y = batch[chosen_key].get("img_emb_y")
    reject_y = batch[rejected_key].get("img_emb_y")
    return (
        context,
        chosen_latents,
        reject_latents,
        chosen_clip_feature,
        reject_clip_feature,
        chosen_y,
        reject_y,
    )


def _broadcast_tensor(input_tensor: torch.Tensor):
    tp_cp_src_rank = get_tensor_and_context_parallel_src_rank()
    if mpu.get_tensor_and_context_parallel_world_size() > 1:
        dist.broadcast(
            input_tensor,
            tp_cp_src_rank,
            group=mpu.get_tensor_and_context_parallel_group(),
        )


def _sample_timestep(flow_scheduler, diffusion_config, device):
    min_timestep_boundary = int(diffusion_config.get("min_timestep_boundary") * flow_scheduler.num_train_timesteps)
    max_timestep_boundary = int(diffusion_config.get("max_timestep_boundary") * flow_scheduler.num_train_timesteps)
    timestep_range = [min_timestep_boundary, max_timestep_boundary]
    timestep_id = torch.randint(timestep_range[0], timestep_range[1], (1,))
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16,
        device=device,
    )
    return timestep, timestep_id, timestep_range


def _generate_noise(chosen_latents, reject_latents, base_seed, curr_iter):
    # Following the official Diffusion-DPO implementation's pairing scheme
    # (the paper samples eps independently; the reference code shares it):
    # same-shape preference pairs share noise so the margin reflects model
    # quality, not an independent noise draw.
    share = chosen_latents.shape == reject_latents.shape
    if base_seed is not None:
        seed_chosen = int(base_seed) + curr_iter * 2
        gen_chosen = torch.Generator(device=chosen_latents.device).manual_seed(seed_chosen)
        noise_chosen = torch.randn(
            chosen_latents.shape,
            device=chosen_latents.device,
            dtype=chosen_latents.dtype,
            generator=gen_chosen,
        )
        if share:
            noise_reject, seed_reject = noise_chosen, seed_chosen
        else:
            seed_reject = int(base_seed) + curr_iter * 2 + 1
            gen_reject = torch.Generator(device=reject_latents.device).manual_seed(seed_reject)
            noise_reject = torch.randn(
                reject_latents.shape,
                device=reject_latents.device,
                dtype=reject_latents.dtype,
                generator=gen_reject,
            )
        if torch.distributed.get_rank() == 0:
            print(f"[Rank 0] noise seeds: chosen={seed_chosen}, reject={seed_reject}" + (" (shared)" if share else ""))
    else:
        noise_chosen = torch.randn_like(chosen_latents)
        noise_reject = noise_chosen if share else torch.randn_like(reject_latents)

    return noise_chosen, noise_reject


def forward_step(data_iterator, model, time_step=None):
    # Args come from teleboost's global namespace (set by ``parse_args``
    # in the standalone launcher's __main__ block, or by
    # ``teleboost.programs.wan.dpo.args_adapter.build_teletron_args`` in the verl
    # plug-in path). Earlier this module relied on a module-level
    # ``args`` set inside the ``if __name__ == "__main__":`` block —
    # that breaks when forward_step is imported by another driver
    # (verl recipe), which is why the call below replaces that pattern.
    from teleboost.engines.teletron import get_args

    args = get_args()

    flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
    flow_scheduler.set_timesteps(1000, training=True)

    dataset_config = set_config().get("dataset", {})
    chosen_key = dataset_config.get("chosen_video_key", "chosen")
    rejected_key = dataset_config.get("rejected_video_key", "rejected")

    timers = get_timers()
    timers.start_timer("get-data-time")
    batch = next(data_iterator)
    timers.stop_timer("get-data-time")

    use_saved_inputs = bool(getattr(args, "use_saved_inputs", False))
    save_dumps = bool(getattr(args, "save_dumps", False))

    if use_saved_inputs:
        # from diffsynth dumps
        (
            payload,
            context,
            chosen_clip_feature,
            reject_clip_feature,
            chosen_y,
            reject_y,
            timestep_c,
            timestep_r,
        ) = _load_saved_payload(args, device=torch.cuda.current_device())
    else:
        payload = None
        (
            context,
            chosen_latents,
            reject_latents,
            chosen_clip_feature,
            reject_clip_feature,
            chosen_y,
            reject_y,
        ) = _load_batch_inputs(batch, chosen_key, rejected_key)
    # =========================
    # timestep sampling
    # =========================
    diffusion_config = set_config().get("model_config", {}).get("training", {}).get("diffusion", {})

    timestep_range = None
    timestep_id_c = None
    timestep_id_r = None
    if not use_saved_inputs:
        # Official-implementation pairing scheme: chosen and rejected share the
        # SAME timestep (and noise, below) so the per-pair margin
        # (loss_reject - loss_chosen) reflects relative model quality, not a
        # timestep/noise gap. Independent draws inflate the preference-signal
        # variance and bias the detached gate eta. Sample once, reuse for both.
        timestep_c, timestep_id_c, timestep_range = _sample_timestep(
            flow_scheduler,
            diffusion_config,
            device=torch.cuda.current_device(),
        )
        timestep_r, timestep_id_r = timestep_c, timestep_id_c

    _broadcast_tensor(timestep_c)
    _broadcast_tensor(timestep_r)
    if dist.get_rank() == get_tensor_and_context_parallel_src_rank():
        print(f"[Rank {dist.get_rank()}] timestep_c={timestep_c}, id_c={timestep_id_c}")
        print(f"[Rank {dist.get_rank()}] timestep_r={timestep_r}, id_r={timestep_id_r}")
    # =========================
    # noise (chosen/reject share noise when same-shape; see _generate_noise)
    # =========================
    if not use_saved_inputs:
        base_seed = getattr(args, "noise_seed", None)
        curr_iter = int(getattr(args, "curr_iteration", 0))
        noise_chosen, noise_reject = _generate_noise(
            chosen_latents,
            reject_latents,
            base_seed=base_seed,
            curr_iter=curr_iter,
        )
        _broadcast_tensor(noise_chosen)
        _broadcast_tensor(noise_reject)
    # =========================
    # forward & loss
    # =========================
    return_debug = bool(save_dumps)

    if use_saved_inputs:
        chosen_noisy_latents = payload["chosen"]["noisy_latents"]
        chosen_training_target = payload["chosen"]["training_target"]
        chosen_loss_weight = payload["chosen"]["loss_weight"]
        chosen_noise_pred = payload["chosen"].get("noise_pred")
        reject_noisy_latents = payload["rejected"]["noisy_latents"]
        reject_training_target = payload["rejected"]["training_target"]
        reject_loss_weight = payload["rejected"]["loss_weight"]
        reject_noise_pred = payload["rejected"].get("noise_pred")

    def _run_branch(tag, latents, clip_feature, y, timestep, noise, saved_noisy_latents=None, saved_training_target=None, saved_loss_weight=None, saved_noise_pred=None):
        if use_saved_inputs:
            loss, loss_wo_w, first_frame = _compute_single_loss_from_saved(
                saved_noisy_latents,
                saved_training_target,
                saved_loss_weight,
                context,
                clip_feature,
                y,
                model,
                timestep,
                expected_noise_pred=saved_noise_pred,
                compare_name=f"{tag}.noise_pred",
                rtol=float(getattr(args, "compare_losses_rtol", 1e-5)),
                atol=float(getattr(args, "compare_losses_atol", 1e-8)),
            )
            return loss, loss_wo_w, first_frame, None
        if return_debug:
            loss, loss_wo_w, first_frame, debug = _compute_single_loss(
                latents,
                context,
                clip_feature,
                y,
                flow_scheduler,
                model,
                timestep,
                noise,
                return_debug=True,
            )
            return loss, loss_wo_w, first_frame, debug
        loss, loss_wo_w, first_frame = _compute_single_loss(
            latents,
            context,
            clip_feature,
            y,
            flow_scheduler,
            model,
            timestep,
            noise,
        )
        return loss, loss_wo_w, first_frame, None

    loss_chosen, loss_wo_w_chosen, first_frame_loss_chosen, debug_chosen = _run_branch(
        "chosen",
        chosen_latents,
        chosen_clip_feature,
        chosen_y,
        timestep_c,
        None if use_saved_inputs else noise_chosen,
        saved_noisy_latents=chosen_noisy_latents if use_saved_inputs else None,
        saved_training_target=chosen_training_target if use_saved_inputs else None,
        saved_loss_weight=chosen_loss_weight if use_saved_inputs else None,
        saved_noise_pred=chosen_noise_pred if use_saved_inputs else None,
    )

    loss_reject, loss_wo_w_reject, first_frame_loss_reject, debug_reject = _run_branch(
        "rejected",
        reject_latents,
        reject_clip_feature,
        reject_y,
        timestep_r,
        None if use_saved_inputs else noise_reject,
        saved_noisy_latents=reject_noisy_latents if use_saved_inputs else None,
        saved_training_target=reject_training_target if use_saved_inputs else None,
        saved_loss_weight=reject_loss_weight if use_saved_inputs else None,
        saved_noise_pred=reject_noise_pred if use_saved_inputs else None,
    )

    beta = float(set_config()["model_config"]["dit"]["train"]["dpo"]["beta"])

    advantage = (loss_reject - loss_chosen).clamp(-20, 20)
    # Detached surrogate: only advantage's value feeds coeff, no grad through it.
    with torch.no_grad():
        coeff = beta * torch.sigmoid(-beta * advantage) / advantage.numel()

    # L = -logσ(β·adv), adv = loss_reject - loss_chosen
    # dL/d(loss_reject) = -coeff,  dL/d(loss_chosen) = +coeff
    loss_reject_scaled = (-coeff * loss_reject).sum()
    loss_chosen_scaled = (coeff * loss_chosen).sum()

    dpo_loss_for_log = (-F.logsigmoid(beta * advantage)).mean().detach()

    if return_debug and not args.test_with_pseudo_data and not use_saved_inputs:
        interval = max(1, int(getattr(args, "save_dumps_interval", 1)))
        curr_iter = int(getattr(args, "curr_iteration", 0))
        if curr_iter % interval == 0:
            save_dir = getattr(args, "save_dumps_dir", None) or getattr(args, "save_inputs_dir", None) or f"../test_data/saved_inputs_{args.test_resolution}"
            os.makedirs(save_dir, exist_ok=True)
            dp_rank = int(mpu.get_data_parallel_rank())
            payload = {
                "meta": {
                    "iter": curr_iter,
                    "rank": int(torch.distributed.get_rank()),
                    "dp_rank": dp_rank,
                },
                "scheduler_params": _summarize_scheduler(flow_scheduler),
                "context": context,
                "chosen": {
                    "clip_feature": chosen_clip_feature,
                    "y": chosen_y,
                    "timestep_c": timestep_c,
                    **debug_chosen,
                },
                "rejected": {
                    "clip_feature": reject_clip_feature,
                    "y": reject_y,
                    "timestep_r": timestep_r,
                    **debug_reject,
                },
                "losses": {
                    "loss_chosen": loss_chosen.detach(),
                    "loss_reject": loss_reject.detach(),
                    "dpo_loss": dpo_loss_for_log.detach(),
                    "loss_wo_w_chosen": loss_wo_w_chosen.detach(),
                    "loss_wo_w_reject": loss_wo_w_reject.detach(),
                    "first_frame_loss_chosen": first_frame_loss_chosen.detach(),
                    "first_frame_loss_reject": first_frame_loss_reject.detach(),
                },
            }
            save_path = os.path.join(save_dir, f"dpo_inputs_iter{curr_iter}_rank{dp_rank}.pt")
            torch.save(_detach_to_cpu(payload), save_path)

    return [
        loss_reject_scaled,
        loss_chosen_scaled,
        loss_reject.detach(),
        loss_chosen.detach(),
        dpo_loss_for_log.detach(),
    ], dpo_loss_func
