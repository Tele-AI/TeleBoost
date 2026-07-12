# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025-2026 TeleAI and the TeleBoost contributors.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
from collections.abc import Callable
import logging
from typing import Optional

import torch
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils.timer import NoopTimer
from megatron.core import mpu
from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
)
from megatron.core.transformer.module import MegatronModule

from teleboost.engines.teletron import get_args, get_num_microbatches


def _build_param_groups(model, config, no_weight_decay_cond, scale_lr_cond, lr_mult):
    """Adapter for megatron-core 0.16's ``_get_param_groups``.

    mc 0.16.x signature is ``_get_param_groups(model_chunks, config,
    config_overrides)`` — ``config: OptimizerConfig`` carries
    ``lr/min_lr/decoupled_lr/decoupled_min_lr`` directly; per-param
    overrides (bias + length-1 → no weight decay) come from
    ``config_overrides=None`` which mc resolves via
    ``get_standard_config_overrides``. teleboost callers (DPO recipes in
    ``megatron_wan._build_optimizer``) don't pass ``no_weight_decay_cond
    / scale_lr_cond / lr_mult``, so this adapter drops them and lets mc
    apply its standard overrides.

    mc's ``_get_param_groups`` runs an unguarded world-group
    ``torch.distributed.all_gather_object`` to align param keys across
    DP ranks (distributed-checkpoint compat). Under
    ``--distributed-vae`` the producer ranks exit early and never hit
    this code path, so the world-group all_gather deadlocks. Scope the
    call to consumer-only DP group by patching
    ``torch.distributed.{all_gather_object, get_world_size}`` for the
    duration of ``_get_param_groups``.
    """
    import torch.distributed as _dist

    _consumer_group = mpu.get_data_parallel_group(with_context_parallel=True)
    _orig_get_world_size = _dist.get_world_size
    _orig_all_gather_object = _dist.all_gather_object

    def _patched_get_world_size(group=None):
        if group is None:
            group = _consumer_group
        return _orig_get_world_size(group=group)

    def _patched_all_gather_object(object_list, obj, group=None, **kwargs):
        if group is None:
            group = _consumer_group
        return _orig_all_gather_object(object_list, obj, group=group, **kwargs)

    _dist.get_world_size = _patched_get_world_size
    _dist.all_gather_object = _patched_all_gather_object
    try:
        return _get_param_groups(
            model_chunks=model,
            config=config,
            config_overrides=None,
        )
    finally:
        _dist.get_world_size = _orig_get_world_size
        _dist.all_gather_object = _orig_all_gather_object


try:
    from apex.optimizers import FusedAdam as Adam
    from apex.optimizers import FusedSGD as SGD

    _USING_APEX_OPTIMIZER = True
except ImportError:
    from torch.optim import AdamW as Adam
    from torch.optim import SGD

    _USING_APEX_OPTIMIZER = False

logger = logging.getLogger(__name__)


class SchedulerMixin:
    def get_optimizer_for_zero2(
        self,
        config: OptimizerConfig,
        model: list[MegatronModule],
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ):
        args = get_args()
        if not _USING_APEX_OPTIMIZER:
            logger.warning("NVIDIA Apex is unavailable; using torch.optim for the DPO ZeRO optimizer. Install a CUDA-matched Apex build only for the fused-optimizer acceleration path.")
        param_groups = _build_param_groups(
            model,
            config,
            no_weight_decay_cond,
            scale_lr_cond,
            lr_mult,
        )
        if config.optimizer == "adam":
            base_optimizer = Adam(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
            )
        elif config.optimizer == "sgd":
            base_optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
        else:
            raise Exception("{} optimizer is not supported.".format(config.optimizer))
        param_names = {param: name for name, param in model[0].named_parameters()}
        timers = NoopTimer()
        # optimizer_params dict (required by DeepSpeedZeroOptimizer.__init__).
        optimizer_params = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "betas": [config.adam_beta1, config.adam_beta2],
            "eps": config.adam_eps,
        }
        # communication_data_type must match model param dtype, otherwise the
        # ipg_buckets dict (built at __init__ from {communication_data_type})
        # has the wrong key and the first autograd hook fires KeyError.
        # DeepSpeed only auto-discovers per-param dtypes when wrapped in
        # DeepSpeedEngine + autocast; teleboost uses the optimizer alone, so
        # we set it explicitly.
        if args.bf16:
            comm_dtype = torch.bfloat16
        elif args.fp16:
            comm_dtype = torch.float16
        else:
            comm_dtype = torch.float32
        optimizer = DeepSpeedZeroOptimizer(
            base_optimizer,
            param_names,
            timers=timers,
            optimizer_params=optimizer_params,
            static_loss_scale=1.0,
            dynamic_loss_scale=False,
            dynamic_loss_args=None,
            clip_grad=args.clip_grad,
            contiguous_gradients=True,
            reduce_bucket_size=500000000,
            use_multi_rank_bucket_allreduce=True,
            allgather_bucket_size=500000000,
            dp_process_group=mpu.get_data_parallel_group(with_context_parallel=True),
            expert_parallel_group=None,
            expert_data_parallel_group=None,
            communication_data_type=comm_dtype,
            reduce_scatter=True,
            overlap_comm=False,
            offload_optimizer_config=None,
            mpu=None,
            postscale_gradients=True,
            gradient_predivide_factor=1.0,
            gradient_accumulation_steps=get_num_microbatches(),
            ignore_unused_parameters=True,
            partition_grads=True,
            round_robin_gradients=False,
            has_moe_layers=False,
            fp16_master_weights_and_gradients=False,
            elastic_checkpoint=False,
        )
        return optimizer
