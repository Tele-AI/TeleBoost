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
"""Wan video-diffusion engine registered against verl's
``EngineRegistry`` with ``backend="megatron"``.

Wires the standalone Wan + megatron-LM init (currently in
``teleboost/training/trainer.py`` + ``teleboost/engines/teletron/megatron_adaptor.py``) into
verl's ``TrainingWorker.engine`` plug point. Subclassing
``MegatronEngine`` and overriding the three plug-points verl's own
language-model engine overrides (``prepare_model_inputs``,
``prepare_model_outputs``, ``forward_step``) plus the two megatron
init steps that hardcode ``AutoBridge`` against HuggingFace
architectures (``_build_tf_config``, ``_build_megatron_module``).

real overrides for ``_init_device_mesh``,
``_build_tf_config``, ``_build_megatron_module`` — bypass mbridge
(Wan is not HF-registered), reuse the TeleTron ``model_provider`` and the
teleboost.engines.teletron.megatron_adaptor TCP wrap. ``forward_step`` / ``train_batch``
(split-DPO dispatch) / ``prepare_model_inputs/outputs`` still
``NotImplementedError`` — they land in alongside the DPO
loss + the WanBridge weight bridge.

Source-of-truth map:

  _build_tf_config         <- teleboost.engines.teletron.megatron_adaptor.install
  _build_megatron_module   <- teleboost.training.trainer.Trainer.__init__
                              (lines ~70–250: self.model_provider + setup_model_and_optimizer)
  prepare_model_inputs     <- teleboost.training.trainer.forward_step
                              (extracts latents/timesteps/prompt_embeddings from batch)
  prepare_model_outputs    <- teleboost.training.trainer.forward_step return
                              (predicted noise / x0 / log_probs depending on DPO mode)
  forward_step             <- teleboost.training.trainer.Trainer.train_step
                              (megatron pipeline forward+backward via get_forward_backward_func)
  train_batch              <- teleboost.training.trainer.Trainer.train_step (use_zero2 path)
                              + teleboost.training.utils.deepspeed_forward_backward
                              + teleboost.training.utils.deepspeed_backward_step

Feature-parity hard constraints (standalone ``teleboost/programs/wan/dpo/train_dpo.sh``
behaviour must be preserved bit-for-bit, no shortcuts):

  1. Split DPO backward (use_zero2=True path). Standalone path:
     teleboost/training/utils.py:553 deepspeed_backward_step — when the
     loss object is a list/tuple, each loss runs its own
     ``zero_optimizer.backward(t) + overlapping_partition_gradients_reduce_epilogue()``
     so per-backward gradients are reduce-scattered to my-shard slices
     immediately, freeing the full per-layer gradient tensors before
     the next backward starts (~½ peak mem). Math equivalence to
     single ``backward(sum-of-losses)`` empirically verified within
     bf16 ULP (max|d| < 3e-4 on 5-iter Wan). Hard dep on
     ``deepspeed==0.17.5`` (multi-call epilogue); 0.17.6+ replaced it
     with an all_grad_tensors state machine. Migration: override
     ``train_batch`` (NOT ``forward_step``) and dispatch into
     ``teleboost.training.utils.deepspeed_forward_backward`` directly —
     verl's default ``get_forward_backward_func`` path is incompatible
     with per-microbatch list-loss multi-backward.

  2. TCP (teleboost tensor-context-parallel, the ulysses-style
     spatial-temporal split for video tensors). TeleTron's initializer stores
     its TP×CP process group in Megatron-Core 0.16's native
     ``get_tensor_and_context_parallel_*`` state; the adaptor retains the old
     ``get_tensor_context_parallel_*`` names as compatibility aliases.
     Migration: keep
     ``install()`` call order in ``teleboost/programs/wan/dpo/main.py`` (explicit, before engine construction)
     so the patch lands before verl imports megatron-core. The verl
     yaml MUST set ``context_parallel_size: 1`` so verl does not
     allocate a native-CP mesh dimension that would steal GPUs from
     the TCP world.

  3. Separated VAE (``--distributed-vae --distributed-vae-world-size N``).
     teleboost/engines/teletron/parallel_state.py splits ranks into a
     producer (VAE) group and a consumer (DiT) group at mesh-init
     time. Migration: must run inside ``_build_tf_config`` /
     ``_init_device_mesh`` — NOT in forward — so the mesh is correct
     before any tensor lands on a rank..

  4. DPO eval disabled (``--eval-iters 0`` because forward_step
     returns a 5-element list that megatron's eval reducer can't
     divide). Migration: yaml ``trainer.test_freq: 0`` or equivalent.
"""

from __future__ import annotations

from typing import Any

import torch
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngine

from teleboost.programs.wan.dpo.args_adapter import build_teletron_args

_TRAINER_SRC = "teleboost.training.trainer (current standalone megatron DPO impl)"


@EngineRegistry.register(model_type="video_diffusion", backend="megatron")
class MegatronEngineWanVideo(MegatronEngine):
    """Wan video-diffusion verl megatron engine.

    Coexists with ``MegatronEngineWithLMHead`` (``language_model``) and
    ``MegatronEngineWithValueHead`` (``value_model``) — three peer
    EngineRegistry entries, none touch each other.
    """

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        # TeleTron args MUST be set before _init_device_mesh runs because
        # the teleboost.engines.teletron.megatron_adaptor wrap reads get_args()
        # (distributed_vae / consumer_models_num / ...). Without this,
        # mesh init crashes with "args is not initialized".
        build_teletron_args(engine_config, extra=None)

        # Verl base's __init__ does three things our VAE-rank path
        # must skip: ``set_random_seed`` (calls
        # ``model_parallel_cuda_manual_seed`` which needs the TP
        # group — VAE ranks legitimately have no TP group), the
        # offload/router-replay flags (DiT-only), and the layer-name
        # mapping (DiT-only). Replicate the attribute assignments
        # ourselves so role detection can run between mesh-init and
        # the seed call. On DiT ranks we do the full base init tail;
        # on VAE ranks we skip everything past the role split.
        from verl.workers.engine.base import BaseEngine

        BaseEngine.__init__(self)
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"

        self._init_device_mesh()

        # VAE producer vs DiT consumer role detection.
        # After _init_device_mesh → teleboost wrap, the wrap's
        # distributed_vae split leaves ``get_transformer_model_group()``
        # returning None on VAE producer ranks. Same rank-role check as
        # the standalone trainer.
        from teleboost.engines.teletron.parallel_state import get_transformer_model_group

        self._is_vae_producer = get_transformer_model_group() is None

        if self._is_vae_producer:
            # VAE rank: minimal sentinels so the verl Ray actor
            # construction completes. The producer-loop background
            # thread is started in
            # ``_TeleboostTrainingWorker._start_vae_producer_thread`` after
            # this __init__ returns. Every other method on this engine
            # checks ``self._is_vae_producer`` and short-circuits.
            self.mode = None
            return

        # DiT rank: complete verl MegatronEngine.__init__'s tail —
        # set_random_seed + offload flags + layer mapping + router replay.
        from verl.workers.engine.megatron.utils import set_random_seed

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self.mode = None
        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        if self.enable_routing_replay:
            from verl.utils.routing_replay import apply_router_replay_patch

            apply_router_replay_patch()
            self.mini_layer_topk_idx_list = []

    # ``TrainingWorker.__init__`` (verl engine_workers.py:126) calls
    # ``get_data_parallel_rank()`` + ``is_mp_src_rank_with_outputs()``
    # on every worker, including VAE producer ranks that legitimately
    # have no TP/PP/DP groups. Override so VAE ranks return inert
    # sentinels (DP rank 0, no groups, not src) instead of hitting
    # ``mpu.*`` asserts.
    def get_data_parallel_rank(self):
        return 0 if self._is_vae_producer else mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        return 1 if self._is_vae_producer else mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return None if self._is_vae_producer else mpu.get_data_parallel_group()

    def get_model_parallel_group(self):
        return None if self._is_vae_producer else mpu.get_model_parallel_group()

    def get_context_parallel_group(self):
        return None if self._is_vae_producer else mpu.get_context_parallel_group()

    def is_mp_src_rank_with_outputs(self):
        if self._is_vae_producer:
            return False
        return mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1 and mpu.get_context_parallel_rank() == 0

    # ------------------------------------------------------------------
    # .2 — Override verl's _init_device_mesh to match the
    # teleboost.engines.teletron.megatron_adaptor wrap signature.
    #
    # verl's default call passes ``expert_tensor_parallel_size`` as a
    # kwarg; teleboost.engines.teletron.megatron_adaptor's wrap predates ETP and does
    # not accept it — calling verl's default would raise TypeError.
    #
    # We drop ETP (Wan has no expert dimension) and pass the 8 kwargs
    # the wrap actually accepts. distributed_vae splits get_args() at
    # wrap entry, so the producer (VAE) / consumer (DiT) group is set
    # up here automatically — no extra code in this method.
    # ------------------------------------------------------------------
    def _init_device_mesh(self):
        if mpu.is_initialized():
            return

        etp = getattr(self.engine_config, "expert_tensor_parallel_size", 1) or 1
        if int(etp) > 1:
            raise ValueError(
                "MegatronEngineWanVideo: expert_tensor_parallel_size > 1 is not supported. teleboost.engines.teletron.megatron_adaptor wraps megatron.core.parallel_state.initialize_model_parallel with a signature that predates ETP; Wan has no expert dimension, so set engine.expert_tensor_parallel_size=1 in yaml."
            )

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            nccl_communicator_config_path=None,
        )

        # Cheap post-condition: TeleTron's initializer populates
        # Megatron-Core 0.16's native TP-and-CP group state. It must resolve
        # to a non-None group when context_parallel_size > 1.
        if self.engine_config.context_parallel_size > 1:
            tcp_group = mpu.get_tensor_and_context_parallel_group()
            if tcp_group is None:
                raise RuntimeError("TP-and-CP group is None after initialize_model_parallel. The DPO startup order must be apply_runtime_patches -> megatron_adaptor.install -> engine import -> mesh init.")

        # .1a — megatron-core 0.16 partial-DP group fallback.
        # ``teleboost.engines.teletron.parallel_state.initialize_model_parallel_base``
        # is frozen at an older megatron-core revision and does not
        # create the ``_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP``
        # module-level global. Megatron-core 0.16 falls back to aliasing
        # it to ``_DATA_PARALLEL_GROUP_WITH_CP`` when
        # ``intra_partial_data_parallel_size == data_parallel_size`` —
        # i.e. when no DP subdivision is in effect (single-stage DP,
        # which is our case: TP=1, PP=1, CP=1, DP=4). See
        # megatron/core/parallel_state.py line 893
        # (``_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP_WITH_CP``).
        #
        # Without this alias verl's default _build_optimizer path
        # (megatron.core.optimizer.get_megatron_optimizer) asserts
        # ``_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP is not None`` at
        # parallel_state.py:1271 and crashes mid-init. The alias is
        # semantically equivalent to megatron-core's own fallback for
        # the non-subdivided DP case — not a walkaround.
        #
        # distributed_vae=true note: when the teleboost wrap routes a
        # VAE producer rank, it intentionally leaves both DP-with-CP
        # globals as None (VAE ranks have no DiT DP role). Skip the
        # alias entirely on those ranks — get_transformer_model_group()
        # == None is the same role-detection check ``__init__`` uses
        # right after ``super().__init__`` returns.
        from teleboost.engines.teletron.parallel_state import get_transformer_model_group

        if get_transformer_model_group() is None:
            return
        if getattr(mpu, "_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP", None) is None:
            base_group = getattr(mpu, "_DATA_PARALLEL_GROUP_WITH_CP", None)
            if base_group is None:
                raise RuntimeError(
                    "Both _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP and _DATA_PARALLEL_GROUP_WITH_CP are None after initialize_model_parallel — teleboost.engines.teletron.parallel_state .initialize_model_parallel_base failed to set up the base DP-with-CP group; check the wrap return path."
                )
            mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = base_group
            base_gloo = getattr(mpu, "_DATA_PARALLEL_GROUP_WITH_CP_GLOO", None)
            if base_gloo is not None:
                mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = base_gloo

    # ------------------------------------------------------------------
    # .3 — Wan-specific TransformerConfig, bypassing
    # mbridge AutoBridge (Wan is not an HF-registered architecture).
    #
    # Source: teleboost.training.trainer.model_provider (lines 183-198):
    #   dit_model_config = set_config()['model_config']['dit']
    #   args.num_layers       = dit_model_config.config.num_layers
    #   args.hidden_size      = dit_model_config.config.dim
    #   args.ffn_hidden_size  = dit_model_config.config.ffn_dim
    #   args.num_attention_heads = dit_model_config.config.num_heads
    #   megatron_cfg = core_transformer_config_from_args(args)
    #
    # self.bridge / self.provider / self.peft_cls are intentionally None
    # for lands a WanBridge that implements
    # the load_weights / export_hf_weights protocol verl's checkpoint
    # manager + train_batch use.
    # ------------------------------------------------------------------
    def _build_tf_config(self):
        from megatron.training.arguments import core_transformer_config_from_args

        from teleboost.engines.teletron import get_args, set_config

        args = get_args()

        dit_model_config = set_config().get("model_config", {}).get("dit", None)
        if dit_model_config is None:
            raise RuntimeError("set_config()['model_config']['dit'] is None — Wan config not loaded. Expected teletron_args.config_path to reference the Wan config module in yaml.")

        args.num_layers = dit_model_config.config.num_layers
        args.hidden_size = dit_model_config.config.dim
        args.ffn_hidden_size = dit_model_config.config.ffn_dim
        args.num_attention_heads = dit_model_config.config.num_heads

        tf_config = core_transformer_config_from_args(args)
        tf_config.bf16 = bool(args.bf16)
        tf_config.fp16 = bool(args.fp16)

        self.tf_config = tf_config
        self.provider = None
        self.bridge = None
        self.peft_cls = None
        self.param_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        self.dtype = self.param_dtype
        self.weight_converter = None

        # Stash for _build_megatron_module — model_provider reads
        # it from set_config(), but pinning here documents the dependency.
        self._dit_model_config = dit_model_config

    # ------------------------------------------------------------------
    # .4 — Wan module construction via TeleTron model_provider,
    # bypassing verl's make_megatron_module (which threads through
    # mbridge). Mirrors teleboost.training.trainer.Trainer.get_model
    # (line 200) + model_provider (line 183).
    # ------------------------------------------------------------------
    def _build_megatron_module(self):
        from megatron.core.enums import ModelType

        from teleboost.models.build import build_model  # noqa: F401  (registers model builders)
        from teleboost.engines.teletron import get_args

        # verl's MegatronEngine.initialize() reads ``self.is_value_model``
        # at line 347 to label the checkpoint role ("actor" / "critic").
        # The default _build_megatron_module sets it from the HF
        # architecture string; our override bypasses that path. Wan /
        # ParallelWanTeletronModel is always the actor (DPO has no critic).
        self.is_value_model = False

        args = get_args()
        # use_zero2 path is wrap_with_ddp=False; non-zero2 path wraps with DDP.
        not bool(getattr(args, "use_zero2", True))

        model_type = getattr(args, "model_type", ModelType.encoder_or_decoder)
        args.model_type = model_type

        # PP=1, VPP=None default — no virtual-pp branch.
        # Mirrors trainer.get_model line 219+ ``else:`` branch.
        if mpu.get_pipeline_model_parallel_world_size() > 1 and args.virtual_pipeline_model_parallel_size is not None:
            raise NotImplementedError(f"Virtual pipeline parallel path not yet supported (covers the single-VPP rank branch). Source to port: {_TRAINER_SRC} get_model lines 203-218.")

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = self._call_model_provider(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

        # weight loading via pre-converted teletron ckpt
        # + vanilla megatron load_checkpoint. The user-side workflow is:
        #
        #   teleboost-convert-wan-to-teletron \
        #       --src /path/to/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
        #       --dst /path/to/Wan2.1-T2V-1.3B-teletron
        #   # (default Wan→teletron rename matches ParallelWanTeletronModel naming)
        #
        # then yaml ``teletron_args.load: /path/to/Wan2.1-T2V-1.3B-teletron``
        # plus ``finetune: true`` + ``no_load_optim: true`` + ``no_load_rng: true``
        # so megatron skips optimizer-state / rng-state load (we pass
        # optimizer=None; those flags also short-circuit the "iteration"
        # restore — does ckpt → module weight load only;
        # optimizer state lands in alongside split-DPO
        # train_batch via DeepSpeedZeroOptimizer).
        #
        # Skipping WanBridge: the convert tool + megatron load is the
        # production-tested path; runtime mbridge abstraction adds no
        # value for DPO (DPO has no actor↔vLLM rollout sync — the only
        # extra value mbridge would have provided).
        if getattr(args, "load", None):
            from megatron.training.checkpointing import load_checkpoint as megatron_load_checkpoint

            megatron_load_checkpoint(
                [model],
                optimizer=None,
                opt_param_scheduler=None,
                strict=False,
            )

        # .1b — stub ``ddp_config`` so megatron-core 0.16+
        # optimizer construction works on the raw (non-DDP-wrapped)
        # module. Both ``get_megatron_optimizer`` and its inner
        # ``_get_param_groups`` read ``model_chunk.ddp_config`` and
        # only use it to gate two FSDP-specific branches
        # (``use_custom_fsdp`` and ``data_parallel_sharding_strategy``).
        # Raw modules just need a stub where both fields are falsy and
        # the standard ``named_parameters()`` path runs.
        #
        # The use_zero2 path explicitly does NOT DDP-wrap the
        # module (DeepSpeedZeroOptimizer does its own parameter
        # sharding); attaching the stub here is the minimum change to
        # let the megatron-core 0.16+ param-grouping helper accept it.
        if not hasattr(model, "ddp_config"):
            from types import SimpleNamespace

            model.ddp_config = SimpleNamespace(
                use_custom_fsdp=False,
                data_parallel_sharding_strategy=None,
            )

        # .4 — cast model params to the configured dtype.
        # The standalone ``Trainer.get_model`` wraps with ``Float16Module``
        # when args.fp16/bf16 + not wrap_with_ddp, which auto-casts
        # inputs/params. We're not wrapping (the wrap interferes with
        # DeepSpeedZeroOptimizer's param tracking), so cast in place
        # so every Linear weight matches the bf16 latents flowing in.
        # ``load_checkpoint`` above already loaded checkpoint params
        # in their stored dtype; this normalizes everything (incl.
        # initialized-from-scratch RMSNorm scales etc.) to params_dtype.
        if args.bf16:
            model = model.to(dtype=torch.bfloat16)
        elif args.fp16:
            model = model.to(dtype=torch.float16)
        model = model.cuda()

        return [model]

    def _call_model_provider(self, *, pre_process: bool, post_process: bool):
        # Inlined model_provider (teleboost.training.trainer.Trainer
        # .model_provider). We DO NOT rebuild megatron_cfg here — the
        # standalone version calls ``core_transformer_config_from_args(args)``
        # but ``_build_tf_config`` above already did that work and stashed
        # the result on ``self.tf_config``. Reusing it keeps the
        # invariant that the same TransformerConfig instance flows through
        # the rest of the engine.
        from teleboost.models.build import build_model
        from teleboost.engines.teletron import set_config

        dit_model_config = set_config().get("model_config", {}).get("dit", None)
        return build_model(dit_model_config.type, self.tf_config)

    # ------------------------------------------------------------------
    # .1b — dual-path optimizer build.
    #
    # use_zero2=True  → DeepSpeedZeroOptimizer via
    #                   teleboost.training.lr_scheduler.SchedulerMixin
    #                   .get_optimizer_for_zero2 (production default).
    #                   verl's super()._build_optimizer expects DDP-
    #                   wrapped model with a ``ddp_config`` attribute,
    #                   which is incompatible with the use_zero2
    #                   pattern (raw module + DeepSpeedZeroOptimizer
    #                   does its own param sharding).
    # use_zero2=False → super()._build_optimizer (verl/megatron-core
    #                   distributed-optimizer). Currently requires a
    #                   DDP-wrapped module —.1c will land
    #                   the DDP-wrap step in _build_megatron_module for
    #                   that branch. For now this raises with a clear
    #                   pointer.
    # ------------------------------------------------------------------
    def _build_optimizer(self):
        from teleboost.engines.teletron import get_args

        args = get_args()

        if bool(getattr(args, "use_zero2", True)):
            import deepspeed
            from megatron.core.optimizer import OptimizerConfig

            from teleboost.training.lr_scheduler import SchedulerMixin

            # DeepSpeedZeroOptimizer expects DeepSpeed's own ``cdb``
            # (comm backend) initialized. The standalone
            # ``trainer.setup_model_and_optimizer:143`` calls
            # ``deepspeed.init_distributed()`` immediately before
            # ``get_optimizer_for_zero2``; we mirror that here. The
            # call is idempotent on the deepspeed side (no-op if cdb
            # already wired), and reuses the torch distributed
            # backend already set up by verl's _init_device_mesh.
            deepspeed.init_distributed()

            # Translate verl's McoreOptimizerConfig into megatron's
            # OptimizerConfig — ``get_optimizer_for_zero2``
            # consumes the megatron-flavor config object. Field set
            # mirrors trainer.setup_model_and_optimizer lines 137-141.
            import dataclasses as _dc

            kwargs = {}
            for f in _dc.fields(OptimizerConfig):
                if hasattr(args, f.name):
                    kwargs[f.name] = getattr(args, f.name)
            optim_cfg = OptimizerConfig(**kwargs)
            optim_cfg.timers = None

            # SchedulerMixin has no __init__ and ``get_optimizer_for_zero2``
            # only reads ``self`` for dispatch — instantiate standalone.
            scheduler = SchedulerMixin()
            # Module list passed as-is: get_optimizer_for_zero2 expects
            # list[MegatronModule], one entry per VPP rank./c
            # ensures len(self.module) == 1.
            optimizer = scheduler.get_optimizer_for_zero2(
                optim_cfg,
                self.module,
            )
            return optimizer

        # use_zero2=False — verl/megatron distributed optimizer path.
        # Requires self.module[0] to be DDP-wrapped with a ``ddp_config``
        # attribute (megatron-core 0.16 get_megatron_optimizer reads it
        # at line 482). Our _build_megatron_module currently returns raw
        # ParallelWanTeletronModel without DDP wrap; that lands in Phase
        # 2.2.d.1c (DDP wrap via megatron.core.distributed
        # .DistributedDataParallel). Until then, this branch is unsafe.
        if not hasattr(self.module[0], "ddp_config"):
            raise NotImplementedError("Non-zero2 path (use_zero2=False) requires the module to be DDP-wrapped with a ``ddp_config`` attribute, which is not implemented in _build_megatron_module yet. For now set teletron_args.use_zero2: true in yaml to route through DeepSpeedZeroOptimizer.")
        return super()._build_optimizer()

    # ``prepare_model_inputs`` / ``prepare_model_outputs`` / ``forward_step``
    # are NOT overridden — ``train_batch`` below takes over the dispatch
    # for both use_zero2 branches (``deepspeed_forward_backward``
    # for split-DPO, ``super().train_batch`` for the megatron pipeline
    # path). The verl base versions of those three methods never run on
    # this engine.

    def initialize(self):
        # full — VAE producer ranks have no DiT engine to
        # initialize. The actor-level ``_TeleboostTrainingWorker.
        # _start_vae_producer_thread`` already kicked off
        # ``DistDataProducer.run()`` in a daemon background thread; that
        # thread owns VAE encoder + dataset + dist.send to DiT ranks.
        # Verl's ``reset()`` calls ``engine.initialize()`` on every Ray
        # actor including VAE ones — return cleanly so dispatch succeeds.
        if getattr(self, "_is_vae_producer", False):
            return
        super().initialize()

    def train_batch(self, data: TensorDict, loss_function: Any) -> Any:
        # Signature exactly mirrors ``verl.workers.engine.base.BaseEngine
        # .train_batch(data: TensorDict, loss_function: Callable) -> Any``
        # so subclass introspection / instrumentation tooling sees no drift.
        # verl's ``TrainingWorker.train_batch`` asserts
        # ``self.loss_fn is not None`` before dispatching, so
        # ``loss_function`` is always non-None at the call site.
        if getattr(self, "_is_vae_producer", False):
            # VAE producer rank: the background ``DistDataProducer``
            # thread services DiT ranks via ``dist.send`` asynchronously.
            # train_batch on the VAE actor is just a dispatch sink —
            # return an empty TensorDict so verl's per-rank gather
            # collation has something to consume.
            return TensorDict({}, batch_size=[])

        from teleboost.engines.teletron import get_args, get_num_microbatches

        args = get_args()

        if not bool(getattr(args, "use_zero2", True)):
            # Non-zero2 path → super.train_batch (megatron pipeline
            # scheduler). Requires _build_optimizer/_build_megatron_module
            # DDP-wrapping which.1c hasn't landed yet, so
            # this raises today.
            return super().train_batch(data, loss_function)

        # use_zero2 path: route through deepspeed_forward_backward.
        from teleboost.programs.wan.dpo.dpo_loss import forward_step as _dpo_forward_step
        from teleboost.engines.teletron.parallel_state import get_transformer_model_group
        from teleboost.training.utils import deepspeed_forward_backward
        from teleboost.engines.teletron import get_timers

        # Standalone ``Trainer.train_step`` (teleboost/training/trainer.py:769-770)
        # registers two named timers via ``timers.get_timer(name, barrier_group=...)``
        # before the forward_step closure does ``timers.start_timer(name)``.
        # Our override bypasses train_step but the same forward_step still
        # asserts the names are registered. Pre-register both here —
        # idempotent on repeated calls (timer dict caches by name).
        timers = get_timers()
        timers.get_timer("dit-time", barrier_group=get_transformer_model_group())
        timers.get_timer("get-data-time", barrier_group=get_transformer_model_group())

        # verl hands us a single ``data`` batch (TensorDict / dict-like)
        # already in the shape ``_load_batch_inputs`` expects:
        # {context, chosen: {latents, ...}, rejected: {latents, ...}}.
        # Wrap as a single-microbatch iterator —
        # deepspeed_forward_backward will call ``next(data_iterator)``
        # exactly num_microbatches times. For smoke
        # num_microbatches=1 (global_batch_size = micro_batch_size).
        num_micro = get_num_microbatches()
        if num_micro > 1:
            # smoke is single-microbatch; multi-microbatch
            # requires a real preference-pair dataloader
            # that yields one batch dict per call.
            raise NotImplementedError(f"train_batch with num_microbatches={num_micro} requires a real preference-pair dataloader. smoke uses num_microbatches=1 (set global_batch_size = micro_batch_size in teletron_args).")
        data_iterator = iter([data])

        losses_reduced = deepspeed_forward_backward(
            forward_step_func=_dpo_forward_step,
            data_iterator=data_iterator,
            model=self.module,
            num_microbatches=num_micro,
            forward_only=False,
            zero_optimizer=self.optimizer,
        )

        # Step the optimizer — use_zero2 path doesn't return
        # (update_successful, grad_norm, num_zeros_in_grad) like the
        # megatron-native distributed optimizer, just a single .step().
        # Mirrors teleboost.training.trainer.train_step:806.
        self.optimizer.step()

        # Format result for verl. losses_reduced is a list of dicts
        # produced by ``dpo_loss_func`` — one dict per microbatch
        # (here just one). Pull out the scalar metrics for verl.

        metrics = {}
        if losses_reduced and isinstance(losses_reduced[0], dict):
            for k, v in losses_reduced[0].items():
                if hasattr(v, "item"):
                    metrics[k] = float(v.item())
                else:
                    metrics[k] = float(v)
        return TensorDict({}, batch_size=[]).set_non_tensor("metrics", metrics)
