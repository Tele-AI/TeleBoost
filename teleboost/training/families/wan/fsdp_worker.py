# Copyright 2025-2026 TeleAI and the TeleBoost contributors
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
import argparse
import logging
import os
import warnings
from functools import partial

import torch
import torch.distributed
from omegaconf import DictConfig, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.transforms import InterpolationMode

# Ray imports this worker module in fresh processes. This boundary installs
# correctness patches before the following verl symbols are bound.
from teleboost.patches.lifecycle import PATCHES_APPLIED as _PATCHES_APPLIED
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import (
    DistProfiler as WorkerProfiler,
)
from verl.utils.debug import (
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.device import get_device_id, get_device_name
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.fsdp_workers import ActorRolloutRefWorker, get_sharding_strategy
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from teleboost.models.wan.family import PATCH_SIZE, TOKENIZER_SUBPATH, resolve_wan22_dual_paths
from teleboost.engines.fsdp.sharding.runtime import run_with_sharding_managers

if not _PATCHES_APPLIED:  # pragma: no cover - bootstrap raises first
    raise RuntimeError("GRPO worker runtime patches were not installed")

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WanActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, model_deployment=None):
        super().__init__(config, role)

        # Wan-specific: separate Ulysses SP mesh for rollout (rollout SP can differ from actor SP).
        # Upstream verl 0.4.0 ActorRolloutRefWorker.__init__ only sets the actor-side
        # `self.ulysses_sharding_manager`; Wan's diffusion rollout needs its own.
        self.rollout_ulysses_sequence_parallel_size = self.config.rollout.get("ulysses_sequence_parallel_size", 1)
        device_name = get_device_name()
        world_size = torch.distributed.get_world_size()

        self.rollout_ulysses_device_mesh = None
        rollout_dp = world_size // self.rollout_ulysses_sequence_parallel_size
        if self.rollout_ulysses_sequence_parallel_size > 1:
            self.rollout_ulysses_device_mesh = init_device_mesh(
                device_name,
                mesh_shape=(rollout_dp, self.rollout_ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.rollout_ulysses_sharding_manager = FSDPUlyssesShardingManager(self.rollout_ulysses_device_mesh)

    def apply_fsdp_checkpointing(self, model, target_types, p=1.0):
        """Activation checkpointing helper for Wan attention blocks (non-reentrant)."""
        import math

        targets = [m for m in model.modules() if isinstance(m, target_types)]
        k = math.ceil(len(targets) * float(p))
        to_wrap = set(targets[:k])

        non_re_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        def check_fn(m):
            return m in to_wrap

        def wrapper_fn(m):
            return non_re_wrapper(m)

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper_fn, check_fn=check_fn)

    def use_compile(self, model):
        """Wrap each Wan WanAttentionBlock.forward with torch.compile."""
        from wan.modules.model import WanAttentionBlock

        def compile_blocks(target):
            for block in target.blocks:
                if isinstance(block, WanAttentionBlock):
                    block.forward = torch.compile(block.forward, mode="max-autotune-no-cudagraphs")

        if hasattr(model, "low_noise_model") and hasattr(model, "high_noise_model"):
            compile_blocks(model.low_noise_model)
            compile_blocks(model.high_noise_model)
        else:
            compile_blocks(model)
        return model

    def _enable_compile(self, model, compile_export_mode):
        if compile_export_mode == "compile":
            model = self.use_compile(model)
        elif compile_export_mode == "export_aoti":
            pass
        elif compile_export_mode == "disabled":
            pass
        else:
            raise RuntimeError("expected compile_export_mode arg to be one of {compile, export_aoti, disabled}")
        return model

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        """Wan-aware model + optimizer builder.

        Diverges from upstream `ActorRolloutRefWorker._build_model_optimizer` in three places:
          - tokenizer is loaded from the ``tokenizer_subpath`` subdir of
            ``model_path`` (default ``google/umt5-xxl`` — where Wan ships it);
          - `actor_model_config` is built via `GPT2Config.from_pretrained(...)` to
            bypass HF's `model_type` check (Wan does not register a model_type);
          - the actor module class is `wan.modules.model.WanModel`, with a special
            wan22 dual-model branch that wraps a low/high pair in `Wan22DualModel`.
        """
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import GPT2Config
        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # Wan ships its T5 tokenizer under <model_path>/google/umt5-xxl/ by
        # default; ``actor_rollout_ref.tokenizer_subpath`` overrides — the SAME
        # key the driver-side tokenizer load reads (teleboost.programs.wan), so
        # the two can no longer drift apart.
        tokenizer_subpath = self.config.get("tokenizer_subpath", TOKENIZER_SUBPATH)
        tokenizer_path = os.path.join(local_path, tokenizer_subpath)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # Wan transformer is bf16 across the board.
        torch_dtype = torch.bfloat16

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # Bypass AutoConfig — Wan has no `model_type` key.
        actor_model_config = GPT2Config.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")

        init_context = get_init_weight_context_manager(mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Keep the vendored Wan tree upstream-facing. TeleBoost installs
            # its FA3/FA2/SDPA policy before importing model.py, so that
            # module's ``from .attention import flash_attention`` binds the
            # TeleBoost adapter without modifying the vendored source.
            from teleboost.models.wan.attention.runtime import install_wan_attention_adapter

            if not getattr(self, "_wan_attention_adapter_handle", None) or not self._wan_attention_adapter_handle.active:
                self._wan_attention_adapter_handle = install_wan_attention_adapter(namespace="wan")

            from wan.modules.model import WanModel

            actor_module_class = WanModel

            # torch.compile of the attention blocks is an A/B-validated 18%
            # win on the reference stack, but dynamo breaks on some torch
            # builds (torch 2.9.1: "unsafe constant ConstantVariable" in the
            # rollout graph) — keep it selectable instead of hardcoded.
            compile_export_mode = str(self.config.model.get("compile_export_mode", "compile"))
            wan_version = self.config.model.get("wan_version", "wan21")
            use_wan22 = wan_version == "wan22"
            if actor_module_class.__name__ == "WanModel" and use_wan22:
                wan22_high_path, wan22_low_path = resolve_wan22_dual_paths(self.config.model, local_path)

            if actor_module_class.__name__ == "WanModel" and use_wan22:
                from teleboost.models.wan.dual import Wan22DualModel

                def build_wan_model(path):
                    model = actor_module_class.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
                    if use_liger:
                        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                        _apply_liger_kernel_to_instance(model=model)
                    # Upstream verl's `apply_monkey_patch` reads `model.config.num_attention_heads`,
                    # which fails for Wan (FrozenDict, no such attr). Pre-X3's in-tree fork instead
                    # checked `model.config.model_type == "t2v"` and applied Wan-specific Ulysses
                    # patches only when sp_size > 1. We mirror that here: skip the upstream call
                    # entirely, and install our own Ulysses patches when SP > 1.
                    if self.ulysses_sequence_parallel_size > 1:
                        from teleboost.patches.ulysses import apply_wan_ulysses_patches

                        apply_wan_ulysses_patches(model)
                    model = self._enable_compile(model, compile_export_mode)
                    model.to(torch_dtype)
                    if enable_gradient_checkpointing:
                        from wan.modules.model import WanAttentionBlock

                        self.apply_fsdp_checkpointing(model, WanAttentionBlock, 1.0)
                    return model

                low_local_path = copy_to_local(wan22_low_path, use_shm=self.config.model.get("use_shm", False))
                low_model = build_wan_model(low_local_path)
                high_local_path = copy_to_local(wan22_high_path, use_shm=self.config.model.get("use_shm", False))
                high_model = build_wan_model(high_local_path)
                boundary = self.config.get("wan22_boundary", 0.9)
                actor_module = Wan22DualModel(low_model, high_model, boundary=boundary)

                if self._is_lora:
                    logger.warning("LoRA is not supported for Wan2.2 dual-model setup; skipping.")
            else:
                actor_module = actor_module_class.from_pretrained(local_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
                # verl's checkpoint manager probes unwrap_model.can_generate()
                # (HF PreTrainedModel API) at every save. The stock upstream
                # WanModel (diffusers ModelMixin) lacks it — only patched wan
                # checkouts carry one — so pin the answer here: never generative.
                if not hasattr(actor_module, "can_generate"):
                    actor_module.can_generate = lambda: False
                # Checked mirror: teleboost/models/wan/family.py hardcodes the
                # patchification the seq-len math assumes; fail loudly if the
                # loaded checkpoint disagrees (upstream drift, wrong model dir).
                _model_patch = tuple(getattr(actor_module, "patch_size", ()) or ())
                if _model_patch and _model_patch != PATCH_SIZE:
                    raise ValueError(f"Loaded WanModel patch_size {_model_patch} != wan.family.PATCH_SIZE {PATCH_SIZE}; update teleboost/models/wan/family.py in lockstep.")

                if use_liger:
                    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                    _apply_liger_kernel_to_instance(model=actor_module)

                # See note in build_wan_model branch above. Skip apply_monkey_patch for Wan
                # and install our own Ulysses patches when SP > 1.
                if self.ulysses_sequence_parallel_size > 1:
                    from teleboost.patches.ulysses import apply_wan_ulysses_patches

                    apply_wan_ulysses_patches(actor_module)

                actor_module = self._enable_compile(actor_module, compile_export_mode)
                actor_module.to(torch_dtype)
                if enable_gradient_checkpointing:
                    from wan.modules.model import WanAttentionBlock

                    self.apply_fsdp_checkpointing(actor_module, WanAttentionBlock, 1.0)

                if self._is_lora:
                    logger.info("Applying LoRA to actor module")
                    actor_module.enable_input_require_grads()
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get("lora_rank", 0) > 0)

        if self._is_rollout and self.config.rollout.name == "hf":
            auto_wrap_policy = None

        if self.rank == 0:
            logger.info(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
            from verl.utils.ulysses import register_cp_grad_reduce_hook

            register_cp_grad_reduce_hook(actor_module_fsdp)
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch >= 2.4 required for FSDP2"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            # fused=True runs the whole AdamW step as one fused multi-tensor CUDA
            # kernel (same math as the default foreach path, fewer launches +
            # memory round-trips). Numerically equivalent (only kernel-internal
            # reduction order differs, at the ULP level). Requires params+grads+
            # state all CUDA-resident at step time — true here: update_actor loads
            # the offloaded optimizer state to GPU before optimizer.step().
            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
                fused=True,
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                logger.info(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """Wan-aware rollout builder.

        Adds a `config.type == "diffusion"` branch (taken by DanceGRPO) on top of
        upstream's vllm/sglang/hf options. The diffusion path uses TeleBoost's
        `DiffusionRollout` and `DiffusionBaseShardingManager` (the in-tree-verl
        re-exports were dropped by X3).
        """
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        device_name = get_device_name()
        init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name

        if self.config.type == "diffusion":
            from teleboost.training.families.wan.rollout import DiffusionRollout
            from teleboost.engines.fsdp.sharding.diffusion import DiffusionBaseShardingManager

            # Inject a fully-bound VIPO pixel-weight callback so the library
            # rollout owns ZERO recipes.* knowledge — not the import, not the
            # hyperparameters, not even the enable flag. Imported only when
            # enabled, keeping DINOv2/transformers out of the default path.
            pixel_weight_fn = None
            pixel_cfg = self.config.get("pixel_weight", {}) or {}
            if pixel_cfg.get("enable", False):
                from functools import partial

                from teleboost.algorithms.vipo import compute_batch_pixel_weight_maps

                pixel_weight_fn = partial(
                    compute_batch_pixel_weight_maps,
                    model_path=pixel_cfg.get("model_path", "facebook/dinov2-large"),
                    pca_method=pixel_cfg.get("pca_method", "weighted"),
                    sigma=float(pixel_cfg.get("sigma", 1.0)),
                )
            # The rollout takes neutral parameters only; algorithm flags are
            # read HERE (the recipes backbone owns algorithm knowledge).
            # RatioNorm compares actor-recomputed transition means against
            # rollout-frozen means, so the rollout must emit them.
            emit_prev = bool(self.config.actor.get("grpo_guard", {}).get("ratio_norm", False))
            rollout = DiffusionRollout(
                module=self.actor_module_fsdp,
                config=self.config,
                pixel_weight_fn=pixel_weight_fn,
                emit_prev_sample_mean=emit_prev,
            )
            rollout_sharding_manager = DiffusionBaseShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=None,
                model_config=self.actor_model_config,
                offload_param=self._is_offload_param,
            )
            return rollout, rollout_sharding_manager

        # This worker builds Wan diffusion models only; the upstream token-LM
        # rollouts (hf/vllm/sglang) cannot run them. (Replaces ~90 lines copied
        # verbatim from upstream v0.4.0 that could never execute here.)
        raise NotImplementedError(f"rollout.name={rollout_name!r}: WanActorRolloutRefWorker only supports the 'diffusion' rollout; use upstream verl workers for token-LM rollouts.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from teleboost.training.families.wan.actor import DiffusionDataParallelPPOActor as DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels

            # The diffusion actor reads ``pixel_weight`` (VIPO) and
            # ``flow_grpo`` from ``self.config`` inside dp_actor, but
            # those Hydra blocks live at ``actor_rollout_ref.<flag>``,
            # not ``actor_rollout_ref.actor.<flag>``.  Without this
            # merge, VIPO mode produces dense ``(T,H,W)`` log-probs in
            # the rollout but scalar log-probs in the actor (the
            # actor's ``_pixel_enabled()`` returns False), which fails
            # at ``ratio = exp(new - old)`` with a shape mismatch like
            # ``(16) vs (3072)``.  Same path for flow-grpo's
            # ``shuffle_timesteps`` and ``timestep_indices``.
            #
            # ``self.config.actor`` is a struct-typed OmegaConf node;
            # opening struct mode briefly is required to add a new key.
            _was_struct = OmegaConf.is_struct(self.config.actor)
            OmegaConf.set_struct(self.config.actor, False)
            try:
                # ``guide_scale`` must reach the actor too: the recompute's CFG
                # combination has to match the rollout's, or old/new log-probs
                # are computed under two different effective policies and the
                # ratio carries a systematic bias (was hardcoded 5.0 before).
                for _propagate_key in ("pixel_weight", "flow_grpo", "guide_scale", "wan22_boundary", "latent_channels"):
                    if _propagate_key in self.config and _propagate_key not in self.config.actor:
                        self.config.actor[_propagate_key] = self.config[_propagate_key]
            finally:
                OmegaConf.set_struct(self.config.actor, _was_struct)

            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_rollout and hasattr(self.rollout, "vae_module") and str(self.config.model.get("compile_export_mode", "compile")) == "compile":
            self.rollout.vae_module.model.decoder = torch.compile(
                self.rollout.vae_module.model.decoder,
                mode="default",
            )

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                # v0.7.1 split `checkpoint.contents` into `save_contents` + `load_contents`;
                # FSDPCheckpointManager now wants the whole `checkpoint` DictConfig so it can
                # read both halves itself.
                checkpoint_contents=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(get_device_id())
        timing_generate = {}

        def _run_generate(sharded_prompts):
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=sharded_prompts)
            self.rollout_sharding_manager.postprocess_data(sharded_prompts)
            log_gpu_memory_usage("After rollout generation", logger=logger)
            return output

        return run_with_sharding_managers(
            prompts,
            context_managers=(self.rollout_ulysses_sharding_manager, self.rollout_sharding_manager),
            preprocess_managers=(self.rollout_sharding_manager, self.rollout_ulysses_sharding_manager),
            run=_run_generate,
        )

    def _set_expandable_segments(self, enabled: bool) -> None:
        """Toggle expandable_segments per update phase (gated by TELEBOOST_EXPANDABLE_SEGMENTS=1).

        A global ``PYTORCH_CUDA_ALLOC_CONF`` export would reach the colocated
        vLLM judge and break its CuMemAllocator, so toggle at runtime instead.
        Only affects NEW segments, so flipping per phase is safe.
        """
        import os

        if os.environ.get("TELEBOOST_EXPANDABLE_SEGMENTS", "0") != "1":
            return
        if not torch.cuda.is_available():
            return
        try:
            value = f"expandable_segments:{enabled}"
            if getattr(torch._C, "_accelerator_setAllocatorSettings", None) is not None:
                torch._C._accelerator_setAllocatorSettings(value)
            else:
                torch.cuda.memory._set_allocator_settings(value)
            torch.cuda.empty_cache()  # drop blocks cached under the previous mode
        except Exception as exc:  # private API — degrade loudly, not fatally
            print(f"[alloc] expandable_segments unavailable: {exc}", flush=True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """Wan-aware actor update.

        Diverges from upstream `ActorRolloutRefWorker.update_actor` in two ways:
        - data is left on CPU (the actor moves it onto GPU per micro-batch inside
          update_policy; diffusion DataProto is too large to fit a whole batch on
          one device);
        - skip the FLOPs / mfu metrics block: it depends on `meta_info["global_token_num"]`,
          which is set by LM rollouts but not by `DiffusionRollout`. We just step the
          scheduler and return the inner update_policy metrics.
        """
        data = data.to("cpu")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        # expandable_segments on for the update, off before rollout/reward.
        self._set_expandable_segments(True)
        try:

            def _run_update(sharded_data):
                metrics = self.actor.update_policy(data=sharded_data)
                self.actor_lr_scheduler.step()
                output = DataProto(meta_info={"metrics": metrics})
                return output.to("cpu")

            output = run_with_sharding_managers(
                data,
                context_managers=(self.ulysses_sharding_manager,),
                preprocess_managers=(self.ulysses_sharding_manager,),
                preprocess_keyword_first=True,
                run=_run_update,
            )
        finally:
            self._set_expandable_segments(False)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        return output


def dict_to_namespace(d):
    return argparse.Namespace(**d)
