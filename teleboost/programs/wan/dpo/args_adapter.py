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
"""Translate verl OmegaConf engine_config + yaml extras into the
``argparse.Namespace`` consumed by ``teleboost.engines.teletron.get_args()``.

Why this exists
---------------
The TeleTron/Megatron runtime is built around megatron-LM's
``parse_args`` contract; the resulting ``Namespace`` is stashed in
``teleboost.engines.teletron._GLOBAL_ARGS`` and read everywhere — including the
``teleboost.engines.teletron.megatron_adaptor`` wrapper around
``initialize_model_parallel`` (which reads
``margs.distributed_vae`` / ``margs.distributed_vae_world_size`` /
``margs.consumer_models_num`` / ``margs.dit_world_size``).

Verl drives the engine via OmegaConf and never calls ``parse_args``,
so without an adapter ``get_args()`` raises ``args is not initialized``
the moment we reach ``_init_device_mesh``. We rebuild a minimal
megatron-flavor Namespace from the verl config and stash it via
``set_global_args`` (sidesteps the timers/microbatch/tensorboard side
effects of ``set_args`` which assume a megatron launcher).

Mapping rule: any TeleTron/Megatron field read during DPO init or train
must be present here. The yaml escape hatch ``teletron_args`` is a
free-form dict merged last; use it for Megatron/Wan flags that do not
have a clean verl engine-config counterpart (e.g. ``use_zero2``,
``with_ema``, ``vision_pretraining``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

from teleboost.engines.teletron import set_global_args

_REQUIRED_TELETRON_FIELDS = (
    # mesh init (teleboost.engines.teletron.parallel_state.initialize_model_parallel_decorators)
    "distributed_vae",
    "distributed_vae_world_size",
    "consumer_models_num",
    # trainer.initialize_megatron derives this; we precompute it
    "dit_world_size",
    # use_zero2 selects deepspeed_forward_backward vs get_forward_backward_func
    "use_zero2",
)

# Env-var channel for yaml->ray-worker plumbing.
#
# Verl's McoreEngineConfig is a frozen dataclass with no escape-hatch
# field, and verl's TrainingWorkerConfig doesn't carry arbitrary user
# blocks either. Rather than fork verl's config types, teleboost.programs.wan.dpo.main
# JSON-dumps the yaml ``teletron_args`` block into this env var
# BEFORE ray.init runs — Ray inherits the launching process's
# environment by default, so the worker actors pick it up
# transparently. The adapter reads it lazily here.
_ENV_TELETRON_ARGS = "TELEBOOST_TELETRON_ARGS"


def _extras_from_env() -> dict:
    raw = os.environ.get(_ENV_TELETRON_ARGS, "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{_ENV_TELETRON_ARGS} is set but is not valid JSON: {e}. teleboost.programs.wan.dpo.main should be writing OmegaConf.to_container(..., resolve=True) output here — check DPO main plumbing drift.")


def _apply_validate_args_derivations(ns: argparse.Namespace) -> None:
    """Apply the safe subset of megatron's ``validate_args()`` derivations.

    Megatron's ``validate_args`` does TWO things: derives fields that
    other code reads (``params_dtype``, ``data_parallel_size``,
    iteration counters, dtype-coherence fields) AND enforces mesh
    consistency via global-state assertions (``mpu.is_initialized()``,
    world-size×TP×PP×CP arithmetic). The latter conflicts with verl's
    own mesh-init flow (we drive ``mpu.initialize_model_parallel``
    ourselves in ``_init_device_mesh``).

    This helper does only the derivation half — set fields consumed by
    TeleTron/Megatron code, leave mesh validation to verl. Each field
    here mirrors a specific line in megatron's validate_args (cross-
    referenced in comments).

    Mutates ``ns`` in place. Idempotent on re-call.
    """
    # ``params_dtype`` dispatch from --bf16 / --fp16 flags.
    # megatron arguments.py validate_args branch on dtype flags.
    if ns.bf16:
        ns.params_dtype = torch.bfloat16
    elif ns.fp16:
        ns.params_dtype = torch.float16
    else:
        ns.params_dtype = torch.float32

    # Optimizer-config dtype fields — megatron-core's
    # ``OptimizerConfig.__post_init__`` asserts each is float32
    # unless use_precision_aware_optimizer=True
    # (optimizer_config.py lines 202-208). teleboost.training.arguments
    # may set them to non-fp32 strings; force the safe defaults.
    ns.main_grads_dtype = torch.float32
    ns.main_params_dtype = torch.float32
    ns.exp_avg_dtype = torch.float32
    ns.exp_avg_sq_dtype = torch.float32
    ns.use_precision_aware_optimizer = False

    # Iteration-counter fields validate_args initializes to 0.
    # load_checkpoint asserts each == 0 at finetune-mode entry
    # (checkpointing.py lines 1358-1360).
    #
    # ``iteration`` + ``num_floating_point_operations_so_far`` are
    # set by TeleTron ``setup_model_and_optimizer`` after
    # ``load_checkpoint`` (trainer.py lines 153-157). In the
    # verl-recipes path we bypass that Trainer wrapper (engine.initialize
    # calls megatron's load_checkpoint directly without rebinding these
    # back onto ``args``). ``DPODataLoaderBuilder.build_train_valid_test_data_loaders``
    # (teleboost/training/dpo_dataloader.py) reads ``args.iteration``
    # directly; the producer-side ``DistDataProducer.__init__`` reaches
    # the same code path. Seed both to 0 — load_checkpoint will
    # overwrite ``iteration`` to the checkpoint's value if non-finetune.
    for counter_field in (
        "consumed_train_samples",
        "consumed_valid_samples",
        "skipped_train_samples",
        "iteration",
        "num_floating_point_operations_so_far",
    ):
        if not hasattr(ns, counter_field):
            setattr(ns, counter_field, 0)

    # ``data_parallel_size`` = world_size / (TP × PP × CP), derived
    # by validate_args. The microbatches calculator
    # (teleboost/engines/teletron/microbatches.py) reads it; without it
    # ``build_num_microbatches_calculator`` AttributeErrors mid-init.
    # Formula mirrors trainer.initialize_megatron.
    mesh_size = ns.tensor_model_parallel_size * ns.pipeline_model_parallel_size * ns.context_parallel_size
    ns.data_parallel_size = max(1, ns.world_size // mesh_size)


def _bootstrap_teleboost_globals(ns: argparse.Namespace) -> None:
    """Initialize TeleTron runtime side effects: microbatches + timers.

    We don't call ``set_args()`` directly because it asserts
    ``_GLOBAL_ARGS is None`` (one-shot init) which fights Ray actor
    restarts; we replicate its body with idempotent
    ``hasattr``/``is None`` guards.

    Tensorboard writer is intentionally skipped (rank-conditional,
    verl has its own logger pipeline).
    """
    from teleboost.engines.teletron import runtime_state as _gv
    from teleboost.engines.teletron.config import set_config
    from teleboost.engines.teletron.microbatches import build_num_microbatches_calculator
    from teleboost.engines.teletron.timers import Timers

    if _gv._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        _gv._GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(ns)

    if _gv._GLOBAL_TIMERS is None:
        # Timers init pulls dit config off set_config() for per-layer
        # timing labels. set_config reads args.config_path which is
        # set in the calling adapter; this resolves correctly here.
        dit_model_config = set_config().get("model_config", {}).get("dit", None)
        if dit_model_config is not None:
            _gv._GLOBAL_TIMERS = Timers(ns, dit_model_config.config)


def _world_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def _seed_namespace_with_megatron_defaults(extra: dict) -> argparse.Namespace:
    """Call megatron's own ``parse_args`` so every one of its ~450 args
    has a default. ``core_transformer_config_from_args`` reads dozens of
    fields we wouldn't enumerate manually (multi_latent_attention,
    rotary_interleaved, expert_model_parallel_size, etc.); without
    seeding from parse_args every new megatron field would surface as
    an AttributeError mid-build.

    The CLI fields we extract from ``extra`` here are the structural
    ones megatron's argparse marks as required (--num-layers,
    --hidden-size, --num-attention-heads, --micro-batch-size,
    --global-batch-size, --seq-length, --max-position-embeddings).
    Anything else lands on the resulting Namespace via the regular
    overlay below.
    """

    # Pull the required structural fields out of `extra` (don't pop —
    # we still need them merged below so the engine sees consistent
    # values).
    def _g(key, default):
        v = extra.get(key, default)
        return str(v) if v is not None else str(default)

    # `extra` may not carry num_layers etc. yet (mock_dit_config /
    # set_config provides them later); use small placeholder defaults
    # — _build_tf_config writes the real values back from
    # set_config()['model_config']['dit'] before calling
    # core_transformer_config_from_args.
    fake_argv = [
        "build_teletron_args",
        "--num-layers",
        _g("num_layers", 2),
        "--hidden-size",
        _g("hidden_size", 128),
        "--num-attention-heads",
        _g("num_attention_heads", 2),
        "--micro-batch-size",
        _g("micro_batch_size", 1),
        "--global-batch-size",
        _g("global_batch_size", 8),
        "--seq-length",
        _g("seq_length", 64),
        "--max-position-embeddings",
        _g("max_position_embeddings", 64),
    ]
    if extra.get("bf16", True):
        fake_argv.append("--bf16")
    if extra.get("fp16", False):
        fake_argv.append("--fp16")

    saved_argv = sys.argv
    sys.argv = fake_argv
    try:
        # Need BOTH parsers — neither alone is a superset:
        # - teleboost.training.arguments.parse_args has teleboost-specific
        #   extensions (activation_offload, use_zero2, distributed_vae,
        #   consumer_models_num, ...) that megatron-LM doesn't know
        #   about — the Wan model __init__ reads them directly.
        # - megatron-LM's parse_args has newer megatron-core fields
        #   (multi_latent_attention, etc.) that teleboost's frozen copy
        #   of the _add_*_args helpers doesn't have —
        #   core_transformer_config_from_args reads them.
        # Merge: teleboost first (it's the wider superset for our use
        # cases, ~274 fields), then megatron parse_args fills any field
        # teleboost lacks (~180 more from megatron's ~450 total).
        from teleboost.training.arguments import parse_args as teleboost_parse_args

        ns = teleboost_parse_args()
        try:
            from megatron.training.arguments import parse_args as megatron_parse_args

            megatron_ns = megatron_parse_args(ignore_unknown_args=True)
            for k, v in vars(megatron_ns).items():
                if not hasattr(ns, k):
                    setattr(ns, k, v)
        except ImportError:
            pass  # PYTHONPATH may not have Megatron-LM; teleboost-only is OK
        return ns
    finally:
        sys.argv = saved_argv


def build_teletron_args(engine_config: Any, extra: dict | None = None) -> argparse.Namespace:
    """Construct a megatron-flavor Namespace from verl's engine_config.

    Parameters
    ----------
    engine_config : verl McoreEngineConfig (or OmegaConf node)
        Carries TP/PP/CP/EP sizes + dtype.
    extra : dict-like, optional
        Free-form yaml block merged last. Place TeleTron/Wan flags here
        (``use_zero2``, ``with_ema``, ``vision_pretraining``, ...).

    Returns
    -------
    argparse.Namespace
        Already ``set_global_args``-d; also returned for callers that
        want to keep a reference.
    """
    # Resolution order: explicit `extra` arg > env-var > {}.
    # Engines pass `extra=None` (no in-process override available); env
    # is the canonical channel from teleboost.programs.wan.dpo.main.
    if extra is None:
        extra = _extras_from_env()
    extra = dict(extra)

    # Seed the namespace with megatron's full default-populated Namespace
    # — every ``core_transformer_config_from_args``-readable field
    # exists with a sane default; we then overlay verl + yaml settings.
    ns = _seed_namespace_with_megatron_defaults(extra)

    rank, world_size = _world_info()

    distributed_vae = bool(extra.pop("distributed_vae", False))
    distributed_vae_world_size = int(extra.pop("distributed_vae_world_size", 0))
    consumer_models_num = int(extra.pop("consumer_models_num", 1))

    if distributed_vae:
        if consumer_models_num <= 0:
            raise ValueError(f"distributed_vae=True but consumer_models_num <= 0; got consumer_models_num={consumer_models_num}, WORLD_SIZE={world_size}, distributed_vae_world_size={distributed_vae_world_size}.")
        if distributed_vae_world_size <= 0:
            raise ValueError(f"distributed_vae=True but distributed_vae_world_size <= 0; got distributed_vae_world_size={distributed_vae_world_size}, WORLD_SIZE={world_size}, consumer_models_num={consumer_models_num}. Set teletron_args.distributed_vae_world_size in yaml.")
        dit_ranks = world_size - distributed_vae_world_size
        if dit_ranks <= 0:
            raise ValueError(f"distributed_vae=True leaves no DiT ranks; WORLD_SIZE={world_size}, distributed_vae_world_size={distributed_vae_world_size}, consumer_models_num={consumer_models_num}.")
        if dit_ranks % consumer_models_num != 0:
            raise ValueError(f"distributed_vae rank split is not divisible by consumer_models_num; WORLD_SIZE={world_size}, distributed_vae_world_size={distributed_vae_world_size}, dit_world_size_candidate={dit_ranks}, consumer_models_num={consumer_models_num}.")
        # Mirrors trainer.initialize_megatron line 361-362 exactly:
        #   args.world_size = (world_size - distributed_vae_world_size) // consumer_models_num
        #   args.dit_world_size = args.world_size * consumer_models_num
        adjusted_world_size = dit_ranks // consumer_models_num
        dit_world_size = adjusted_world_size * consumer_models_num
    else:
        adjusted_world_size = world_size
        dit_world_size = world_size

    # Overlay onto the megatron-defaulted Namespace (ns already has
    # ~457 fields from parse_args). We OVERWRITE the megatron-default
    # value with the verl-driven / derived value when there's a key
    # collision — that's the intended semantics: yaml > megatron default.
    overlay = {
        # mesh / parallelism (read by teleboost.engines.teletron.megatron_adaptor wrap +
        # teleboost.engines.teletron.parallel_state) — pull from verl engine_config
        "tensor_model_parallel_size": int(getattr(engine_config, "tensor_model_parallel_size", 1)),
        "pipeline_model_parallel_size": int(getattr(engine_config, "pipeline_model_parallel_size", 1)),
        "virtual_pipeline_model_parallel_size": getattr(engine_config, "virtual_pipeline_model_parallel_size", None),
        "context_parallel_size": int(getattr(engine_config, "context_parallel_size", 1)),
        "expert_model_parallel_size": int(getattr(engine_config, "expert_model_parallel_size", 1)),
        # TeleTron distributed-VAE fields (from yaml extra)
        "distributed_vae": distributed_vae,
        "distributed_vae_world_size": distributed_vae_world_size,
        "consumer_models_num": consumer_models_num,
        # derived
        "rank": rank,
        "world_size": adjusted_world_size,
        "dit_world_size": dit_world_size,
        # use_zero2 routes train_batch into deepspeed_forward_backward
        # (split-DPO) vs the megatron default path
        "use_zero2": bool(extra.pop("use_zero2", True)),
        # dtype (read by trainer.setup_model_and_optimizer + a handful of utils)
        "bf16": bool(extra.pop("bf16", True)),
        "fp16": bool(extra.pop("fp16", False)),
    }
    for k, v in overlay.items():
        setattr(ns, k, v)

    # Fill validate_args-derived fields (dtypes, counters,
    # data_parallel_size). See helper for the full list + sources.
    _apply_validate_args_derivations(ns)

    # Free-form merge — anything in yaml `teletron_args` lands on top
    # of the megatron defaults. YAML is the source of truth for
    # TeleTron/Wan-specific overrides.
    for k, v in extra.items():
        setattr(ns, k, v)

    # Teleboost-side global args (read by DPO modules via
    # ``teleboost.engines.teletron.get_args``).
    set_global_args(ns)

    # Bootstrap TeleTron global side state: microbatch calculator +
    # timers. See helper docstring for why we don't call ``set_args``
    # directly.
    _bootstrap_teleboost_globals(ns)

    # Megatron-side global args (read by megatron.training.* via
    # ``megatron.training.get_args``). The two namespaces are independent
    # — megatron.training.checkpointing.load_checkpoint calls
    # ``megatron.training.get_args()`` directly, so without this dual-set
    # the load_checkpoint path asserts ``args is not initialized``.
    # ``megatron.training.global_vars.set_args`` is a single-line global
    # assignment (no timers / microbatch-calculator / tensorboard side
    # effects, unlike megatron.training.initialize.set_args). Safe to call.
    try:
        from megatron.training.global_vars import set_args as _megatron_set_args

        _megatron_set_args(ns)
    except ImportError:
        # PYTHONPATH may not have Megatron-LM in some test contexts —
        # args_adapter still works for teleboost-only callers.
        pass

    return ns
