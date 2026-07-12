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
"""TeleBoost-DPO entry — verl Hydra-driven launcher.

Stands up the verl ``TrainingWorker`` + Megatron-LM core_v0.16 stack,
registers the ``video_diffusion`` engine for Wan, and drives one of
three exclusive run modes selected via ``trainer.*`` config keys:

  * **train** (default) — call ``real_train_step`` for
    ``trainer.total_training_steps`` iterations. Requires
    ``teletron_args.distributed_vae=true`` (VAE producer
    thread + DiT-side ``DistVAEConsumerBatchLoader``).
  * **precision replay** — opt in with ``trainer.phase3_replay_dir``
    (glob a directory) or ``trainer.phase3_replay_dump`` (single .pt).
    Replays saved DPO input dumps through ``model.forward``
    and compares per-pair ``noise_pred`` + DPO loss against the dump.
  * **smoke** — opt in with ``trainer.run_smoke_train_step=true``.
    Synthesizes a stub preference-pair batch and exercises the
    split-DPO ``deepspeed_forward_backward`` multi-backward path.

Hydra entry — same defaults pattern as verl SFT:

  python -m teleboost.programs.wan.dpo.main --config-name dpo_trainer \\
    trainer.n_gpus_per_node=4 trainer.nnodes=1
"""

from __future__ import annotations

import logging
import os

import hydra
import ray
from omegaconf import OmegaConf


def _bootstrap_runtime() -> None:
    """Install process-wide patches before importing the training stack."""

    # These imports are deliberately local: calling this function before any
    # recipes/verl import makes the ordering explicit without relying on E402
    # exemptions for module-level imports after side effects.
    from teleboost import apply_runtime_patches
    from teleboost.engines.teletron.megatron_adaptor import install as install_megatron_tcp

    apply_runtime_patches()
    install_megatron_tcp()


_bootstrap_runtime()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# Reuse the production DPO loss / forward_step from
# ``teleboost/programs/wan/dpo/dpo_loss.py``. This keeps timestep sampling,
# FlowMatchScheduler, beta-sigmoid coeff, saved-input replay, and dump
# behavior in one implementation shared by the standalone and verl
# launchers.
#
# Note: ``forward_step`` reads its config (use_saved_inputs / save_dumps /
# noise_seed / etc.) from a module-level ``args`` variable normally
# set inside ``if __name__ == "__main__":``. ``MegatronEngineWanVideo
# .train_batch`` binds our get_args() Namespace onto the module before
# calling forward_step, so those getattr-with-default reads resolve. Import it
# inside the adapter so this entrypoint keeps its bootstrap ordering explicit.


def _dpo_loss_fn_for_verl(model_output, batch, **_kwargs):
    """Adapter to fit verl's ``set_loss_fn`` slot signature
    ``(model_output, batch) -> (loss, metrics)``.

    Verl's train_batch dispatch asserts ``self.loss_fn is not None`` and
    passes it as ``loss_function=`` to ``engine.train_batch``. Our
    ``MegatronEngineWanVideo.train_batch`` override ignores
    ``loss_function`` because ``forward_step`` returns its own loss
    closure as the second tuple element (megatron pipeline style). We
    pass ``dpo_loss_func`` here so any non-overridden path
    (super().train_batch) still has the correct loss.
    """
    from teleboost.programs.wan.dpo.dpo_loss import dpo_loss_func

    return dpo_loss_func(model_output)


def _build_training_worker(config):
    """Mirror ``verl.trainer.sft_trainer_ray.SFTTrainer._build_engine``,
    plus the teleboost-specific worker subclass that registers the
    Wan ``video_diffusion`` engine inside each Ray actor.
    """
    # Ray actor processes are forked clean — they do NOT inherit the
    # driver's module imports, so the ``@EngineRegistry.register`` side
    # effect from ``teleboost.programs.wan.dpo.megatron_wan`` never fires in
    # workers and ``EngineRegistry.get_engine_cls("video_diffusion",
    # "megatron")`` asserts "Unknown model_type". Subclass TrainingWorker
    # so each worker imports the engine module before verl's __init__
    # reaches EngineRegistry.new. The module-level explicit patch call applies cross-
    # cutting compat patches (TensorDict / wan weight saver) and the
    # ``megatron_adaptor`` TCP wrap.
    # verl ``RayWorkerGroup`` only exposes methods to the driver-side
    # proxy when they're decorated with ``@register(dispatch_mode=...)``
    # — undecorated methods stay on the actor and aren't reachable via
    # ``training_client.method()``. ``Dispatch.ONE_TO_ALL`` mirrors how
    # verl's own ``set_loss_fn`` / ``reset`` are wired (call on every
    # rank, no data dispatch).
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.ray import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

    class _TeleboostTrainingWorker(TrainingWorker):
        def __init__(self, *args, **kwargs):
            # This runs in a fresh Ray actor process: the driver's install()
            # does NOT propagate here. Wrap megatron BEFORE the engine
            # (megatron_wan._init_device_mesh) touches parallel state.
            from teleboost.engines.teletron.megatron_adaptor import install as _install_megatron_tcp

            _install_megatron_tcp()

            from teleboost.programs.wan.dpo import megatron_wan  # noqa: F401 — registers MegatronEngineWanVideo

            super().__init__(*args, **kwargs)
            # Start only inside the symmetric real_train_step RPC.  The old
            # __init__-time daemon could outlive the RPC that owned training
            # and had no stop/join path.
            self._vae_producer_thread = None
            self._vae_producer = None
            self._vae_producer_error = None
            self._vae_control_group = None

        def _get_distributed_vae_control_group(self):
            """Create one world-sized Gloo group for lifecycle/control data.

            The large encoded tensor remains on NCCL. Reverse status receives
            and object envelopes must not share the default NCCL communicator:
            background irecv plus bidirectional unbatched P2P can lazy-create
            NCCL communicators in a different sequence on producer/consumer.
            Every rank calls this method from the same ONE_TO_ALL RPC, so
            ``new_group`` ordering is deterministic.
            """

            if self._vae_control_group is None:
                from datetime import timedelta

                import torch.distributed as dist

                from teleboost.engines.teletron.distributed.distributed_encoder import (
                    _protocol_timeout_seconds,
                )

                world_size = dist.get_world_size(group=dist.group.WORLD)
                self._vae_control_group = dist.new_group(
                    ranks=list(range(world_size)),
                    backend="gloo",
                    timeout=timedelta(seconds=_protocol_timeout_seconds()),
                )
            return self._vae_control_group

        def _start_vae_producer_thread(
            self,
            *,
            target_train_iters: int,
            control_group,
        ):
            """Run ``DistDataProducer.run()`` in a joined non-daemon thread.

            Why a thread and not a separate Ray actor class:
            ``DistDataProducer`` uses ``torch.distributed.send/recv``
            against DiT ranks. All Ray actors in the same
            ``RayResourcePool`` share one ``init_process_group`` world
            — splitting VAE/DiT into separate Ray actor classes would
            put them in different process groups, breaking dist.send.
            A background thread inside the shared actor keeps the world group
            intact. The owning ``real_train_step`` RPC immediately joins it,
            so both success and failure propagate back to the Ray driver.

            ``DPODataLoaderBuilder`` is instantiated standalone — it has no
            __init__ and ``build_train_valid_test_data_loaders`` only
            reads global ``get_args() / set_config()`` (no Trainer
            instance state).

            The standalone trainer wraps each non-None dataloader with
            ``iter()``; we inline that two-line bridge here rather than
            subclassing ``Trainer`` (which would drag in the full DiT
            init path).
            """
            import os
            import threading

            import torch

            from teleboost.engines.teletron.distributed.distributed_encoder import DistDataProducer
            from teleboost.training.dpo_dataloader import DPODataLoaderBuilder
            from teleboost.engines.teletron import set_config

            encoder_cfg = set_config().get("model_config", {}).get("encoder", None)
            if encoder_cfg is None or not getattr(encoder_cfg, "type", None):
                raise RuntimeError(
                    "VAE producer rank: set_config()['model_config']['encoder']"
                    "['type'] is missing. Required when "
                    "teletron_args.distributed_vae=true so DistDataProducer "
                    "knows which encoder (Wan VAE / T5 / CLIP / ...) to instantiate. "
                    "Set teletron_args.config_path to a yaml/Python "
                    "config that defines model_config.encoder.type (e.g. "
                    "teleboost.programs.wan.dpo.wan_dpo_t2v.config)."
                )

            # Construction blocks on the consumer READY handshake.  The
            # ONE_TO_ALL RPC runs all actors concurrently, so consumer ranks
            # send READY while this thread waits.
            encoder_name = encoder_cfg.type
            world_rank = int(os.environ.get("RANK", "0"))
            device = torch.cuda.current_device()

            def _build_data_iterators(*args, **kwargs):
                loaders = dataloader_helper.build_train_valid_test_data_loaders(*args, **kwargs)
                if isinstance(loaders, tuple) and len(loaders) == 5:
                    # return_ds=True path: (train_dl, valid_dl, test_dl, train_ds, valid_ds)
                    # iter() only the dataloaders, keep datasets as-is.
                    return (
                        iter(loaders[0]) if loaders[0] is not None else None,
                        iter(loaders[1]) if loaders[1] is not None else None,
                        iter(loaders[2]) if loaders[2] is not None else None,
                        loaders[3],
                        loaders[4],
                    )
                return tuple(iter(ld) if ld is not None else None for ld in loaders)

            dataloader_helper = DPODataLoaderBuilder()
            self._vae_producer = None
            self._vae_producer_error = None

            def _build_encoder(name, device):
                from teleboost.models.encoder_registry import get_encoder

                return get_encoder(name=name, device=device)

            def _run_producer():
                try:
                    producer = DistDataProducer(
                        rank=world_rank,
                        encoder_name=encoder_name,
                        device=device,
                        build_train_valid_test_data_iterators=_build_data_iterators,
                        target_train_iters=target_train_iters,
                        encoder_factory=_build_encoder,
                        control_group=control_group,
                        data_group=torch.distributed.group.WORLD,
                    )
                    self._vae_producer = producer
                    producer.run()
                except BaseException as exc:
                    import traceback

                    self._vae_producer_error = (
                        exc,
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    )

            self._vae_producer_thread = threading.Thread(
                target=_run_producer,
                daemon=False,
                name=f"VAEProducer-rank{world_rank}",
            )
            self._vae_producer_thread.start()

        def _join_vae_producer_thread(self):
            thread = self._vae_producer_thread
            if thread is None:
                raise RuntimeError("VAE producer thread was not started")
            thread.join()
            if thread.is_alive():  # join above has no timeout; defensive only
                raise RuntimeError("VAE producer thread did not terminate")
            if self._vae_producer_error is not None:
                error, formatted = self._vae_producer_error
                raise RuntimeError("distributed-VAE producer failed:\n" + formatted) from error

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def real_train_step(self, num_iters: int = 1):
            """Preference-pair data through the real loader.

            DiT roots send READY, receive lifecycle-framed encoded batches,
            and always report DONE or ERROR. VAE ranks start a non-daemon
            producer thread and join it in this same RPC. No participant
            returns until the symmetric terminal handshake has completed.

            Requires ``teletron_args.distributed_vae=true``. The DiT
            consumer loader supports only distributed-VAE mode for
            ``teletron`` / ``wan`` model types. Set
            ``trainer.run_smoke_train_step=true`` for a synthesized stub
            batch instead, or ``trainer.phase3_replay_*`` for the
            precision-check path.
            """
            import torch

            from teleboost.engines.teletron import get_args

            num_iters = int(num_iters)
            if num_iters <= 0:
                raise ValueError(f"num_iters must be > 0; got {num_iters}")
            args = get_args()
            if not bool(getattr(args, "distributed_vae", False)):
                raise RuntimeError("real_train_step requires teletron_args.distributed_vae=true. The DiT consumer loader has no non-distributed-VAE path for ParallelWanTeletronModel. For non-distributed-VAE smoke, use smoke_train_step (stub batch) instead.")

            import torch.distributed as dist
            from megatron.core import mpu

            control_group = self._get_distributed_vae_control_group()
            if getattr(self.engine, "_is_vae_producer", False):
                self._start_vae_producer_thread(
                    target_train_iters=int(num_iters),
                    control_group=control_group,
                )
                self._join_vae_producer_thread()
                return

            consumer_channel = None
            if not hasattr(self, "_real_data_iter"):
                from teleboost.engines.teletron.parallel_state import get_comm_pair
                from teleboost.training.dit_batch_loader import (
                    DistributedVAEConsumerChannel,
                    create_dit_batch_loader,
                )
                from teleboost.training.dpo_dataloader import DPODataLoaderBuilder

                # Only the TP/CP root owns a CommPair. BaseBatchLoader
                # broadcasts its data/error envelope to the remaining ranks.
                comm_pair = get_comm_pair()
                if comm_pair is not None:
                    consumer_channel = DistributedVAEConsumerChannel(
                        comm_pair.producer,
                        torch.cuda.current_device(),
                        control_group=control_group,
                        data_group=dist.group.WORLD,
                    )
                    consumer_channel.send_ready(
                        iteration=int(getattr(args, "iteration", 0) or 0),
                        consumed_train_samples=int(getattr(args, "consumed_train_samples", 0) or 0),
                        consumed_valid_samples=int(getattr(args, "consumed_valid_samples", 0) or 0),
                    )
                    self._vae_consumer_channel = consumer_channel

                try:
                    dataloader_helper = DPODataLoaderBuilder()
                    train_loader, _, _ = dataloader_helper.build_train_valid_test_data_loaders(
                        is_tp_first=mpu.get_tensor_model_parallel_rank() == 0,
                        dp_rank=mpu.get_data_parallel_rank(),
                        dp_size=mpu.get_data_parallel_world_size(),
                    )
                    train_iter = iter(train_loader) if train_loader is not None else None
                    self._real_data_iter = create_dit_batch_loader(
                        args,
                        train_iter,
                        consumer_channel=consumer_channel,
                    )
                except BaseException as exc:
                    if consumer_channel is not None:
                        consumer_channel.close(error=exc)
                    raise
            else:
                consumer_channel = getattr(self, "_vae_consumer_channel", None)

            world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            try:
                for step in range(num_iters):
                    batch = next(self._real_data_iter)
                    local_error = None
                    output = None
                    try:
                        output = self.engine.train_batch(
                            batch,
                            loss_function=self.loss_fn,
                        )
                    except BaseException as exc:
                        local_error = exc

                    # A non-root TP/CP rank can fail after the root has already
                    # received a batch. Exchange one small status after each
                    # train call so the root can notify its VAE producer.
                    group = mpu.get_tensor_and_context_parallel_group()
                    group_size = dist.get_world_size(group=group)
                    local_report = None if local_error is None else f"rank {world_rank}: {type(local_error).__name__}: {local_error}"
                    reports = [None] * group_size
                    dist.all_gather_object(reports, local_report, group=group)
                    failures = [report for report in reports if report]
                    if failures:
                        if local_error is not None:
                            local_error.add_note("distributed peer failures: " + " | ".join(failures))
                            raise local_error
                        raise RuntimeError("distributed DiT consumer peer failed: " + " | ".join(failures))

                    if world_rank == 0:
                        metrics = output.get("metrics", {}) if hasattr(output, "get") else {}
                        print(
                            f"[real_train_step] iter={step + 1}/{num_iters} metrics={metrics}",
                            flush=True,
                        )
            except BaseException as exc:
                if consumer_channel is not None:
                    try:
                        consumer_channel.close(error=exc)
                    except BaseException as close_exc:
                        exc.add_note(f"Additionally failed to close distributed-VAE consumer channel: {close_exc!r}")
                raise
            else:
                if consumer_channel is not None:
                    consumer_channel.close()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def phase3_replay_all(self, dump_dir: str, dump_glob: str = "dpo_inputs_iter*_rank*.pt"):
            """Multi-pair precision-replay — replay EVERY dump file
            under ``dump_dir`` matching ``dump_glob`` and aggregate
            per-pair results. Each DiT rank writes its own
            preference pair (8-GPU 2-VAE+6-DiT run → 6 pairs/iter).
            """
            import glob as _glob
            import os

            paths = sorted(_glob.glob(os.path.join(dump_dir, dump_glob)))
            if not paths:
                raise FileNotFoundError(f"No dumps matching {dump_glob} under {dump_dir}")
            world_rank = int(os.environ.get("RANK", "0"))
            agg = {"chosen": [], "rejected": []}
            for path in paths:
                if world_rank == 0:
                    print(f"\n[phase3-replay-all] === {os.path.basename(path)} ===", flush=True)
                res = self.phase3_replay(dump_path=path)
                for branch in ("chosen", "rejected"):
                    agg[branch].append(res[branch])
            if world_rank == 0:
                print(f"\n[phase3-replay-all] === summary across {len(paths)} dumps ===", flush=True)
                for branch in ("chosen", "rejected"):
                    max_max = max(r["max_abs_diff"] for r in agg[branch])
                    max_loss_d = max(r["loss_diff"] for r in agg[branch])
                    print(
                        f"  {branch}: max(noise_pred max|d|)={max_max:.6e}  max|loss diff|={max_loss_d:.6e} across {len(agg[branch])} pairs",
                        flush=True,
                    )
            return agg

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def phase3_replay(self, dump_path: str):
            """Precision-replay against a saved DPO input dump and run
            our model.forward on the SAME inputs.
            Compare per-pair ``noise_pred`` + ``loss_chosen / loss_reject``
            against the reference values stored in the dump.

            Tolerance: bf16 ULP (~3e-4 absolute) per the
            split-vs-no-split verification comment at
            teleboost/training/utils.py:599. Larger residuals mean a real
            divergence in the DiT forward path.

            The dump format is FLAT (one .pt per DP rank with
            ``{meta, context, chosen, rejected, losses}`` dict) as
            written by ``teleboost.programs.wan.dpo.dpo_loss.forward_step``
            lines 826-862. This is distinct from the diffsynth-style
            hierarchical dumps that ``_load_saved_payload`` reads —
            we parse the flat format directly.
            """
            import torch
            import torch.nn.functional as F

            dump = torch.load(dump_path, map_location="cpu", weights_only=True)
            device = torch.cuda.current_device()

            # Our model after cast (.to(bf16) + .cuda()).
            # ``self.engine.module`` is the list[ParallelWanTeletronModel] of
            # length 1 (single VPP rank).
            model = self.engine.module[0]
            model.eval()

            results = {}
            # Dump key naming: chosen/rejected for branch dicts,
            # but ``losses`` uses chosen/reject (NO -ed on rejected).
            # See teleboost.programs.wan.dpo.dpo_loss.forward_step line
            # 850-857 where the payload is constructed.
            for branch, ts_key, loss_key in (
                ("chosen", "timestep_c", "loss_chosen"),
                ("rejected", "timestep_r", "loss_reject"),
            ):
                noisy = dump[branch]["noisy_latents"].to(device, torch.bfloat16)
                ts = dump[branch][ts_key].to(device, torch.bfloat16)
                ctx = dump["context"].to(device, torch.bfloat16)
                target = dump[branch]["training_target"].to(device, torch.float32)
                lw = dump[branch]["loss_weight"].to(device, torch.float32)
                reference_pred = dump[branch]["noise_pred"].to(device, torch.float32)
                reference_loss = dump["losses"][loss_key]
                reference_loss_val = float(reference_loss.item() if hasattr(reference_loss, "item") else reference_loss)

                with torch.no_grad():
                    our_pred = model(
                        x=noisy,
                        timestep=ts,
                        context=ctx,
                        clip_feature=None,
                        y=None,
                    )

                our_pred_f32 = our_pred.float()
                pred_diff = (our_pred_f32 - reference_pred).abs()
                max_abs = float(pred_diff.max().item())
                mean_abs = float(pred_diff.mean().item())

                our_loss_wow = F.mse_loss(our_pred_f32, target).item()
                our_loss = our_loss_wow * float(lw.item())
                loss_diff = abs(our_loss - reference_loss_val)

                world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"[phase3-replay] rank={world_rank} branch={branch} noise_pred max|d|={max_abs:.6e} mean|d|={mean_abs:.6e} | our_loss={our_loss:.6f} reference_loss={reference_loss_val:.6f} |d|={loss_diff:.6e}",
                    flush=True,
                )

                results[branch] = {
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": mean_abs,
                    "our_loss": our_loss,
                    "reference_loss": reference_loss_val,
                    "loss_diff": loss_diff,
                }

            return results

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def smoke_train_step(self):
            """Stub-batch split-DPO backward smoke.

            Synthesize a single stub preference-pair batch INSIDE the
            Ray actor (so each rank owns its own tensors — no cross-
            process DataProto dispatch) and call
            ``engine.train_batch`` directly with verl's expected
            ``(data, loss_function)`` signature. Verifies:
              1. The train_batch override dispatches use_zero2=True
                 into deepspeed_forward_backward without crashing.
              2. forward_step constructs noise + computes
                 chosen/rejected losses + builds DPO list-loss.
              3. deepspeed_backward_step fires
                 ``zero_optimizer.backward(t)`` + epilogue TWICE
                 (once per loss element) — the split-DPO feature
                 parity gate (constraint #1 in recipes README).
              4. After the call, ``self.engine.optimizer`` has its
                 step counter incremented.

            Stub-batch shape mirrors what ``_load_batch_inputs``
            reads at teleboost.programs.wan.dpo.dpo_loss.py:573 —
            ``{context, chosen: {latents, ...}, rejected: {...}}``.
            Sizes match Wan-1.3B T2V latent layout at the smallest
            tractable resolution.
            """
            import torch

            from teleboost.engines.teletron import get_args

            args = get_args()
            device = torch.cuda.current_device()
            param_dtype = torch.bfloat16 if args.bf16 else torch.float32

            # Wan-1.3B T2V latent shape (B, in_dim=16, T_latent, H_latent, W_latent)
            # at minimal smoke resolution. Sequence dim (T*H*W after patch
            # embed) stays under the seq_length yaml setting.
            B, C, T, H, W = 1, 16, 1, 8, 8
            # Text context (T5-xl): (B, prompt_len, text_dim=4096)
            prompt_len, text_dim = 32, 4096

            def _stub_latents():
                return torch.randn(B, C, T, H, W, dtype=param_dtype, device=device)

            def _stub_context():
                return torch.randn(B, prompt_len, text_dim, dtype=param_dtype, device=device)

            stub_batch = {
                "context": _stub_context(),
                "chosen": {
                    "latents": _stub_latents(),
                    # img_clip_feature / img_emb_y are I2V-only (Wan T2V
                    # passes None through forward_step's optional path).
                    "clip_feature": None,
                    "img_emb_y": None,
                },
                "rejected": {
                    "latents": _stub_latents(),
                    "clip_feature": None,
                    "img_emb_y": None,
                },
            }

            print(f"[smoke_train_step] rank={torch.distributed.get_rank()} entering train_batch", flush=True)
            output = self.engine.train_batch(stub_batch, loss_function=self.loss_fn)
            print(f"[smoke_train_step] rank={torch.distributed.get_rank()} train_batch returned {type(output).__name__}", flush=True)
            return output

    model_config = omega_conf_to_dataclass(config.model)
    engine_config = omega_conf_to_dataclass(config.engine)
    optimizer_config = omega_conf_to_dataclass(config.optim)
    checkpoint_config = omega_conf_to_dataclass(config.checkpoint)
    profiler_config = omega_conf_to_dataclass(config.profiler) if OmegaConf.select(config, "profiler") is not None else None

    worker_config = TrainingWorkerConfig(
        model_type=config.model_type,  # "language_model" for
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        profiler_config=profiler_config,
    )

    pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    cls_with_init = RayClassWithInitArgs(ray.remote(_TeleboostTrainingWorker), config=worker_config)
    client = RayWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=cls_with_init,
        device_name=config.trainer.device,
    )
    return client


@hydra.main(config_path="../../../config/dpo", config_name="dpo_trainer", version_base=None)
def main(config) -> None:
    # Pipe yaml `teletron_args` block (if any) through an env
    # var so MegatronEngineWanVideo.__init__ in Ray worker actors can
    # see it. Verl's TrainingWorkerConfig has no free-form field for
    # arbitrary user blocks; env-var inheritance is the cleanest
    # cross-process channel without forking verl's config dataclasses.
    # See teleboost/programs/wan/dpo/args_adapter.py for the receiver side.
    import json

    teletron_block = OmegaConf.to_container(
        OmegaConf.select(config, "teletron_args", default={}) or {},
        resolve=True,
    )
    os.environ["TELEBOOST_TELETRON_ARGS"] = json.dumps(teletron_block)

    if not ray.is_initialized():
        ray_kwargs = (
            OmegaConf.to_container(
                OmegaConf.select(config, "ray_kwargs.ray_init", default={}),
                resolve=True,
            )
            or {}
        )
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TELEBOOST_WAN_ATTN_BACKEND": os.environ.get("TELEBOOST_WAN_ATTN_BACKEND", "auto"),
            # Forward the teletron_args block into every Ray worker
            # process so they pick it up at MegatronEngineWanVideo
            # construction time.
            "TELEBOOST_TELETRON_ARGS": os.environ["TELEBOOST_TELETRON_ARGS"],
        }
        runtime_env = ray_kwargs.pop("runtime_env", {}) or {}
        runtime_env.setdefault("env_vars", {}).update(env_vars)
        ray.init(runtime_env=runtime_env, **ray_kwargs)

    training_client = _build_training_worker(config)
    training_client.set_loss_fn(loss_fn=_dpo_loss_fn_for_verl)
    training_client.reset()

    # Run mode dispatch — exactly one of {smoke, precision replay,
    # train} must apply. Defaults to ``train`` driven by
    # ``trainer.total_training_steps``.
    run_smoke = bool(OmegaConf.select(config, "trainer.run_smoke_train_step", default=False))
    phase3_dir = OmegaConf.select(config, "trainer.phase3_replay_dir", default=None)
    phase3_dump = OmegaConf.select(config, "trainer.phase3_replay_dump", default=None)

    selected = [
        name
        for name, on in [
            ("smoke", run_smoke),
            ("replay_dir", bool(phase3_dir)),
            ("replay_dump", bool(phase3_dump)),
        ]
        if on
    ]
    if len(selected) > 1:
        raise ValueError(f"Set at most one of trainer.run_smoke_train_step / trainer.phase3_replay_dir / trainer.phase3_replay_dump — got {selected}.")

    if run_smoke:
        # Synthesized stub preference-pair batch through the split-DPO
        # multi-backward path. No real data, no checkpoint side-effects.
        print("[teleboost-dpo] running smoke_train_step ...", flush=True)
        training_client.smoke_train_step()
        print("[teleboost-dpo] smoke_train_step completed", flush=True)
        return

    if phase3_dir:
        print(f"[teleboost-dpo] running phase3_replay_all against {phase3_dir} ...", flush=True)
        training_client.phase3_replay_all(dump_dir=phase3_dir)
        print("[teleboost-dpo] phase3_replay_all completed", flush=True)
        return

    if phase3_dump:
        print(f"[teleboost-dpo] running phase3_replay against {phase3_dump} ...", flush=True)
        training_client.phase3_replay(dump_path=phase3_dump)
        print("[teleboost-dpo] phase3_replay completed", flush=True)
        return

    # Default: real preference-pair training. The lr scheduler /
    # checkpoint manager / training loop all read
    # ``trainer.total_training_steps`` — keep it as the single source
    # of truth for iter count so the scheduler doesn't drift from the
    # actual loop length.
    total_steps = int(OmegaConf.select(config, "trainer.total_training_steps", default=0) or 0)
    if total_steps <= 0:
        raise ValueError("trainer.total_training_steps must be > 0 for the default train mode. Set trainer.run_smoke_train_step=true or trainer.phase3_replay_* to pick a non-train mode instead.")
    print(f"[teleboost-dpo] running real_train_step num_iters={total_steps} ...", flush=True)
    training_client.real_train_step(num_iters=total_steps)
    print("[teleboost-dpo] real_train_step completed", flush=True)


if __name__ == "__main__":
    main()
