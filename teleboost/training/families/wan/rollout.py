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
"""
Rollout with huggingface models.

FSDP HybridShard / HYBRID_SHARD_ZERO2 is *unsupported* — under those
strategies the sampling loop deadlocks instead of producing latents.
``DiffusionRollout.__init__`` detects an FSDP-wrapped module on a
hybrid strategy and fails fast with a clear error. To use a hybrid
strategy you would need to materialize a single-GPU model + bind the
full state_dict + run generation on that copy; that refactor isn't
landed.
"""

import logging
import math
import os

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from tqdm.auto import tqdm
from verl import DataProto
from verl.utils.device import get_device_id
from verl.workers.rollout.base import BaseRollout  # verl is pip-installed
from wan.modules.vae import WanVAE

from teleboost.algorithms.grpo.sigma_schedule import compute_sde_step
from teleboost.algorithms.wan_transition import (
    compute_wan_pixel_weight_maps_with_fallback,
    compute_flow_grpo_window,
    finalize_wan_transition_fields,
    make_wan_solver_metadata,
    reduce_wan_log_density,
)
from teleboost.models.wan.family import (
    select_wan22_guide_scale as _select_wan22_guide_scale,
)
from teleboost.models.wan.family import wan_seq_len

__all__ = ["DiffusionRollout"]
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_rollout_allocator_mode() -> None:
    """Turn CUDA allocator ``expandable_segments`` OFF for the generation phase.

    Load-bearing against the ~120-step fragmentation OOM: generation's large
    transient buffers fragment expandable segments, so rollout runs with the
    feature off. Uses a private torch API on purpose — there is no public
    per-phase toggle (torch pin: see requirements).
    """
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


def _compute_flow_grpo_window(window_size: int, window_range: tuple[int, int], num_steps: int):
    """Backward-compatible local name for the shared, tested selector."""
    return compute_flow_grpo_window(window_size, window_range, num_steps)


_HYBRID_FSDP_STRATEGIES = frozenset(
    {
        ShardingStrategy.HYBRID_SHARD,
        ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
)


def _check_unsupported_sharding(module: nn.Module) -> None:
    """Reject FSDP HybridShard up front — the sampling loop deadlocks
    under HYBRID_SHARD / HYBRID_SHARD_ZERO2 (the inter-node replicate
    group does not get exercised by the rollout's diffusion sampler).
    """
    if not isinstance(module, FSDP):
        return
    strategy = getattr(module, "sharding_strategy", None)
    if strategy in _HYBRID_FSDP_STRATEGIES:
        raise RuntimeError(f"DiffusionRollout does not support FSDP {strategy.name}; the sampling loop hangs under hybrid sharding. Use FULL_SHARD / SHARD_GRAD_OP (fsdp1) or fsdp2 with a single-dim mesh, or run the rollout on a single-GPU full-state-dict copy of the module.")


def _antithetic_epsilon(ref: torch.Tensor, global_idx: int, n_resp: int, base_seed: int, step: int):
    """Deterministic antithetic Gaussian noise for one rollout sample at one SDE step.

    Within a prompt's ``n_resp`` responses (adjacent, ``repeat(interleave=True)``
    layout), responses are paired (0,1),(2,3),...  The two members of a pair share
    a BASE epsilon (same seed derived from group/pair/step, independent of parity)
    and take OPPOSITE signs, so ``eps_odd == -eps_even``.  This cancels the
    first-order ``adv·ε`` term that dominates the variance of the GRPO
    score-function gradient ``g = -(1/σ)·mean(adv·ε·∇μ)`` — without enlarging the
    batch and without changing the objective.

    Returns ``(eps, sign, pair, group)`` with ``eps`` matching ``ref``'s
    shape/dtype/device.
    """
    within = int(global_idx) % n_resp
    pair = within // 2
    sign = 1.0 if (within % 2 == 0) else -1.0
    group = int(global_idx) // n_resp
    seed = (((int(base_seed) & 0x7FFFFFFF) * 1000003 + group) * 1000003 + pair) * 1000003 + int(step)
    seed &= 0x7FFFFFFFFFFFFFFF
    gen = torch.Generator(device=ref.device).manual_seed(int(seed))
    eps = torch.randn(ref.shape, generator=gen, device=ref.device, dtype=ref.dtype)
    return eps * sign, sign, pair, group


class DiffusionRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, *, pixel_weight_fn=None, emit_prev_sample_mean: bool = False):
        # Upstream v0.7.1 ``BaseRollout.__init__`` requires
        # ``(config, model_config, device_mesh)`` — args meaningful only to
        # vLLM/SGLang rollouts. Diffusion rollout is FSDP-direct, so the base
        # ctor's state would be empty. Skip ``super().__init__`` and set our
        # own ``self.config`` below; ABC contract is satisfied by overriding
        # ``resume`` / ``update_weights`` / ``release`` / ``generate_sequences``.
        _check_unsupported_sharding(module)
        self.config = config
        self.module = module
        # bfloat16 (not float16): the Wan VAE's intermediate activations
        # routinely exceed fp16's ~65504 max under autocast, producing NaN
        # decoded videos that silently propagate through the
        # ``(video_frames * 255).astype(uint8)`` cast (warns "invalid value
        # encountered in cast" but yields zeros), then NaN HPS scores ->
        # NaN advantages -> NaN grads.  bfloat16 has the same memory cost
        # as fp16 but the dynamic range of fp32, so VAE decode stays
        # finite.  Verified empirically on Wan2.2-T2V-A14B at sampling
        # steps in {4, 10}.
        vae_dtype = torch.bfloat16
        vae = WanVAE(
            vae_pth=os.path.join(self.config.model.vae_model_path),
            dtype=vae_dtype,
        )
        self.vae_module = vae

        # ----- VIPO: pixel-weight feature flag ---------------------------
        # The recipes worker fully owns VIPO: when enabled it injects a
        # fully-bound ``pixel_weight_fn`` (DINOv2 path / PCA method / sigma
        # already captured), otherwise None. The library reads NO
        # ``pixel_weight.*`` config and carries no training-entry dependency.
        # The presence of the injected fn IS the enable signal (single source
        # of truth). When enabled the rollout preserves spatial dims so dense
        # (T,H,W) log-probs can flow into a pixel-weighted loss; otherwise it
        # keeps the original scalar log-probability behaviour.
        self._pixel_weight_fn = pixel_weight_fn
        # Neutral emission switch: when True the sampling loop also returns
        # the per-step SDE transition means (``prev_sample_mean``) — a
        # latent-sized trajectory, so it is opt-in. The recipes worker decides
        # from its algorithm config (the rollout reads no algorithm flags).
        self._emit_prev_sample_mean = bool(emit_prev_sample_mean)
        self._pixel_enable = pixel_weight_fn is not None

        # ----- σ_t SDE form (DanceGRPO vs Flow-GRPO) ---------------------
        # See ``teleboost/algorithms/grpo/sigma_schedule.py``.  Default
        # ``"dancegrpo"`` keeps the existing constant-eta noise schedule
        # byte-equivalent to the pre-registry implementation.
        self._sigma_form = self.config.actor.get("sigma_form", "dancegrpo")

    # Upstream verl v0.7.1 ``BaseRollout`` added three async lifecycle hooks
    # — ``resume`` / ``update_weights`` / ``release`` — used by vLLM/SGLang
    # rollouts to flip GPU memory between actor-train and rollout-infer
    # roles. Diffusion has no separate inference engine: the FSDP-wrapped
    # ``module`` is the rollout, weights are live, KV cache doesn't exist.
    # No-op stubs satisfy the ABC contract.
    async def resume(self, tags: list[str]):
        return

    async def update_weights(self, weights, **kwargs):
        return

    async def release(self):
        return

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # TempFlow trajectory branching: an entirely separate ODE->(SDE@k)->ODE
        # rollout that emits one DataProto row per branch. Default off -> the
        # normal generate path below is byte-for-byte unchanged.
        if self.config.actor.get("tempflow", {}).get("branch", {}).get("enable", False):
            return self.generate_branched_sequences(prompts)
        _set_rollout_allocator_mode()
        context = prompts.batch["context"]
        context_orig_lengths = prompts.batch["context_orig_lengths"]
        neg_context = prompts.batch["null_context"]
        sigma_schedule = prompts.batch["sigma_schedule"]
        input_latents = prompts.batch["input_latents"]
        latent_shape = input_latents[0].shape
        seq_len = wan_seq_len(latent_shape[1], latent_shape[2], latent_shape[3])

        B = prompts.batch.batch_size[0]

        all_latents = []
        all_log_probs = []
        all_video_frames = []
        all_video_ids = []
        all_prev_sample_mean = []  # stays empty unless the sampler returns means

        flow_cfg = self.config.get("flow_grpo", {})
        if not flow_cfg.get("enable", False):
            window_size = 0
        else:
            window_size = int(flow_cfg.get("sde_window_size", 0) or 0)
        window_range = tuple(flow_cfg.get("sde_window_range", (0, self.config.sampling_steps)))
        flow_window = _compute_flow_grpo_window(window_size, window_range, self.config.sampling_steps)

        # 1 sample/chunk: run_wan_sampling_loop is single-sample (it tracks latents[0]
        # and reassigns latents=[next_latents]); SP>1 with multi-sample chunks silently
        # drops all but sample 0. Ulysses still splits each sample's sequence in the
        # shared block.forward, so SP memory savings are kept.
        batch_indices = torch.chunk(torch.arange(B), B)

        # ---- antithetic (common-random-number) SDE noise: root-fix for the
        # high-variance adv·ε term in the GRPO score-function gradient. Default
        # OFF -> bit-identical to the independent-ε path. ----
        antithetic_on = bool(self.config.actor.get("antithetic_noise", False))
        n_resp = int(self.config.rollout.n)
        antithetic_base_seed = 0
        if antithetic_on:
            if n_resp % 2 != 0:
                raise ValueError(f"antithetic_noise=true requires even rollout.n, got n_resp={n_resp}")
            if int(self.config.rollout.ulysses_sequence_parallel_size) != 1:
                raise ValueError("antithetic_noise=true currently requires ulysses_sequence_parallel_size=1")
            if B % n_resp != 0:
                # Data-parallel sharding must not split a prompt's responses across
                # ranks, or the ±ε pair lands on different ranks (different base_seed)
                # and the pairing breaks. (Pair construction is unit-tested in
                # tests/test_antithetic_variance.py.)
                raise ValueError(f"antithetic_noise: this rank's rollout shard B={B} is not a multiple of n_resp={n_resp} — prompt-groups are split across DP ranks. Run single-GPU or size the batch so each rank holds whole groups (per-rank B % n_resp == 0).")
            antithetic_base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            logger.info(f"[antithetic] enabled: n_resp={n_resp} base_seed={antithetic_base_seed} (within-prompt ±ε pairs)")

        self.vae_module.model.to(get_device_id(), dtype=torch.bfloat16)
        for index, batch_idx in enumerate(batch_indices):
            progress_bar = tqdm(range(0, self.config.sampling_steps), desc="WAN Sampling Progress")
            batch_contexts = [context[i].to(get_device_id()) for i in batch_idx]
            batch_neg_context = [neg_context[i].to(get_device_id()) for i in batch_idx]
            batch_context_orig_lengths = [context_orig_lengths[i] for i in batch_idx]
            batch_input_latents = [input_latents[i] for i in batch_idx]

            # per-chunk antithetic context (chunk size == 1 enforced above when on)
            antithetic_ctx = None
            if antithetic_on:
                antithetic_ctx = {
                    "on": True,
                    "global_idx": int(batch_idx[0].item()),
                    "n_resp": n_resp,
                    "base_seed": antithetic_base_seed,
                }

            for i in range(len(batch_contexts)):
                batch_contexts[i] = batch_contexts[i][: batch_context_orig_lengths[i]]

            wan_outputs = self.run_wan_sampling_loop(
                batch_input_latents,
                progress_bar,
                sigma_schedule[0],
                self.module,
                batch_contexts,
                batch_neg_context,
                seq_len,
                flow_window=flow_window,
                antithetic_ctx=antithetic_ctx,
            )

            if len(wan_outputs) == 5:
                _, final_latents, batch_latents, batch_log_probs, batch_prev_sample_mean = wan_outputs
            else:
                _, final_latents, batch_latents, batch_log_probs = wan_outputs
                batch_prev_sample_mean = None

            all_latents.append(batch_latents.unsqueeze(0))
            all_log_probs.append(batch_log_probs.unsqueeze(0))
            if batch_prev_sample_mean is not None:
                all_prev_sample_mean.append(batch_prev_sample_mean.unsqueeze(0))

            video_frames, preview = self._decode_chunk(final_latents)
            all_video_ids.append(preview)
            all_video_frames.append(video_frames)

        self.vae_module.model.to("cpu", dtype=torch.float32)
        torch.cuda.empty_cache()

        # Everything except all_video_paths is a single tensor
        if len(all_latents) > 1:
            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_video_frames = torch.cat(all_video_frames, dim=0)
            if all_prev_sample_mean:
                all_prev_sample_mean = torch.cat(all_prev_sample_mean, dim=0)
        else:
            all_latents = all_latents[0]
            all_log_probs = all_log_probs[0]
            all_video_frames = all_video_frames[0]
            if all_prev_sample_mean:
                all_prev_sample_mean = all_prev_sample_mean[0]

        if flow_window is None:
            raw_transition_indices = list(range(self.config.sampling_steps))
        else:
            raw_transition_indices = list(range(flow_window[0], flow_window[1]))

        sigma_values = sigma_schedule[0].squeeze()
        timestep_value = [int(sigma_values[i] * 1000) for i in raw_transition_indices]
        timesteps = torch.tensor(
            [timestep_value[:] for _ in range(B)],
            device=get_device_id(),
            dtype=torch.long,
        )

        # Every field below describes the same raw SDE transitions.  The global
        # final transition ends at sigma=0, so remove it from *all* fields only
        # when this rollout/window actually includes it.  Previously only
        # ``timesteps`` was sliced, leaving latent/log-prob lengths inconsistent
        # and making a size-one non-terminal window empty.
        transition_fields = {
            "latents": all_latents[:, :-1],
            "next_latents": all_latents[:, 1:],
            "log_probs": all_log_probs,
            "timesteps": timesteps,
        }
        if isinstance(all_prev_sample_mean, torch.Tensor):
            transition_fields["prev_sample_mean"] = all_prev_sample_mean
        if flow_window is not None:
            transition_fields["timestep_indices"] = (
                torch.tensor(
                    raw_transition_indices,
                    device=get_device_id(),
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .repeat(B, 1)
            )

        transition_fields, _trainable_indices = finalize_wan_transition_fields(
            transition_fields,
            transition_indices=raw_transition_indices,
            num_steps=self.config.sampling_steps,
        )
        latents = transition_fields["latents"]
        batch_dict = {
            "context_orig_lengths": context_orig_lengths,
            "contexts": context,
            "null_context": neg_context,
            "video_frames": all_video_frames,
            "sigma_schedule": sigma_schedule,
            **transition_fields,
        }

        # batch is built after the TransferQueue block — the producer may
        # ``pop("video_frames")`` from batch_dict when enabled so the
        # driver pulls frames directly from TQ.
        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch["video_ids"] = np.array(all_video_ids)
        non_tensor_batch.update(
            make_wan_solver_metadata(
                batch_size=B,
                sigma_form=self._sigma_form,
                pixel_enabled=self._pixel_enable,
            )
        )

        # Phase 2: mirror video_frames into TransferQueue under per-sample
        # keys. The legacy DataProto path stays intact (``batch["video_frames"]``)
        # so reward / actor consumers that haven't been flipped over keep
        # working; consumers that have been wired to TQ pick up the same
        # tensor zero-copy.
        #
        # Gating: ``actor_rollout_ref.transfer_queue.enable`` (mirror of
        # top-level ``transfer_queue.enable`` — pushed into the worker's
        # config slice by Hydra). When False, skip entirely; when True
        # but the ``transfer_queue`` package isn't init'd in this Ray
        # process, kv_put raises and we degrade to legacy-only without
        # crashing the rollout.
        # Gating: align with upstream verl v0.7.1 — read
        # ``TRANSFER_QUEUE_ENABLE`` env var propagated through Ray's
        # runtime_env (see ``main_teleboost._init_ray``). This keeps the
        # rollout worker decoupled from any specific config layout; the
        # toggle reaches every Ray actor uniformly.
        from teleboost.engines.transport import transfer_queue as _tqr

        if _tqr.enabled():
            try:
                _vf_keys = _tqr.put_video_frames([all_video_frames[_i] for _i in range(B)])
                non_tensor_batch["video_frame_tq_keys"] = np.array(_vf_keys, dtype=object)
                # Drop the per-sample video tensor from the DataProto we
                # ship back to the driver. The driver hydrates it via
                # ``_ensure_video_frames(...)`` straight from TQ before
                # the reward / validation paths read it. This is the
                # whole point of Phase 2: a single ``ray.get`` of an
                # 80GB-Hopper video tensor turns into a constant-size
                # uuid list + a peer-to-peer pull at consume time.
                batch_dict.pop("video_frames", None)
                # Warning level so the marker shows up under default
                # ``WARN`` logging without requiring VERL_LOGGING_LEVEL.
                logger.warning(
                    "[teleboost.tq.producer] mirrored %d samples to partition=%s (batch.video_frames popped)",
                    B,
                    _tqr.partition_id(),
                )
            except Exception as _tq_err:
                logger.warning(
                    "[teleboost.tq.producer] mirror failed: %s; keeping batch.video_frames",
                    _tq_err,
                )
        # Rebuild the batch tensordict — batch_dict may have lost video_frames.
        batch = TensorDict(batch_dict, batch_size=B)

        # ----- VIPO: attach per-sample pixel-weight maps -----------------
        # Only executed when the feature flag is on.  We derive the
        # spatial/temporal target sizes from the produced latents so the
        # map aligns with whatever the actor will receive.  Failures in
        # DINOv2 fall back to an all-ones map inside the util. A batch-level
        # failure also gets a correctly-shaped all-ones map below, because the
        # actor is already in dense-log-prob mode and cannot safely continue
        # with a missing field.
        if self._pixel_enable:
            # The injected fn is fully bound (VIPO hyperparameters captured by
            # the recipes worker); the library passes rollout data only. The
            # shared helper guarantees a map is attached even if the whole
            # DINO batch fails.
            pixel_weight_maps, pixel_weight_error = compute_wan_pixel_weight_maps_with_fallback(
                self._pixel_weight_fn,
                videos=all_video_frames,
                latents=latents,
            )
            batch["pixel_weight_maps"] = pixel_weight_maps
            if pixel_weight_error is not None:
                logger.warning(
                    "VIPO pixel_weight_maps computation failed: %s. Using uniform dense weights for this batch.",
                    pixel_weight_error,
                )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @staticmethod
    def _unwrap_model_pred(pred):
        if isinstance(pred, dict) and "rgb" in pred:
            return pred["rgb"][0]
        if isinstance(pred, list):
            return pred[0]
        return pred

    def _compute_model_output_for_wan_step(self, latents, index, sigmas, transformer, context, neg_context, seq_len):
        """CFG model output for ONE denoise step — a deliberate copy of the
        per-step forward inside run_wan_sampling_loop, for the branch driver only
        (so the normal path stays byte-identical). Returns (model_output, tv)."""
        device = latents[0].device
        boundary = getattr(self.config, "wan22_boundary", 0.9)
        base_guide_scale = getattr(self.config, "guide_scale", 5.0)
        B = len(context) if isinstance(context, list) else context.shape[0]
        sigma = sigmas[index]
        timestep_value = int(sigma * 1000)
        timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
        sample_guide_scale = _select_wan22_guide_scale(base_guide_scale, timestep, sigma, boundary)
        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                cond = self._unwrap_model_pred(transformer(x=latents, t=timestep, context=context, seq_len=seq_len))
                uncond = self._unwrap_model_pred(transformer(x=latents, t=timestep, context=neg_context, seq_len=seq_len))
            model_output = uncond + sample_guide_scale * (cond - uncond)
        return model_output, timestep_value

    def _decode_chunk(self, final_latents):
        """VAE-decode one chunk's final latents into [0,1] frames + a uint8 preview.

        Returns ``(video_frames, preview)`` where ``video_frames`` is
        ``(1, C, T, H, W)`` in [0, 1] and ``preview`` is a 15-FPS-subsampled
        ``(T', H, W, C)`` uint8 numpy array for logging.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final_latents_vae = final_latents.to(dtype=torch.float32)
            decoded_videos = self.vae_module.decode([final_latents_vae])
            video_frames = decoded_videos[0]

            # Post-process: normalize from [-1, 1] to [0, 1]
            video_frames = (video_frames + 1.0) / 2.0
            video_frames = torch.clamp(video_frames, 0, 1)

            # Preview thumbnails: subsample at 15 FPS, convert to uint8 HWC.
            assert video_frames.dim() == 4, f"expected (C,T,H,W), got {tuple(video_frames.shape)}"
            preview = video_frames[:, ::15, :, :]
            preview = preview.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
            preview = (preview * 255).astype(np.uint8)

            video_frames = video_frames.unsqueeze(0)
        return video_frames, preview

    def generate_branched_sequences(self, prompts: DataProto) -> DataProto:
        """ODE→(SDE@k)→ODE trajectory-branching rollout. ONE DataProto ROW PER
        BRANCH, each a length-1 trained transition at global step k (so the
        actor's existing step loop runs once per row, no ragged handling).
        reward stays a placeholder — the reward model is NOT called here."""
        from teleboost.algorithms.tempflow.trajectory import (
            BranchPlan,
            select_trainable_branch_points,
        )

        # constraint: branch mode does not yet support VIPO pixel maps / TQ —
        # they change row semantics (rows go from #prompts to #branches); fail
        # loud rather than silently misalign.
        if getattr(self, "_pixel_enable", False):
            raise NotImplementedError("TempFlow branch rollout does not support VIPO pixel maps yet")
        if os.environ.get("TRANSFER_QUEUE_ENABLE") == "1":
            raise NotImplementedError("TempFlow branch rollout does not support TransferQueue yet (set TRANSFER_QUEUE_ENABLE=0)")

        branch_cfg = self.config.actor.get("tempflow", {}).get("branch", {})
        plan = BranchPlan(
            branch_points=branch_cfg.get("branch_points", "early_k"),
            early_k=branch_cfg.get("early_k", 1),
            exploration_k=branch_cfg.get("exploration_k", 2),
            reward_target=branch_cfg.get("reward_target", "final_image"),
        )
        # sampled_k needs caller-provided sampled_indices (kept deterministic);
        # the driver does not generate them, so only all/early_k are wired here.
        if plan.branch_points not in ("all", "early_k"):
            raise NotImplementedError(f"branch rollout supports branch_points in (all, early_k); got {plan.branch_points!r} (sampled_k is not wired into the driver)")
        # the driver only decodes + scores the FINAL video; middle_frame /
        # video_reward are validated by BranchPlan but not wired here — fail loud
        # rather than silently scoring final_image under another name.
        if plan.reward_target != "final_image":
            raise NotImplementedError(f"branch rollout only implements reward_target='final_image'; got {plan.reward_target!r}")
        # exploration_k=1 makes every (prompt, k) group a singleton, so the
        # group-relative advantage is identically 0 → a zero-gradient no-op step.
        # Warn loudly rather than silently train on nothing.
        if plan.exploration_k == 1:
            logger.warning("[tempflow.branch] exploration_k=1 → every branch group is a singleton, so all branched advantages are 0 (no policy gradient). Set exploration_k>=2 for a usable group-relative signal.")
        # constraint: the branch driver defines its OWN ODE-SDE-ODE; the
        # flow_grpo SDE window must not participate. Ignore it (yaml defaults it
        # on) and say so once.
        if self.config.get("flow_grpo", {}).get("enable", False):
            logger.warning("[tempflow.branch] ignoring flow_grpo SDE window: prefix/tail are ODE (eta=0), only the branch step k uses actor.eta")

        context = prompts.batch["context"]
        context_orig_lengths = prompts.batch["context_orig_lengths"]
        neg_context = prompts.batch["null_context"]
        sigmas = prompts.batch["sigma_schedule"][0]  # shared 1-D [T+1] schedule
        input_latents = prompts.batch["input_latents"]
        latent_shape = input_latents[0].shape
        seq_len = wan_seq_len(latent_shape[1], latent_shape[2], latent_shape[3])
        B = prompts.batch.batch_size[0]
        # Globally-unique prompt id per row, stamped by the trainer BEFORE the
        # sharded dispatch. This worker only sees its shard; a local range(B)
        # would collide with the other workers' ids after concat. We index this
        # worker's own (sharded) tensors by the LOCAL p, but tag each branch row
        # with the GLOBAL id so advantage grouping is correct across workers.
        if "branch_global_prompt_id" not in prompts.non_tensor_batch:
            raise KeyError("branch rollout: gen_batch is missing 'branch_global_prompt_id'; the trainer must stamp it before the sharded dispatch (else branch advantage groups collide across GPU workers)")
        global_ids = np.asarray(prompts.non_tensor_batch["branch_global_prompt_id"])
        # Distributed-identity guard: the sharding manager MUST slice the
        # non-tensor metadata to the same B rows as batch tensors. If it ever
        # doesn't, the presence check above still passes but we'd silently reuse
        # the first B ids/captions. Catch that here, not with a corrupt gradient.
        if len(global_ids) != B:
            raise ValueError(f"branch rollout: branch_global_prompt_id length {len(global_ids)} != shard size {B}; non-tensor metadata was not sharded consistently with the batch")
        sampling_steps = self.config.sampling_steps
        actor_eta = float(self.config.actor.eta)
        return_prev = self._emit_prev_sample_mean
        device = get_device_id()
        branch_ks = select_trainable_branch_points(sampling_steps, plan)

        self.vae_module.model.to(device, dtype=torch.bfloat16)

        latent_k_rows, next_rows, logp_rows, mean_rows = [], [], [], []
        ts_rows, tsidx_rows, video_rows = [], [], []
        m_prompt, m_localp, m_k, m_sample, m_row = [], [], [], [], []
        row_id = 0

        for p in range(B):
            ctx_p = [context[p].to(device)[: context_orig_lengths[p]]]
            neg_p = [neg_context[p].to(device)]
            # latents stay float32 between steps (the model forward casts to bf16
            # under autocast) — exactly the normal path's dtype discipline.
            x_T = input_latents[p].to(device).to(torch.float32)
            for k in branch_ks:
                # --- ODE prefix 0..k-1 (deterministic, eta=0, no logprob) ---
                x = x_T
                for i in range(k):
                    mo, _ = self._compute_model_output_for_wan_step([x], i, sigmas, self.module, ctx_p, neg_p, seq_len)
                    nx, _ = self.wan_step(mo, x, 0.0, sigmas, i, None, grpo=False)
                    x = nx.to(torch.float32)
                x_k = x
                for s in range(plan.exploration_k):
                    # --- branch: ONE SDE step at k (stochastic, logprob) ---
                    mo_k, tv_k = self._compute_model_output_for_wan_step([x_k], k, sigmas, self.module, ctx_p, neg_p, seq_len)
                    next_x, _pred, logp, prev_mean = self.wan_step(mo_k, x_k, actor_eta, sigmas, k, None, grpo=True, return_prev_sample_mean=True)
                    next_x = next_x.to(torch.float32)
                    # --- ODE tail k+1..T-1 (deterministic) from next_x ---
                    xt = next_x
                    for i in range(k + 1, sampling_steps):
                        mo, _ = self._compute_model_output_for_wan_step([xt], i, sigmas, self.module, ctx_p, neg_p, seq_len)
                        nx, _ = self.wan_step(mo, xt, 0.0, sigmas, i, None, grpo=False)
                        xt = nx.to(torch.float32)
                    final_latent = xt
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        dec = self.vae_module.decode([final_latent.to(torch.float32)])
                    video = torch.clamp((dec[0] + 1.0) / 2.0, 0, 1)

                    latent_k_rows.append(x_k)
                    next_rows.append(next_x)
                    # Baseline wan_step reduces this unbatched latent to one
                    # scalar log-prob. Stack -> [n], then unsqueeze -> [n, 1]
                    # to match the actor's [B, T] rollout contract.
                    logp_rows.append(logp)
                    if return_prev:
                        mean_rows.append(prev_mean)
                    ts_rows.append(tv_k)
                    tsidx_rows.append(k)
                    video_rows.append(video)
                    m_prompt.append(int(global_ids[p]))  # GLOBAL id (grouping key)
                    m_localp.append(p)  # LOCAL shard index (data lookup)
                    m_k.append(k)
                    m_sample.append(s)
                    m_row.append(row_id)
                    row_id += 1

        self.vae_module.model.to("cpu", dtype=torch.float32)
        torch.cuda.empty_cache()

        n = row_id
        # Scale is NOT just prompt_bsz: rollout.n already repeated the prompt rows
        # before this call (each repeat is its own prompt_id), so branch rows =
        # B(repeated) x branch_ks x exploration_k, and advantage groups are
        # (prompt_id, k) — at the repeated-row level, not the original prompt.
        logger.info(
            "[tempflow.branch] %d branch rows = B(%d) x branch_ks(%d) x exploration_k(%d); rollout.n multiplies B via repeated prompt rows",
            n,
            B,
            len(branch_ks),
            plan.exploration_k,
        )
        # length-1 trained step per row: [n, 1, ...]
        batch_dict = {
            "latents": torch.stack(latent_k_rows, dim=0).unsqueeze(1),
            "next_latents": torch.stack(next_rows, dim=0).unsqueeze(1),
            "log_probs": torch.stack(logp_rows, dim=0).unsqueeze(1),
            "video_frames": torch.stack(video_rows, dim=0),
            "sigma_schedule": sigmas.unsqueeze(0).repeat(n, 1),
            "timesteps": torch.tensor(ts_rows, device=device, dtype=torch.long).unsqueeze(1),
            "timestep_indices": torch.tensor(tsidx_rows, device=device, dtype=torch.long).unsqueeze(1),
            # per-row prompt context, indexed by the LOCAL shard index (these are
            # this worker's sharded tensors), so the actor's recompute aligns to
            # the branch's prompt. (m_prompt is the GLOBAL grouping id and must
            # NOT index these local tensors.)
            "contexts": context[torch.tensor(m_localp, device=context.device)],
            "null_context": neg_context[torch.tensor(m_localp, device=neg_context.device)],
            "context_orig_lengths": context_orig_lengths[torch.tensor(m_localp)],
        }
        if return_prev:
            batch_dict["prev_sample_mean"] = torch.stack(mean_rows, dim=0).unsqueeze(1)

        batch = TensorDict(batch_dict, batch_size=n)
        # NOTE: only plain numpy metadata in non_tensor_batch. The rich
        # BranchSample objects hold GPU tensors; shipping them through Ray to the
        # CPU driver triggers a CUDA-deserialize error. All training tensors are
        # already proper DataProto tensors in batch_dict, and identity lives in
        # these parallel arrays — which is exactly what M3 reads.
        m_localp_np = np.array(m_localp)
        non_tensor_batch = {
            "prompt_id": np.array(m_prompt),  # GLOBAL id — the grouping key
            "branch_timestep_index": np.array(m_k),
            "sample_id": np.array(m_sample),
            "branch_row_id": np.array(m_row),  # worker-local; kept for debug only
        }
        # The actor's SolverContract validation reads per-row solver metadata;
        # the normal path inherits it via prompts.non_tensor_batch, but this
        # fresh dict must emit it for every branch ROW or update_policy raises.
        non_tensor_batch.update(
            make_wan_solver_metadata(
                batch_size=int(m_localp_np.size),
                sigma_form=self._sigma_form,
                pixel_enabled=self._pixel_enable,
            )
        )
        # CRITICAL: the reward path reads non_tensor_batch["caption"] (qwen judge
        # + single RM). The normal path inherits prompts.non_tensor_batch wholesale;
        # the branched path builds a fresh dict, so per-prompt reward metadata must
        # be re-scattered to each branch row — indexed by the LOCAL shard index
        # (caption is this worker's sharded array) — else the judge scores every
        # video against an EMPTY caption and the reward is garbage.
        src_ntb = prompts.non_tensor_batch
        for key in ("caption", "video_ids"):
            if key in src_ntb:
                arr = np.asarray(src_ntb[key])
                # same distributed-identity guard as global_ids: the local index
                # only addresses this shard's B rows, so the source array must be
                # length B (consistently sharded), else we'd reuse the first rows.
                if len(arr) != B:
                    raise ValueError(f"branch rollout: non_tensor '{key}' length {len(arr)} != shard size {B}; metadata was not sharded consistently with the batch")
                non_tensor_batch[key] = arr[m_localp_np]
        if "caption" not in non_tensor_batch:
            raise KeyError(f"branch rollout: prompts.non_tensor_batch has no 'caption'; the reward judge would score against empty prompts (keys={list(src_ntb.keys())})")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def run_wan_sampling_loop(
        self,
        latents,  # [(16, 7, 64, 64)]
        progress_bar,
        sigma_schedule,
        transformer,
        context,
        neg_context,
        seq_len,
        flow_window=None,
        antithetic_ctx=None,
    ):
        """Run the full Wan SDE sampling loop. Latent input layout is (C, T, H, W)."""
        all_latents = []

        all_log_probs = []
        return_prev_sample_mean = self._emit_prev_sample_mean
        all_prev_sample_mean = [] if return_prev_sample_mean else None
        B = len(context) if isinstance(context, list) else context.shape[0]
        device = latents[0].device
        boundary = getattr(self.config, "wan22_boundary", 0.9)
        base_guide_scale = getattr(self.config, "guide_scale", 5.0)

        if flow_window is None:
            window_start = 0
            window_end = self.config.sampling_steps
        else:
            window_start, window_end = flow_window

        for i in progress_bar:
            sigma = sigma_schedule[i]

            timestep_value = int(sigma * 1000)
            timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
            sample_guide_scale = _select_wan22_guide_scale(base_guide_scale, timestep, sigma, boundary)

            with torch.autocast("cuda", torch.bfloat16):
                # Wan model input: x is a list of (C, T, H, W) tensors
                with torch.no_grad():
                    pred_cond = transformer(
                        x=latents,  # [(16, 7, 64, 64)]
                        t=timestep,
                        context=context,
                        seq_len=seq_len,
                    )
                if isinstance(pred_cond, dict) and "rgb" in pred_cond:
                    model_output_cond = pred_cond["rgb"][0]
                elif isinstance(pred_cond, list):
                    model_output_cond = pred_cond[0]
                else:
                    model_output_cond = pred_cond

                # Unconditional prediction

                with torch.no_grad():
                    pred_uncond = transformer(
                        x=latents,  # [(16, 7, 64, 64)]
                        t=timestep,
                        context=neg_context,
                        seq_len=seq_len,
                    )

                if isinstance(pred_uncond, dict) and "rgb" in pred_uncond:
                    model_output_uncond = pred_uncond["rgb"][0]
                elif isinstance(pred_uncond, list):
                    model_output_uncond = pred_uncond[0]
                else:
                    model_output_uncond = pred_uncond

                del pred_cond, pred_uncond

                # CFG combine
                model_output = model_output_uncond + sample_guide_scale * (model_output_cond - model_output_uncond)
                del model_output_cond, model_output_uncond
                torch.cuda.empty_cache()

            in_window = window_start <= i < window_end
            if i == window_start:
                all_latents.append(latents[0])

            if in_window:
                if return_prev_sample_mean:
                    next_latents, pred_original, log_prob, prev_sample_mean = self.wan_step(
                        model_output,
                        latents[0].to(torch.float32),  # (16, 7, 64, 64)
                        self.config.actor.eta,
                        sigma_schedule,
                        i,
                        prev_sample=None,
                        grpo=True,
                        return_prev_sample_mean=True,
                        antithetic_ctx=antithetic_ctx,
                    )
                    all_prev_sample_mean.append(prev_sample_mean)
                else:
                    next_latents, pred_original, log_prob = self.wan_step(
                        model_output,
                        latents[0].to(torch.float32),  # (16, 7, 64, 64)
                        self.config.actor.eta,
                        sigma_schedule,
                        i,
                        prev_sample=None,
                        grpo=True,
                        antithetic_ctx=antithetic_ctx,
                    )
                all_log_probs.append(log_prob)
                all_latents.append(next_latents.to(torch.float32))
            else:
                # Outside the SDE window we want a deterministic (ODE
                # Euler) step.  Setting ``eta=0.0`` zeros both the
                # score-correction term *and* the Gaussian noise std
                # in either σ_t form, so the step degenerates cleanly
                # to ``latents + dsigma · model_output``.
                next_latents, pred_original = self.wan_step(
                    model_output,
                    latents[0].to(torch.float32),
                    0.0,
                    sigma_schedule,
                    i,
                    prev_sample=None,
                    grpo=False,
                )

            latents = [next_latents.to(torch.float32)]
        final_latents = pred_original

        # all_latents shape is (num_steps+1, 16, 7, 64, 64)
        all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
        if return_prev_sample_mean:
            all_prev_sample_mean = torch.stack(all_prev_sample_mean, dim=0)
            return latents, final_latents, all_latents, all_log_probs, all_prev_sample_mean

        return latents, final_latents, all_latents, all_log_probs

    def wan_step(
        self,
        model_output: torch.Tensor,  # model-predicted flow
        latents: torch.Tensor,  # current-timestep latents (16, 7, 64, 64)
        eta: float,  # randomness strength
        sigmas: torch.Tensor,  # sigma schedule (FLUX-style)
        index: int,  # current timestep index
        prev_sample: torch.Tensor,  # previous-step sample (for GRPO re-computation)
        grpo: bool,  # True -> also return logprob
        return_prev_sample_mean: bool = False,
        antithetic_ctx=None,  # {on, global_idx, n_resp, base_seed} -> antithetic ε
    ):
        """One Wan Flow-Matching sampling step, recast as an SDE solver for GRPO."""

        if grpo and float(eta) <= 0.0:
            # The SDE log-prob density divides by std_dev_t**2 = (eta*sqrt(dt))**2;
            # eta=0 makes it 0/0 = NaN log-probs with no error.  grpo=True call
            # sites pass actor.eta — a zero there is a misconfiguration, not a
            # valid ODE request (use grpo=False for deterministic steps).
            raise ValueError(f"wan_step(grpo=True) requires eta > 0: the Gaussian log-prob density is undefined at std_dev_t=0 (got eta={eta}).")

        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]

        # Predicted original sample (universal flow-matching geometry; used
        # by callers and by the DanceGRPO score-correction inside the
        # registry).
        pred_original_sample = latents - sigma * model_output

        # Dispatch to the σ_t form's SDE step (DanceGRPO constant-η or
        # Flow-GRPO t-dependent).  ``std_dev_t`` is the effective Gaussian
        # std for both noise injection and the log-prob density.  Pure
        # ODE Euler is reached via ``eta=0.0`` (e.g. outside the SDE
        # window above), which zeros both the score correction and the
        # noise std in either form.
        prev_sample_mean, std_dev_t, _sqrt_dt = compute_sde_step(
            form=self._sigma_form,
            model_output=model_output,
            latents=latents,
            eta=eta,
            sigma=sigma,
            sigma_next=sigma_next,
            pred_original_sample=pred_original_sample,
        )

        if grpo and prev_sample is None:
            # Default: independent ε per sample (bit-identical to the original).
            # Antithetic: within-prompt-group ±ε pairs to cancel the adv·ε noise.
            if antithetic_ctx is not None and antithetic_ctx.get("on"):
                eps, _sign, _pair, _grp = _antithetic_epsilon(
                    prev_sample_mean,
                    antithetic_ctx["global_idx"],
                    antithetic_ctx["n_resp"],
                    antithetic_ctx["base_seed"],
                    index,
                )
            else:
                eps = torch.randn_like(prev_sample_mean)
            prev_sample = prev_sample_mean + eps * std_dev_t

        if grpo:
            log_prob = (-((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))) - torch.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # Shared with actor recompute. In baseline mode an unbatched
            # (C,T,H,W) latent is one sample and must become one scalar; only a
            # genuine 5-D input retains its leading B dimension.
            log_prob = reduce_wan_log_density(
                log_prob,
                pixel_enabled=self._pixel_enable,
            )
            if return_prev_sample_mean:
                return prev_sample, pred_original_sample, log_prob, prev_sample_mean
            return prev_sample, pred_original_sample, log_prob
        else:
            if return_prev_sample_mean:
                return prev_sample_mean, pred_original_sample, prev_sample_mean
            return prev_sample_mean, pred_original_sample
