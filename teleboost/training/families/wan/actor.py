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
import functools
import logging
import math
import os
from dataclasses import dataclass

import torch

# Direct worker deserialization can import this actor without fsdp_worker.
from teleboost.training.families.wan.diag_helpers import (
    to_metric_value as _to_metric_value,
)
from teleboost.patches.lifecycle import PATCHES_APPLIED as _PATCHES_APPLIED
from teleboost.algorithms.tempflow.noise import REWEIGHT_MODES, resolve_timestep_weights
from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.workers.actor import DataParallelPPOActor

from teleboost.models.wan.family import LATENT_CHANNELS, wan_seq_len
from teleboost.models.wan.family import (
    select_wan22_guide_scale as _select_wan22_guide_scale,
)
from teleboost.algorithms.grpo_guard import (
    GRAD_REWEIGHT_FORMS,
    compute_grad_reweight_delta,
    compute_ratio_norm_bias,
)
from teleboost.algorithms.grpo.loss import grpo_policy_loss
from teleboost.algorithms.grpo.sigma_schedule import compute_sde_step
from teleboost.algorithms.solver_contract import SolverContract
from teleboost.algorithms.wan_transition import (
    align_wan_log_probs_for_loss,
    reduce_wan_log_density,
    validate_wan_solver_metadata,
)

if not _PATCHES_APPLIED:  # pragma: no cover - bootstrap raises first
    raise RuntimeError("GRPO actor runtime patches were not installed")

__all__ = ["DiffusionDataParallelPPOActor"]


@dataclass
class _DenoisePlan:
    """Per-update denoise scaffolding produced by ``_denoise_setup``."""

    dataloader: list
    perms: torch.Tensor
    train_timesteps: int
    gradient_accumulation: int
    device: torch.device


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _clip_grad_norm(module, max_norm):
    """Global-norm gradient clip that is FSDP-aware.

    FSDP shards the gradients, so the vanilla ``torch.nn.utils.clip_grad_norm_``
    computes a rank-LOCAL shard norm and every rank clips by a different factor,
    silently desynchronizing the parameter shards (and the logged grad_norm is a
    local value). FSDP's own ``clip_grad_norm_`` all-reduces the global norm
    first — a symmetric collective every rank issues at this same point.
    """
    fsdp_clip = getattr(module, "clip_grad_norm_", None)
    if callable(fsdp_clip):
        return fsdp_clip(max_norm)
    return torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)


class DiffusionDataParallelPPOActor(DataParallelPPOActor):
    # ------------------------------------------------------------------
    # VIPO (pixel-weighted advantage) helpers
    # ------------------------------------------------------------------
    # These small helpers avoid repeating the ``config.get("pixel_weight")``
    # boilerplate in every method and keep the flag name in one place.
    # When the flag is off the actor runs the original scalar GRPO path
    # bit-for-bit identical to the pre-merge baseline.

    def _pixel_cfg(self):
        return self.config.get("pixel_weight", {}) or {}

    def _pixel_enabled(self) -> bool:
        return bool(self._pixel_cfg().get("enable", False))

    def _mismatch_diagnostics_cfg(self):
        return self.config.get("mismatch_diagnostics", {}) or {}

    def _mismatch_diagnostics_enabled(self) -> bool:
        return bool(self._mismatch_diagnostics_cfg().get("enable", True))

    @staticmethod
    def _record_mismatch_metrics(
        metrics,
        *,
        prefix,
        log_prob_delta,
        ratio,
        clipped_mask,
        timestep=None,
        eps=1e-8,
    ) -> None:
        """Record rollout-vs-training policy mismatch diagnostics.

        ``ratio`` is the exact policy ratio used by the GRPO loss after any
        configured RatioNorm adjustment, so these metrics describe the
        effective off-policy pressure seen by the optimizer.
        """
        with torch.no_grad():
            delta = log_prob_delta.detach().float()
            weights = ratio.detach().float()
            clipped = clipped_mask.detach().float()

            weights_flat = weights.reshape(-1)
            n = max(weights_flat.numel(), 1)
            weight_sum = weights_flat.sum()
            weight_sq_sum = weights_flat.square().sum()
            ess = (weight_sum.square() / (weight_sq_sum + eps)) / n

            data_dict = {
                f"{prefix}/logprob_diff_mean": delta.mean(),
                f"{prefix}/logprob_diff_abs_mean": delta.abs().mean(),
                f"{prefix}/logprob_diff_max_abs": delta.abs().max(),
                f"{prefix}/ratio_mean": weights.mean(),
                f"{prefix}/ratio_std": weights.std(unbiased=False),
                f"{prefix}/ratio_max": weights.max(),
                f"{prefix}/chi2": (weights - 1.0).square().mean(),
                f"{prefix}/ess": ess,
                f"{prefix}/clipped_frac": clipped.mean(),
            }
            if timestep is not None:
                data_dict[f"{prefix}/timestep"] = timestep.float().mean()

            append_to_dict(metrics, {key: _to_metric_value(value) for key, value in data_dict.items()})

    @functools.cached_property
    def _sigma_form(self) -> str:
        """SDE σ_t form (cached at first access).

        Mirrors ``DiffusionRollout._sigma_form`` so the rollout and the
        actor's own ``wan_step`` agree on the σ_t convention.  See
        ``teleboost/algorithms/grpo/sigma_schedule.py``.
        """
        return self.config.get("sigma_form", "dancegrpo")

    @staticmethod
    def _build_perms(timesteps: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        seq_len = len(timesteps[0])
        if shuffle:
            return torch.stack([torch.randperm(seq_len) for _ in range(timesteps.shape[0])])
        return torch.stack([torch.arange(seq_len) for _ in range(timesteps.shape[0])])

    @staticmethod
    def _broadcast_perms(perms: torch.Tensor) -> None:
        from verl.utils.ulysses import (
            get_ulysses_sequence_parallel_group,
            get_ulysses_sequence_parallel_world_size,
        )

        if get_ulysses_sequence_parallel_world_size() > 1:
            sp_size = get_ulysses_sequence_parallel_world_size()
            src_rank = (torch.distributed.get_rank() // sp_size) * sp_size
            torch.distributed.broadcast(perms, src=src_rank, group=get_ulysses_sequence_parallel_group())
            torch.distributed.barrier()

    @staticmethod
    def _reorder_batch_by_perms(data: DataProto, perms: torch.Tensor, keys) -> None:
        batch_idx = torch.arange(data.batch.batch_size[0])[:, None]
        for key in keys:
            data.batch[key] = data.batch[key][batch_idx, perms]

    @staticmethod
    def _prepare_contexts(td, device):
        ctx_lens = td["context_orig_lengths"].tolist() if torch.is_tensor(td["context_orig_lengths"]) else td["context_orig_lengths"]
        ctxs_cpu = [td["contexts"][i][: int(ctx_lens[i])] for i in range(len(ctx_lens))]
        nctx_cpu = [td["null_context"][i] for i in range(len(ctx_lens))]
        ctxs = [c.to(device) for c in ctxs_cpu]
        nctxs = [c.to(device) for c in nctx_cpu]
        return ctxs, nctxs

    @staticmethod
    def _calc_seq_len(latents: torch.Tensor) -> int:
        # (B, C, T, H, W) -> token count under the shared Wan patchification.
        latent_shape = latents.shape
        return wan_seq_len(latent_shape[2], latent_shape[3], latent_shape[4])

    # Shared denoise-update mechanics: permutation collectives, step selection,
    # chunking, and the gradient-accumulation boundary.
    def _denoise_setup(self, data, *, permute_keys, select_keys, non_tensor_select_keys, shuffle):
        """Build+broadcast per-sample step perms (distributed collective), reorder the
        batch, compute train_timesteps, chunk into per-sample micro-batches, resolve
        device + gradient_accumulation. Byte-identical to the inline sequence it replaces."""
        perms = self._build_perms(data.batch["timesteps"], shuffle=shuffle)
        self._broadcast_perms(perms)
        self._reorder_batch_by_perms(data, perms, permute_keys)

        # The rollout already trims the final sigma->0 step (peaked log-prob -> NaN), so
        # the pool is len(timesteps)=sampling_steps-1; max(1,...) keeps tiny test configs alive.
        timestep_count = len(data.batch["timesteps"][0])
        if timestep_count <= 0:
            raise RuntimeError("No trainable timesteps in batch. The rollout produced len(timesteps)=0; bump actor_rollout_ref.sampling_steps to >=2 (the rollout drops the final sigma->0 step).")
        train_timesteps = max(1, int(timestep_count * self.config.timestep_fraction))
        gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
        # Kept as an attribute too: the inline GRPO body reads ``self.gradient_accumulation``.
        self.gradient_accumulation = gradient_accumulation
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(data.batch.batch_size[0])
        device = torch.device(f"cuda:{get_device_id()}")
        perms = perms.to(device)
        return _DenoisePlan(
            dataloader=dataloader,
            perms=perms,
            train_timesteps=train_timesteps,
            gradient_accumulation=gradient_accumulation,
            device=device,
        )

    def _denoise_step_inputs(self, batch_on_device, perms, batch_idx, step_idx):
        """Extract this denoising step's tensors."""
        latent_t = batch_on_device.batch["latents"][:, step_idx]
        nlatent_t = batch_on_device.batch["next_latents"][:, step_idx]
        t_t = batch_on_device.batch["timesteps"][:, step_idx]
        sigma_0 = batch_on_device.batch["sigma_schedule"][0]
        seq_len = self._calc_seq_len(latent_t)
        if "timestep_indices" in batch_on_device.batch:
            step_indices = batch_on_device.batch["timestep_indices"][0, step_idx]
        else:
            step_indices = perms[batch_idx][step_idx]
        return latent_t, nlatent_t, t_t, sigma_0, seq_len, step_indices

    def _denoise_optimizer_step(self, step_counter, gradient_accumulation, metrics):
        """Apply the GRPO gradient-accumulation boundary: every
        ``gradient_accumulation`` micro-batches, clip + optimizer.step() + zero_grad().
        This is the drift-prone part (expandable_segments / FSDP forward_prefetch /
        fused AdamW / the zero_grad-placement P0 all live here) — kept in ONE place."""
        if (step_counter + 1) % gradient_accumulation == 0:
            grad_norm = _clip_grad_norm(self.actor_module, self.config.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        # Read the dense/scalar mode before constructing the solver contract:
        # it determines the exact log-prob reduction recorded by rollout.
        pixel_enable = self._pixel_enabled()
        pixel_cfg = self._pixel_cfg()

        # SolverContract startup guard (Patch B). Build the expected contract
        # from config once and validate it: sigma_form must be registered, and
        # the rollout and recompute transitions must match (the GRPO ratio is
        # only valid when the policy log-prob is recomputed with the same
        # transition that drew the trajectory). Wan drives both rollout and
        # recompute with its own SDE sampler, so this is a no-op for valid
        # configs and a clear hard-fail for a bad ``sigma_form``.
        if getattr(self, "_solver_contract", None) is None:
            self._solver_contract = SolverContract(
                name=f"wan_{self._sigma_form}",
                sigma_form=self._sigma_form,
                rollout_transition="wan_sde",
                recompute_transition="wan_sde",
                eval_transition="wan_ode",
                logprob_reduction=("channel_sum_dense" if pixel_enable else "mean"),
            )
        validate_wan_solver_metadata(
            data.non_tensor_batch,
            batch_size=data.batch.batch_size[0],
            expected_contract=self._solver_contract,
        )

        # ----- VIPO: runtime guards & feature detection -------------------

        # GRPO-Guard options.  Paper: arxiv 2510.22319.  RatioNorm (Eq. 8)
        # rewrites the importance-sampling ratio with a Δμ-derived bias term
        # plus an outer ``σ_t · √Δt`` scale; grad-reweight ``δ`` rescales the
        # final policy loss so the gradient magnitude is dt-invariant.  The
        # paper's §4.3 ablation treats RatioNorm and grad-reweight as
        # **separable** levers (Mean-revised / RatioNorm / GRPO-Guard combined),
        # so we expose them as two flags:
        #
        #   ratio_norm       — apply Eq. 8 to the policy ratio
        #   grad_reweight    — rescale the policy loss by δ (= β/dt)
        #
        # ``grad_reweight_form`` selects between the paper's two δ shapes
        # (default null => follows sigma_form; see the read below):
        #
        #   flow_grpo:  δ = 1/dt                      (β ≈ const, Flow-GRPO style)
        #   dancegrpo:  δ = (1 + η²(1−t)/(2t)) / dt   (DanceGRPO form)
        #
        # Both ``ratio_norm`` and ``grad_reweight`` default to ``guard_enable``
        # so legacy ``grpo_guard.enable=true`` configs keep their behaviour
        # (the previous code bundled grad_reweight inside the ``if ratio_norm``
        # branch with a hardcoded ``policy_loss /= sqrt_dt^2`` — i.e. the
        # flow_grpo form).
        guard_cfg = self.config.get("grpo_guard", {})
        guard_enable = guard_cfg.get("enable", False)
        ratio_norm = guard_cfg.get("ratio_norm", guard_enable)
        ratio_norm_eps = guard_cfg.get("ratio_norm_eps", 1e-6)
        grad_reweight = guard_cfg.get("grad_reweight", guard_enable)
        grad_reweight_eps = guard_cfg.get("grad_reweight_eps", 1e-6)
        # Paper ties the reweight δ to the SDE kernel's σ_t (Eq.12): dancegrpo
        # kernel => δ=(1+η²(1−t)/(2t))/dt, flow_grpo => δ=1/dt. So the reweight
        # form must MATCH sigma_form. null/absent config => follow sigma_form; an
        # explicit value overrides (and is validated below) for ablations. Test
        # ``is None`` (not truthiness) so a bad explicit value still hits the raise.
        grad_reweight_form = guard_cfg.get("grad_reweight_form")
        if grad_reweight_form is None:
            grad_reweight_form = self._sigma_form

        if grad_reweight_form not in GRAD_REWEIGHT_FORMS:
            # Fail fast at config read so a typo doesn't silently pass on
            # ``grad_reweight=False`` runs (the helper would only catch it
            # on the first actual δ call).
            raise ValueError(f"Unsupported grpo_guard.grad_reweight_form={grad_reweight_form!r}; valid forms: {sorted(GRAD_REWEIGHT_FORMS.keys())} (see arxiv 2510.22319 §3.2.3).")

        # Dense pixel-weighted advantages require a matching dense KL path.
        use_kl_loss = bool(self.config.get("use_kl_loss", False))
        if pixel_enable and use_kl_loss and not bool(pixel_cfg.get("kl_loss_compatible", False)):
            raise NotImplementedError("VIPO pixel-weighted advantages are incompatible with the KL loss path. Set `actor_rollout_ref.actor.use_kl_loss=false` or flip `actor_rollout_ref.pixel_weight.kl_loss_compatible=true` once the dense-KL path has been implemented and tested.")

        # TempFlow Eq. 8 noise-aware per-timestep loss weighting (paper
        # arXiv 2508.04324). Default "none" ⇒ Wan-identical (weight 1 every
        # step). The [T] mean-1 weights are derived once per micro-batch from
        # the rollout's own sigma schedule (below) so they match the actual
        # transition stds; the math lives in teleboost.algorithms.tempflow.noise.
        tempflow_cfg = self.config.get("tempflow", {})
        noise_reweight_mode = tempflow_cfg.get("noise_reweight_mode", "none")
        if noise_reweight_mode not in REWEIGHT_MODES:
            raise ValueError(f"Unsupported tempflow.noise_reweight_mode={noise_reweight_mode!r}; valid: {list(REWEIGHT_MODES)}.")
        # VIPO's pixel ratio is [B,T,H,W]; the TempFlow per-step weight is a
        # per-sample scalar that won't broadcast cleanly against a dense 4-D
        # ratio, so forbid the combination loudly rather than silently
        # mis-weighting.
        if pixel_enable and noise_reweight_mode != "none":
            raise NotImplementedError("tempflow.noise_reweight_mode is not supported together with VIPO pixel weighting (pixel ratio is [B,T,H,W]; the per-row weight is a per-sample scalar). Disable one of them.")
        mismatch_cfg = self._mismatch_diagnostics_cfg()
        mismatch_enable = self._mismatch_diagnostics_enabled()
        mismatch_by_timestep = bool(mismatch_cfg.get("per_timestep", True))
        mismatch_max_timestep_logs = int(mismatch_cfg.get("max_timestep_logs", 16))
        mismatch_eps = float(mismatch_cfg.get("eps", 1e-8))

        flow_cfg = self.config.get("flow_grpo", {})
        shuffle_timesteps = flow_cfg.get("shuffle_timesteps", True)
        if "timestep_indices" in data.batch and shuffle_timesteps:
            shuffle_timesteps = False

        permute_keys = ["timesteps", "latents", "next_latents", "log_probs"]
        if ratio_norm:
            permute_keys.append("prev_sample_mean")
        if "timestep_indices" in data.batch:
            permute_keys.append("timestep_indices")

        select_keys = [
            "timesteps",
            "latents",
            "next_latents",
            "log_probs",
            "contexts",
            "sigma_schedule",
            "advantages",
            "context_orig_lengths",
            "null_context",
        ]
        if ratio_norm:
            select_keys.append("prev_sample_mean")
        if "timestep_indices" in data.batch:
            select_keys.append("timestep_indices")
        non_tensor_select_keys = ["caption"]

        move_keys = ["latents", "next_latents", "timesteps", "log_probs", "advantages", "sigma_schedule"]
        if ratio_norm:
            move_keys.append("prev_sample_mean")
        if "timestep_indices" in data.batch:
            move_keys.append("timestep_indices")

        # Shared scaffolding (perms collective, train_timesteps, chunk, device, ga).
        plan = self._denoise_setup(
            data,
            permute_keys=permute_keys,
            select_keys=select_keys,
            non_tensor_select_keys=non_tensor_select_keys,
            shuffle=shuffle_timesteps,
        )
        train_timesteps = plan.train_timesteps
        dataloader = plan.dataloader
        device = plan.device
        perms = plan.perms

        metrics = {}
        # zero_grad must live OUTSIDE the mini-batch loop: optimizer.step() runs
        # every `gradient_accumulation` mini-batches, and a per-mini-batch
        # zero_grad wipes the previously accumulated gradients — each step would
        # then use only the LAST mini-batch's gradient while the loss is still
        # divided by gradient_accumulation. The post-step zero_grad re-arms the
        # next accumulation window.
        self.actor_optimizer.zero_grad()
        for batch_idx, mini_batch in enumerate(dataloader):
            td = mini_batch.batch
            ctxs, nctxs = self._prepare_contexts(td, device)
            batch_on_device = mini_batch.pop(batch_keys=move_keys).to(device)

            # TempFlow Eq. 8 weights for THIS micro-batch's denoise schedule:
            # a [T] mean-1 vector (or None ⇒ unweighted/Wan-identical). Derived
            # once here, then indexed per step by the GLOBAL denoise step below.
            noise_weights = resolve_timestep_weights(
                batch_on_device.batch["sigma_schedule"],
                mode=noise_reweight_mode,
                eta=float(self.config.get("eta", 0.25)),
                sigma_form=self._sigma_form,
            )

            for step_idx in range(train_timesteps):
                clip_range = self.config.clip_range
                adv_clip_max = self.config.adv_clip_max

                latent_t, nlatent_t, t_t, sigma_0, seq_len, step_indices = self._denoise_step_inputs(batch_on_device, perms, batch_idx, step_idx)

                need_step_stats = ratio_norm or grad_reweight
                if need_step_stats:
                    (
                        new_log_probs,
                        prev_sample_mean,
                        std_dev_t,
                        sqrt_dt,
                    ) = self.grpo_wan_one_step(
                        latent_t,
                        nlatent_t,
                        ctxs,
                        nctxs,
                        seq_len,
                        self.actor_module,
                        t_t,
                        step_indices,
                        sigma_0,
                        guide_scale=self.config.get("guide_scale", 5.0),
                        return_stats=True,
                    )
                else:
                    new_log_probs = self.grpo_wan_one_step(
                        latent_t,
                        nlatent_t,
                        ctxs,
                        nctxs,
                        seq_len,
                        self.actor_module,
                        t_t,
                        step_indices,
                        sigma_0,
                        guide_scale=self.config.get("guide_scale", 5.0),
                    )

                advantages = torch.clamp(
                    batch_on_device.batch["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )

                # ----- log-prob / advantage shape reconciliation ---------
                # Baseline (pixel_enable=False): both log_probs are scalar
                # per sample -> flatten to (B,).  The matching advantages
                # tensor is also (B,).
                #
                # VIPO (pixel_enable=True): log_probs have dense spatial
                # shape (B, T_lat, H_lat, W_lat).  We keep those dims and
                # broadcast a matching dense advantages tensor - which the
                # trainer already expanded via `_apply_vipo_broadcast` -
                # so the policy loss is computed per spatial location.
                old_log_probs_step = batch_on_device.batch["log_probs"][:, step_idx]
                new_log_probs_step, old_log_probs_step = align_wan_log_probs_for_loss(
                    new_log_probs,
                    old_log_probs_step,
                    advantages,
                    batch_size=batch_on_device.batch.batch_size[0],
                    pixel_enabled=pixel_enable,
                )

                if ratio_norm:
                    prev_sample_mean_step = prev_sample_mean
                    prev_sample_mean_old = batch_on_device.batch["prev_sample_mean"][:, step_idx]

                    # RatioNorm bias + outer scale (paper arxiv 2510.22319 Eq. 8).
                    # Logic-preserving: ``compute_ratio_norm_bias`` performs
                    # the same reduction order, scalar collapse, and eps
                    # placement as the previous inline implementation.
                    ratio_mean_bias, scale, sqrt_dt_scalar = compute_ratio_norm_bias(
                        prev_sample_mean_step,
                        prev_sample_mean_old,
                        sqrt_dt,
                        std_dev_t,
                        eps=ratio_norm_eps,
                    )

                    # VIPO: broadcast the per-sample bias scalar over the
                    # dense spatial dims when the actor is running in
                    # pixel mode.  In scalar mode this is a no-op (shapes
                    # already match).
                    if pixel_enable:
                        ratio_mean_bias_bcast = ratio_mean_bias.view(-1, 1, 1, 1)
                    else:
                        ratio_mean_bias_bcast = ratio_mean_bias

                    # Importance-sampling ratio with RatioNorm adjustment (GRPO_Guard)
                    log_prob_delta = new_log_probs_step - old_log_probs_step
                    effective_log_ratio = (log_prob_delta + ratio_mean_bias_bcast) * scale
                    ratio = torch.exp(effective_log_ratio)
                else:
                    log_prob_delta = new_log_probs_step - old_log_probs_step
                    ratio = torch.exp(log_prob_delta)

                clipped_mask = (ratio < (1.0 - clip_range)) | (ratio > (1.0 + clip_range))
                clip_count = clipped_mask.sum().detach().item()
                clip_fraction = clipped_mask.float().mean().detach().item()

                if mismatch_enable:
                    self._record_mismatch_metrics(
                        metrics,
                        prefix="mismatch/grpo",
                        log_prob_delta=log_prob_delta,
                        ratio=ratio,
                        clipped_mask=clipped_mask,
                        timestep=t_t,
                        eps=mismatch_eps,
                    )
                    if mismatch_by_timestep and step_idx < mismatch_max_timestep_logs:
                        self._record_mismatch_metrics(
                            metrics,
                            prefix=f"mismatch/grpo_timestep/{step_idx}",
                            log_prob_delta=log_prob_delta,
                            ratio=ratio,
                            clipped_mask=clipped_mask,
                            eps=mismatch_eps,
                        )
                # Common diffusion-RL loss boundary (Patch A). Bit-identical to
                # the prior inline ``mean(max(-adv*ratio, -adv*clamp(ratio)))``
                # when timestep_weight=None / beta=0; also asserts
                # advantage.shape == ratio.shape (kills the std-broadcast P0)
                # and guards nan/inf. GRPO-Guard grad_reweight stays a separate
                # post-hoc scale below.
                # TempFlow Eq. 8 weight for THIS step, PER ROW: index noise_weights
                # by EACH row's global denoise step. On the normal path every row
                # shares the schedule's k at this step, so the [B] vector is
                # constant — numerically identical to the prior scalar form; under
                # trajectory branching each branch row is trained at its OWN k, so
                # the weight must be per row. ``ratio`` is per-sample-scalar, so a
                # [B] weight broadcasts cleanly. ``None`` ⇒ unweighted (Wan-identical).
                if noise_weights is None:
                    timestep_weight = None
                elif "timestep_indices" in batch_on_device.batch:
                    row_ks = batch_on_device.batch["timestep_indices"][:, step_idx].long()
                    timestep_weight = noise_weights[row_ks.to(noise_weights.device)]
                else:
                    timestep_weight = noise_weights[step_indices.to(noise_weights.device)]
                policy_loss, _ = grpo_policy_loss(
                    advantage=advantages,
                    ratio=ratio,
                    clip_range=clip_range,
                    timestep_weight=timestep_weight,
                )
                # GRPO-Guard grad-reweight (paper arxiv 2510.22319 §3.2.3,
                # Eq. 12).  ``sqrt_dt_scalar`` is computed inside the
                # ``if ratio_norm`` branch above; recompute it here when
                # grad_reweight is on but ratio_norm is off.
                if grad_reweight:
                    if not ratio_norm:
                        sqrt_dt_scalar = sqrt_dt.mean() if sqrt_dt.ndim > 0 else sqrt_dt
                    dt_scalar = sqrt_dt_scalar**2
                    eta = float(self.config.get("eta", 0.25))
                    # Per-sample t_t reduced to a scalar so δ is uniform
                    # across the mini-batch (paper writes δ as a function
                    # of t at the outer step level, not per-sample).
                    # Wan timesteps are ``int(sigma * 1000)`` (0–1000), but
                    # the dancegrpo δ formula requires SDE time in [0, 1] —
                    # without this the β term collapses to ≈1−η²/2.
                    t_scalar = t_t.float().mean()
                    if t_scalar > 1.0:
                        t_scalar = t_scalar / 1000.0
                    delta = compute_grad_reweight_delta(
                        grad_reweight_form,
                        t_scalar,
                        dt_scalar,
                        eta,
                        eps=grad_reweight_eps,
                    )
                    policy_loss = policy_loss * delta

                loss = policy_loss / (self.gradient_accumulation * train_timesteps)

                data_dict = {
                    "actor/clip_count": clip_count,
                    "actor/clip_fraction": clip_fraction,
                    "actor/loss": loss.detach().item(),
                }
                if ratio_norm:
                    data_dict["actor/ratio_mean_bias"] = ratio_mean_bias.detach().mean().item()
                    data_dict["actor/ratio_scale"] = scale if isinstance(scale, float) else scale.item()
                    data_dict["actor/sqrt_dt"] = sqrt_dt_scalar if isinstance(sqrt_dt_scalar, float) else sqrt_dt_scalar.item()
                if grad_reweight:
                    data_dict["actor/grad_reweight_delta"] = delta if isinstance(delta, float) else delta.item()
                append_to_dict(metrics, data_dict)

                loss.backward()

            self._denoise_optimizer_step(batch_idx, self.gradient_accumulation, metrics)

            del ctxs, nctxs
            for key in move_keys:
                if key in batch_on_device:
                    del batch_on_device[key]
            del batch_on_device, mini_batch
            torch.cuda.empty_cache()

        return metrics

    def grpo_wan_one_step(
        self,
        latents,
        pre_latents,
        context,
        context_null,
        seq_len,
        transformer,
        timesteps,
        i,
        sigma_schedule,
        guide_scale=5.0,
        return_stats: bool = False,
    ):
        """One GRPO training step (with FP16 support).

        Grad is CALLER-controlled, NOT forced here: the GRPO actor update calls
        this in a grad-enabled context so ``log_prob`` carries the policy
        gradient (a ``@torch.no_grad()`` decorator here silently detaches it and
        makes ``loss.backward()`` raise "does not require grad"); any no-grad
        caller wraps the call in ``with torch.no_grad():`` itself. The
        unconditional CFG branch below is detached explicitly regardless."""
        transformer.train()

        # Ensure latents shape (16, 7, 64, 64)
        if latents.dim() == 5:
            latents = latents.squeeze(0)

        if pre_latents.dim() == 5:
            pre_latents = pre_latents.squeeze(0)

        expected_channels = int(self.config.get("latent_channels", LATENT_CHANNELS))
        if latents.shape[0] != expected_channels:
            raise ValueError(f"Expected {expected_channels} latent channels (actor_rollout_ref.latent_channels), got {latents.shape[0]}")

        boundary = getattr(self.config, "wan22_boundary", 0.9)
        sigma = sigma_schedule[i] if sigma_schedule is not None else None
        sample_guide_scale = _select_wan22_guide_scale(guide_scale, timesteps, sigma, boundary)
        autocast_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=autocast_dtype):
            with torch.no_grad():
                pred_uncond = transformer(
                    x=[latents],
                    t=timesteps,
                    context=context_null,
                    seq_len=seq_len,
                )

            if isinstance(pred_uncond, dict) and "rgb" in pred_uncond:
                model_output_uncond = pred_uncond["rgb"][0].detach()
            elif isinstance(pred_uncond, list):
                model_output_uncond = pred_uncond[0].detach()
            else:
                model_output_uncond = pred_uncond.detach()

            pred_cond = transformer(
                x=[latents],
                t=timesteps,
                context=context,
                seq_len=seq_len,
            )

            if isinstance(pred_cond, dict) and "rgb" in pred_cond:
                model_output_cond = pred_cond["rgb"][0]
            elif isinstance(pred_cond, list):
                model_output_cond = pred_cond[0]
            else:
                model_output_cond = pred_cond

            model_output = model_output_uncond + sample_guide_scale * (model_output_cond - model_output_uncond)

        if return_stats:
            _, _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = self.wan_step(
                model_output,
                latents.to(torch.float32),
                self.config.eta,
                sigma_schedule,
                i,
                prev_sample=pre_latents,
                grpo=True,
                return_stats=True,
            )
            return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

        _, _, log_prob = self.wan_step(
            model_output,
            latents.to(torch.float32),
            self.config.eta,
            sigma_schedule,
            i,
            prev_sample=pre_latents,
            grpo=True,
        )

        return log_prob

    def wan_step(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        sigmas: torch.Tensor,
        index: int,
        prev_sample: torch.Tensor,
        grpo: bool,
        return_stats: bool = False,
    ):
        """One Wan Flow-Matching sampling step, recast as an SDE solver for GRPO."""
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        pred_original_sample = latents - sigma * model_output

        # Dispatch to σ_t form's SDE step (DanceGRPO / Flow-GRPO).  The
        # returned ``std_dev_t`` and ``sqrt_dt`` keep the contract the
        # GRPO-Guard RatioNorm path expects:
        #   sigma_t = std_dev_t / sqrt_dt
        # which equals η for DanceGRPO form and η·√(t/(1−t)) for
        # Flow-GRPO form (see ``teleboost/algorithms/grpo/sigma_schedule.py``).
        prev_sample_mean, std_dev_t, sqrt_dt = compute_sde_step(
            form=self._sigma_form,
            model_output=model_output,
            latents=latents,
            eta=eta,
            sigma=sigma,
            sigma_next=sigma_next,
            pred_original_sample=pred_original_sample,
        )

        if grpo and prev_sample is None:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        if grpo:
            log_prob = (-((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))) - torch.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # Exactly the same rank-aware reduction as rollout generation.
            log_prob = reduce_wan_log_density(
                log_prob,
                pixel_enabled=self._pixel_enabled(),
            )
            if return_stats:
                return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
            return prev_sample, pred_original_sample, log_prob

        if return_stats:
            return prev_sample_mean, pred_original_sample, prev_sample_mean, std_dev_t, sqrt_dt
        return prev_sample_mean, pred_original_sample
