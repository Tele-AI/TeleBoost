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
"""
TeleBoost Ray trainer: the algorithm-agnostic diffusion-GRPO driver,
orchestrating video generation and reward computation across GPU
workers. Driver-phase algorithms compose program-specific trainer mixins.
"""

import logging
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import marked_timer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - stripped-down unit environments

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, *args, **kwargs):
            del args, kwargs
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or ())

        def update(self, *_args, **_kwargs):
            pass

        def close(self):
            pass


from teleboost.reward.routing import is_video_vlm_reward_config
from teleboost.algorithms.grpo.advantage import per_prompt_zscore_advantage
from teleboost.training.core.payload import (
    drop_batch_tensor,
    video_tensor_to_uint8_frames,
)
from teleboost.training.rewarding.joint_reward_trainer import JointRewardMixin
from teleboost.training.core.loop import (
    epoch_for_training_step,
    should_continue_training,
)

logger = logging.getLogger(__name__)


def _save_video_and_prompt(video_frames: torch.Tensor, rank: int, index: int) -> None:
    """Write a (C, T, H, W) tensor to ./videos/output/video_batch_<ts>_<index>.mp4.

    Pre-X3 lived as `verl.utils.checkpoint.checkpoint_manager.save_video_and_prompt`
    in the in-tree fork; moved here since it's purely a recipe-level validation
    preview helper and has no place in upstream verl.
    """
    from datetime import datetime

    import cv2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assert video_frames.dim() == 4
    C, T, H, W = video_frames.shape
    # Some decoders emit frames in bf16, which numpy cannot hold.
    video_np = video_tensor_to_uint8_frames(video_frames, clamp=False)
    video_filename = f"video_batch_{timestamp}_{index}.mp4"
    video_path = os.path.join("./videos/output", video_filename)
    os.makedirs("videos/output", exist_ok=True)
    out = cv2.VideoWriter(
        video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=video_np.shape[0],
        frameSize=(W, H),
    )
    for t in range(T):
        frame = video_np[t]
        if C == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    out.release()


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """Diffusion-friendly timing metrics.

    Upstream verl 0.4.0 `compute_timing_metrics` requires `batch["responses"]`
    (LM token shape) to derive per-token throughput. Diffusion batches don't
    carry that — they have latents. Pre-X3's in-tree fork commented out the
    per-token block and emitted only raw `timing_s/{name}` entries; mirror
    that here.
    """
    return {f"timing_s/{name}": value for name, value in timing_raw.items()}


def compute_advantage(
    data: DataProto,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Per-prompt group-relative GRPO advantage.

    Paper: GRPO arxiv 2402.03300 §4.1.2 + DanceGRPO arxiv 2505.07818
    Eq. 10.  The "group" is the ``num_repeat`` samples generated for
    the same prompt — z-score is taken **within** that group, not
    across the whole batch (see ``algorithms/grpo_advantage.py``).

    GRPO-only inline body; adding ReMax / GAE back needs the verl-side
    multi-estimator switch reinstated, not just an ``if`` branch.
    """
    rewards = data.batch["rewards"]
    # min_group_std: groups whose reward std is below this floor carry no
    # usable ranking signal — the z-score would amplify float noise to ±O(1)
    # random-sign advantages (the freeze mechanism).  Default 0.0 = off.
    min_group_std = float(config.get("min_group_std", 0.0)) if config is not None else 0.0
    data.batch["advantages"] = per_prompt_zscore_advantage(rewards, num_repeat, min_group_std=min_group_std)
    return data


class RayTeleBoostTrainer(JointRewardMixin, RayPPOTrainer):
    """Driver-side TeleBoost GRPO trainer — the algorithm-agnostic base.

    This class names no model family or algorithm. Family trainers provide
    generation-batch construction; driver-phase algorithms attach through
    recipe-local subclasses and override the extension seams below. A backend
    selects the concrete trainer, failing loudly on unsupported combinations.

    Extension seams:

    * ``_build_gen_batch(batch)`` — required family-owned generation inputs;
      the base implementation fails loudly.
    * ``_pre_rollout_transform(gen_batch)`` — runs before the sharded
      rollout dispatch (e.g. stamping globally-unique prompt ids).
    * ``_transform_rewards(reward_output, source_batch, metrics)`` —
      reward post-processing after each reward path returns.
    * ``_compute_algorithm_advantage(batch)`` — return a batch to
      REPLACE the default advantage computation, or ``None`` to keep it.
    * ``_transform_advantages(batch, gen_batch, metrics)`` — advantage
      post-processing (scaling, dense broadcast, ...).

    ``JointRewardMixin`` stays on the base: ``reward.reward_model.type=joint``
    is multi-reward worker topology, not a paper algorithm — it is
    driven by the reward-model config, orthogonal to the seams above.

    See ``teleboost/algorithms/README.md`` for the algorithm map and
    ``recipes/README.md`` for the layout rationale.
    """

    # ---- Family / algorithm extension seams -------------------------------
    # Family trainers own generation inputs. Algorithm trainers override the
    # remaining seams and are composed in teleboost.programs.<family>.
    # Adapters use cooperative super() so an explicit combination trainer
    # chains them in MRO order.

    def _build_gen_batch(self, new_batch: DataProto) -> DataProto:
        del new_batch
        raise NotImplementedError(f"{type(self).__name__} must implement family-owned generation-batch construction")

    def _pre_rollout_transform(self, gen_batch: DataProto) -> DataProto:
        return gen_batch

    def _transform_rewards(self, reward_output: DataProto, source_batch: DataProto, metrics: dict) -> DataProto:
        return reward_output

    def _compute_algorithm_advantage(self, gen_batch_output: DataProto):
        return None

    def _transform_advantages(self, gen_batch_output: DataProto, gen_batch: DataProto, metrics: dict) -> DataProto:
        return gen_batch_output

    def __init__(self, *args, **kwargs):
        """Driver-side ctor: delegate to upstream's struct ctor, then
        wire the reward lifecycle. Reward dispatch is fit()-internal
        (``_compute_*_reward``); there is no injected reward_fn."""
        super().__init__(*args, **kwargs)
        self._setup_reward_lifecycle()

    def _setup_reward_lifecycle(self) -> None:
        """Configure the reward lifecycle (default path).

        A subclass may override this for a colocated vLLM-router reward.
        """
        # Upstream ``need_reward_model`` checks
        # ``config.reward.reward_model.enable``, which config_schema
        # intentionally turns off for worker-group reward paths (HPS / joint don't
        # want upstream's ``RewardModelManager`` to launch a vLLM
        # replica). But TeleBoost's own ``_compute_single_rm_reward`` /
        # ``_compute_joint_parallel_reward`` need ``self.use_rm=True``
        # to dispatch into ``rm_wg.compute_rm_score``. Override
        # self.use_rm with the teleboost-level ``trainer.use_rm`` flag
        # (defaults True for all diffusion recipes) so the reward path
        # selector inside ``_compute_rewards`` doesn't fall through to
        # the plain function reward path used for smoke runs.
        teleboost_use_rm = self.config.trainer.get("use_rm", None)
        if teleboost_use_rm is not None:
            self.use_rm = bool(teleboost_use_rm)

    def _postprocess_rollout(self, gen_batch_output: DataProto, metrics: dict) -> None:
        """Hydrate bulk video frames from the TransferQueue after rollout."""
        del metrics
        self._read_video_frames(gen_batch_output)

    def init_workers(self):
        """Spawn worker groups + pin a teleboost-side ``rm_wg`` handle.

        Upstream v0.7.1 ``RayPPOTrainer.init_workers`` deleted both
        Role.RewardModel's ``resource_pool_to_cls`` registration AND
        the ``self.rm_wg`` attach — the new path is ``RewardLoopManager``
        / ``RewardModelManager``. TeleBoost's own fit() still
        dispatches HPS / joint / vision rewards through
        ``self.rm_wg.compute_rm_score(...)``. To put that back without
        copying upstream's 165-line init_workers body, we
        monkey-patch two upstream symbols just for the duration of
        super().init_workers():

        1. ``create_colocated_worker_cls`` — injects a
           ``Role.RewardModel`` entry into the first colocated
           class_dict that upstream builds (the actor's pool), so the
           reward worker gets spawned alongside the actor on the same
           GPUs (matches the v0.4 fork's colocated lifecycle).
        2. ``RayWorkerGroup.spawn`` — captures the returned mapping
           so we can grab the freshly-spawned reward worker group and
           pin it to ``self.rm_wg``.

        Both patches are restored in a ``finally`` block. Skip
        entirely when Role.RewardModel isn't registered (VLM-router mode
        uses verl's RewardModelManager through RewardLoopManager).
        """
        import verl.trainer.ppo.ray_trainer as _rt
        from verl.single_controller.ray.base import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        rm_role_registered = Role.RewardModel in self.role_worker_mapping

        if not rm_role_registered:
            super().init_workers()
            return

        rm_key = str(Role.RewardModel)
        rm_cls = _rt.RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.RewardModel],
            config=self.config.reward.reward_model,
        )
        rm_injected = [False]
        captured: dict = {}

        _orig_create = _rt.create_colocated_worker_cls
        _orig_spawn = RayWorkerGroup.spawn

        def _injecting_create(class_dict):
            # First colocated build is the actor's pool — inject RM
            # there so it shares GPUs with the actor (matches
            # ``mapping[Role.RewardModel] = global_pool_id`` set by
            # ``main_teleboost._register_reward_workers``).
            if not rm_injected[0]:
                class_dict[rm_key] = rm_cls
                rm_injected[0] = True
            return _orig_create(class_dict=class_dict)

        def _capturing_spawn(self_wg, prefix_set):
            result = _orig_spawn(self_wg, prefix_set)
            captured.update(result)
            return result

        try:
            _rt.create_colocated_worker_cls = _injecting_create
            RayWorkerGroup.spawn = _capturing_spawn
            super().init_workers()
        finally:
            _rt.create_colocated_worker_cls = _orig_create
            RayWorkerGroup.spawn = _orig_spawn

        if rm_key in captured:
            self.rm_wg = captured[rm_key]
            self.rm_wg.init_model()
            logger.warning("[teleboost.init_workers] rm_wg pinned + init_model() called")
        else:
            raise RuntimeError(f"Role.RewardModel ({rm_key}) was registered in role_worker_mapping but not captured during spawn. Captured keys: {list(captured.keys())}")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)

        train_dataloader_len = len(self.train_dataloader)
        if train_dataloader_len <= 0:
            raise RuntimeError("train_dataloader is empty; cannot run training")

        epoch = epoch_for_training_step(self.global_steps, train_dataloader_len)
        while should_continue_training(self.global_steps, self.total_training_steps):
            # ======== 1. Data ========
            saw_batch = False
            for batch_dict in self.train_dataloader:
                if not should_continue_training(self.global_steps, self.total_training_steps):
                    break

                saw_batch = True
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                gen_batch = self._build_gen_batch(new_batch)

                # Pre-rollout seam (e.g. TempFlow stamps globally-unique
                # prompt ids before the sharded dispatch). Base = no-op.
                gen_batch = self._pre_rollout_transform(gen_batch)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    gen_batch_output = self._rollout_phase(gen_batch, metrics, timing_raw)
                    self._maybe_validate(gen_batch_output, is_last_step, timing_raw)
                    gen_batch_output = self._reward_and_advantage_phase(gen_batch_output, gen_batch, metrics, timing_raw)
                    gen_batch_output = self._update_phase(gen_batch_output, metrics, timing_raw)
                    self._maybe_save_checkpoint(is_last_step, timing_raw)

                metrics.update(compute_timing_metrics(batch=new_batch, timing_raw=timing_raw))
                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                logger.log(data=metrics, step=self.global_steps)
                timing_raw = defaultdict(float)
                progress_bar.update(1)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._mark_reward_vllm_teardown_expected()
                    # Eagerly tear down the StatefulDataLoader's worker
                    # subprocesses before this Ray actor enters its own
                    # cleanup phase. Skipping this lets Ray SIGKILL the
                    # dataloader workers, which the iterator's __del__
                    # then re-raises as
                    # ``RuntimeError: DataLoader worker (pid X) is killed
                    # by signal: Killed`` — cosmetic but pollutes every
                    # smoke log. Setting the attrs to None drops the
                    # last reference; the loader's __del__ runs while
                    # the Ray actor is still alive and shuts workers
                    # down cleanly via SIGTERM.
                    self.train_dataloader = None
                    self.val_dataloader = None
                    return

                self.global_steps += 1
            if not saw_batch:
                continue
            epoch += 1

    def _mark_reward_vllm_teardown_expected(self) -> None:
        reward_loop_manager = getattr(self, "reward_loop_manager", None)
        marker = getattr(reward_loop_manager, "mark_vllm_teardown_expected", None)
        if not callable(marker):
            return
        try:
            marker()
        except Exception:  # pragma: no cover - best-effort cleanup quieting
            logger.warning("Failed to mark reward vLLM teardown as expected", exc_info=True)

    # ---- fit() phases ----
    def _rollout_phase(self, gen_batch: DataProto, metrics: dict, timing_raw) -> DataProto:
        """Generate rollouts, hydrate frames, stamp per-prompt group uids."""
        with marked_timer("gen", timing_raw):
            # gen_batch_output is a DataProto aggregated across all GPUs.
            # See DiffusionActorRolloutRefWorker.generate_sequences.
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            # TransferQueue: when enabled the rollout worker
            # popped ``video_frames`` from the DataProto and
            # shipped only ``video_frame_tq_keys`` to save a
            # full-tensor pickle through the Ray driver.
            # Hydrate the batch up front so every downstream
            # consumer (validation video dump, joint pre-
            # compute, teleboost reward_manager, rm_wg select,
            # video_vlm router) reads ``batch["video_frames"]``
            # transparently.
            self._postprocess_rollout(gen_batch_output, metrics)

        # Per-prompt group id: one UUID per prompt, broadcast rollout.n times so
        # each sample carries its prompt's uid. Unused at present — advantage
        # grouping uses layout reshape (grpo_advantage), not uid scatter — but a
        # valid group id for any future uid-scatter grouping.
        n_resp = int(self.config.actor_rollout_ref.rollout.n)
        total = len(gen_batch_output.batch)
        if total % n_resp != 0:
            raise RuntimeError(f"gen_batch_output length ({total}) is not a multiple of rollout.n ({n_resp}); cannot derive prompt count for uid broadcast.")
        prompt_count = total // n_resp
        prompt_uids = np.array(
            [str(uuid.uuid4()) for _ in range(prompt_count)],
            dtype=object,
        )
        # ``np.repeat`` with axis=0 = interleave: each prompt's
        # n samples share its uid (matches the rollout's
        # ``DataProto.repeat(interleave=True)`` layout).
        gen_batch_output.non_tensor_batch["uid"] = np.repeat(prompt_uids, n_resp, axis=0)
        return gen_batch_output

    def _maybe_validate(self, gen_batch_output: DataProto, is_last_step: bool, timing_raw) -> None:
        """Save validation videos on the test-freq cadence."""
        if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
            with marked_timer("validation", timing_raw):
                self._save_validation_videos(gen_batch_output)

    def _reward_and_advantage_phase(self, gen_batch_output: DataProto, gen_batch: DataProto, metrics: dict, timing_raw) -> DataProto:
        """Reward + advantage as one economic unit: the joint mode precomputes
        both together, so the precomputed flag never crosses a method boundary."""
        # When joint precompute path fires we must skip the
        # default reward + advantage blocks below (they would recompute).
        joint_adv_precomputed = False
        # Joint mode: pre-compute per-reward advantages + joint weights
        # BEFORE the reward timer so the downstream block can skip.
        if self.use_rm and self.config.reward.reward_model.type == "joint":
            gen_batch_output, joint_adv_precomputed = self._precompute_joint_advantages(
                gen_batch_output,
                gen_batch,
                metrics,
            )

        with marked_timer("reward", timing_raw):
            if not joint_adv_precomputed:
                gen_batch_output = self._compute_rewards(gen_batch_output, metrics, gen_batch)
            # Validation/snapshots run before this phase, and every reward path
            # has now consumed the decoded frames. Keep them out of advantage
            # processing and the actor RPC; a default video batch can otherwise
            # add gigabytes of needless driver serialization.
            self._drop_decoded_video_frames(gen_batch_output)

        with marked_timer("adv", timing_raw):
            if not joint_adv_precomputed:
                # An algorithm trainer may REPLACE the default advantage
                # (e.g. TempFlow's per-branch-point advantage). None = keep
                # the standard GRPO path below, byte-for-byte.
                replaced = self._compute_algorithm_advantage(gen_batch_output)
                if replaced is not None:
                    gen_batch_output = replaced
                else:
                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    gen_batch_output = compute_advantage(
                        gen_batch_output,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
            # Advantage post-processing seam (scaling / dense broadcast;
            # MRO order chains adapters in a combination trainer).
            gen_batch_output = self._transform_advantages(gen_batch_output, gen_batch, metrics)
            metrics["train/advantage"] = gen_batch_output.batch["advantages"].mean()
        return gen_batch_output

    def _update_phase(self, gen_batch_output: DataProto, metrics: dict, timing_raw) -> DataProto:
        """Actor update behind the critic-warmup gate."""
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw):
                gen_batch_output = self.actor_rollout_wg.update_actor(gen_batch_output)
            actor_output_metrics = reduce_metrics(gen_batch_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return gen_batch_output

    def _maybe_save_checkpoint(self, is_last_step: bool, timing_raw) -> None:
        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
            with marked_timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

    def _save_validation_videos(self, gen_batch_output: DataProto) -> None:
        """Persist validation videos. Default (video) path; a subclass may
        override this as a no-op for models whose rollouts have no frames."""
        video_frames = gen_batch_output.batch["video_frames"]
        for i in range(video_frames.shape[0]):
            _save_video_and_prompt(video_frames[i], 0, i)

    def _compute_rewards(self, gen_batch_output: DataProto, metrics: dict, source_batch: DataProto = None):
        # compute scores. Diffusion supports both worker-group rewards and
        # in-process reward functions (some backends use the vLLM router only).
        if self.use_rm:
            logger.debug("Computing reward")
            with torch.amp.autocast("cuda"):
                # Joint mode: parallel computation using multiple worker groups
                if self.config.reward.reward_model.type == "joint":
                    reward_output = self._compute_joint_parallel_reward(gen_batch_output, metrics)
                    return self._transform_rewards(reward_output, source_batch, metrics)

                if is_video_vlm_reward_config(self.config) or self.config.reward.reward_model.type == "single":
                    reward_output = self._compute_single_rm_reward(gen_batch_output, metrics)
                    return self._transform_rewards(reward_output, source_batch, metrics)

                raise ValueError(f"Unsupported reward model type: {self.config.reward.reward_model.type}")

        # ``use_rm=false``: no reward worker group exists. Use the
        # upstream-style custom reward function configured in yaml.
        from teleboost.reward.execution.custom import load_custom_reward_fn, normalize_custom_reward_output

        reward_fn = load_custom_reward_fn(self.config)
        reward_tensor = normalize_custom_reward_output(reward_fn(gen_batch_output), gen_batch_output)
        gen_batch_output = gen_batch_output.union(reward_tensor)
        return self._transform_rewards(gen_batch_output, source_batch, metrics)

    def _compute_single_rm_reward(self, gen_batch_output: DataProto, metrics: dict):
        if is_video_vlm_reward_config(self.config):
            # v0.7.1 serves the video VLM via RewardLoopManager's
            # colocated RewardModelManager/vLLMReplica, but skips upstream
            # RewardLoopWorker because that worker assumes LLM token
            # tensors. TeleBoost keeps the same public interface by
            # implementing the multimodal adapter in
            # RewardLoopManager.compute_rm_score().
            self._read_video_frames(gen_batch_output)
            reward_tensor = self.reward_loop_manager.compute_rm_score(gen_batch_output)

        else:  # "single"
            reward_input = gen_batch_output.select(
                batch_keys=["video_frames"],
                non_tensor_batch_keys=["caption"],
            )
            reward_tensor = self.rm_wg.compute_rm_score(reward_input)

        # These are non-tensor metadata keys only. ``video_frames`` lives in
        # DataProto.batch and is dropped centrally after reward diagnostics.
        _keys_to_pop = [k for k in ("caption", "video_ids") if k in gen_batch_output.non_tensor_batch]
        if _keys_to_pop:
            gen_batch_output.pop(non_tensor_batch_keys=_keys_to_pop)

        self._debug_proto_batch("gen_batch_output", gen_batch_output)
        self._debug_proto_batch("reward_tensor", reward_tensor)
        gen_batch_output = gen_batch_output.union(reward_tensor)

        if "rewards" not in gen_batch_output.batch:
            _src_key = next((k for k in gen_batch_output.batch.keys() if k.endswith("_rewards")), None)
            if _src_key is None:
                raise KeyError(f"no '*rewards' key in gen_batch_output.batch (keys={list(gen_batch_output.batch.keys())})")
            gen_batch_output.batch["rewards"] = gen_batch_output.batch[_src_key]
        metrics["train/rewards"] = gen_batch_output.batch["rewards"].mean()
        # Some rollout payloads keep log_probs outside the batch tensor plane.
        if "log_probs" in gen_batch_output.batch.keys():
            metrics["train/log_probs"] = gen_batch_output.batch["log_probs"].mean()
        return gen_batch_output

    @staticmethod
    def _drop_decoded_video_frames(gen_batch_output: DataProto) -> None:
        """Remove decoded videos from the tensor plane after their last reader.

        Rollout diagnostics and validation run before reward. Single, joint,
        video-VLM, and custom rewards all finish before this helper is called.
        Keeping the operation idempotent also covers custom paths that may
        already have removed the key themselves.
        """
        drop_batch_tensor(gen_batch_output, "video_frames")

    def _read_video_frames(self, gen_batch_output: DataProto) -> torch.Tensor:
        """Return (B, C, T, H, W) video frames; hydrate from TransferQueue
        if batch.video_frames is missing.

        Phase 2 contract: when ``actor_rollout_ref.transfer_queue.enable=True``
        the rollout writes frames into TQ and pops ``video_frames`` from
        the DataProto it ships back to the driver. The driver hydrates
        the batch lazily — first reader pulls from TQ, stores the result
        back into ``gen_batch_output.batch`` so subsequent readers (e.g.
        ``select(batch_keys=["video_frames"])`` in single / joint rm_wg
        paths, ``_save_validation_videos``) see it transparently.

        Fallback for every legacy path (TQ disabled, package missing,
        kv_batch_get failure, or producer kept batch.video_frames): plain
        ``batch["video_frames"]``.
        """
        if "video_frames" in gen_batch_output.batch.keys():
            return gen_batch_output.batch["video_frames"]

        # Align with upstream verl: read TQ state from env vars
        # propagated through Ray runtime_env (see
        # ``main_teleboost._init_ray``). The driver also runs as a TQ
        # consumer here, so reading ``os.environ`` works in this
        # process too.
        from teleboost.engines.transport import transfer_queue as _tqr

        tq_keys = gen_batch_output.non_tensor_batch.get("video_frame_tq_keys")
        if _tqr.enabled() and tq_keys is not None and len(tq_keys) > 0:
            try:
                frames = _tqr.get_video_frames(tq_keys)
                # Hydrate the DataProto so single / joint paths' subsequent
                # ``select(batch_keys=["video_frames"])`` calls don't need
                # to know about TQ.
                gen_batch_output.batch["video_frames"] = frames
                # Warning level (not info) so the marker shows up under
                # the default ``WARN`` logging level — surfaces TQ
                # activity in production logs without needing
                # VERL_LOGGING_LEVEL=INFO.
                logger.warning(
                    "[teleboost.tq.consumer] hydrated %d samples from partition=%s",
                    len(tq_keys),
                    _tqr.partition_id(),
                )
                return frames
            except Exception as exc:
                logger.warning(
                    "[teleboost.tq.consumer] read failed (%s); video_frames missing from batch",
                    exc,
                )
        # Final fallback — let the original KeyError surface so the caller
        # sees a clean stack trace instead of silently propagating None.
        return gen_batch_output.batch["video_frames"]

    @staticmethod
    def _debug_proto_batch(name, proto):
        if proto is None:
            logger.debug("%s is None", name)
            return
        batch = getattr(proto, "batch", None)
        if batch is None:
            non_tensor = getattr(proto, "non_tensor_batch", None) or {}
            logger.debug("%s.batch is None; non_tensor_keys=%s", name, list(non_tensor.keys()))
            return
        logger.debug("%s.batch_size=%s", name, batch.batch_size)
