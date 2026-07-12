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
"""VIPO trainer: Wan generation policy + VIPO's driver-phase hook.

Selected by ``teleboost.programs.wan.backend.WanBackendSpec.trainer_cls`` when
``actor_rollout_ref.pixel_weight.enable`` is true. The pure math lives in
``teleboost.algorithms.vipo`` (paper arXiv 2511.18719); the trainer
extension below wires it into the base trainer's seams.
"""

from verl import DataProto

from teleboost.training.families.wan import RayWanTrainer


class VIPOMixin:
    """Trainer mixin that broadcasts scalar advantages with a pixel-weight map.

    The rollout worker is responsible for attaching ``pixel_weight_maps``
    to the batch (only when ``actor_rollout_ref.pixel_weight.enable``).
    Missing or already-dense inputs are contract errors: silently falling back
    would run ordinary GRPO while reporting VIPO as enabled.
    """

    # ---- Base-trainer seam adapter ---------------------------------------

    def _transform_advantages(self, gen_batch_output, gen_batch, metrics):
        if self._is_pixel_weight_enabled():
            gen_batch_output = self._apply_vipo_broadcast(gen_batch_output, metrics)
        return super()._transform_advantages(gen_batch_output, gen_batch, metrics)

    def _is_pixel_weight_enabled(self) -> bool:
        pw_cfg = self.config.actor_rollout_ref.get("pixel_weight", {}) or {}
        return bool(pw_cfg.get("enable", False))

    def _apply_vipo_broadcast(self, gen_batch_output: DataProto, metrics: dict) -> DataProto:
        """Multiply scalar advantages by the rollout-produced ``(B, T, H, W)``
        weight map.

        Contract: the rollout worker stores ``pixel_weight_maps`` with shape
        ``(B, T_lat, H_lat, W_lat)`` in ``gen_batch_output.batch``.  When
        this is missing, fail before actor update rather than silently running
        a different algorithm.
        """
        if "pixel_weight_maps" not in gen_batch_output.batch:
            raise RuntimeError("VIPO is enabled but batch['pixel_weight_maps'] is missing; the rollout did not satisfy the VIPO payload contract.")

        advantages = gen_batch_output.batch["advantages"]
        pixel_maps = gen_batch_output.batch["pixel_weight_maps"].to(dtype=advantages.dtype, device=advantages.device)

        if pixel_maps.ndim != 4:
            raise ValueError(f"pixel_weight_maps must be 4-D (B, T, H, W); got shape {tuple(pixel_maps.shape)}")
        if advantages.ndim != 1:
            raise ValueError(f"VIPO expects one scalar advantage per sample before dense broadcast; got advantages.ndim={advantages.ndim}.")
        if advantages.shape[0] != pixel_maps.shape[0]:
            raise ValueError(f"Batch-size mismatch: advantages[{advantages.shape[0]}] vs pixel_weight_maps[{pixel_maps.shape[0]}]")

        dense = advantages.view(-1, 1, 1, 1) * pixel_maps
        gen_batch_output.batch["advantages"] = dense
        metrics["train/pixel_weight_mean"] = float(pixel_maps.mean().item())
        metrics["train/pixel_weight_std"] = float(pixel_maps.std(unbiased=False).item())
        metrics["train/dense_advantage_mean"] = float(dense.mean().item())
        return gen_batch_output


class RayVIPOTrainer(VIPOMixin, RayWanTrainer):
    """Wan trainer + VIPO pixel-weighted dense advantage broadcast."""
