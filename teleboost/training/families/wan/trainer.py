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
"""Driver-side trainer policy for Wan GRPO programs."""

from __future__ import annotations

import torch
from verl import DataProto

from teleboost.models.wan.family import LATENT_CHANNELS, VAE_STRIDE
from teleboost.training.core.trainer import RayTeleBoostTrainer

DEFAULT_VAE_STRIDE = list(VAE_STRIDE)
DEFAULT_LATENT_CHANNELS = LATENT_CHANNELS


class WanGenerationMixin:
    """Build latent-space generation inputs for Wan-family rollouts."""

    def _build_gen_batch(self, new_batch: DataProto) -> DataProto:
        non_tensor_keys = ["caption"]
        # Preserve optional dataset/reward metadata used by downstream hooks.
        for optional_key in ("prior", "index_prompt", "id", "data_source"):
            if optional_key in new_batch.non_tensor_batch:
                non_tensor_keys.append(optional_key)

        gen_batch = new_batch.pop(
            batch_keys=["context", "context_orig_lengths", "null_context"],
            non_tensor_batch_keys=non_tensor_keys,
        )
        self._prepare_diffusion_inputs(new_batch, gen_batch)
        gen_batch = gen_batch.repeat(self.config.actor_rollout_ref.rollout.n)

        if not bool(self.config.actor_rollout_ref.get("init_same_noise", True)):
            # Repeating the prompt batch also repeats its initial latent. Draw
            # again when each group member must start from independent noise.
            repeated_latents = gen_batch.batch["input_latents"]
            gen_batch.batch["input_latents"] = torch.randn_like(repeated_latents)
        return gen_batch

    def _prepare_diffusion_inputs(
        self,
        new_batch: DataProto,
        gen_batch: DataProto,
    ) -> None:
        """Attach random latent inputs and a shifted sigma schedule."""
        batch_size = new_batch.batch.batch_size[0]
        num_steps = self.config.actor_rollout_ref.sampling_steps
        num_frames = self.config.actor_rollout_ref.num_frames
        width = self.config.actor_rollout_ref.w
        height = self.config.actor_rollout_ref.h
        vae_stride = self.config.actor_rollout_ref.get(
            "vae_stride",
            DEFAULT_VAE_STRIDE,
        )
        latent_channels = self.config.actor_rollout_ref.get(
            "latent_channels",
            DEFAULT_LATENT_CHANNELS,
        )

        latent_shape = (
            latent_channels,
            (num_frames - 1) // vae_stride[0] + 1,
            height // vae_stride[1],
            width // vae_stride[2],
        )
        input_latents = torch.empty(
            (batch_size, *latent_shape),
            dtype=torch.float32,
        )
        sigma_schedule_batch = torch.empty(
            (batch_size, num_steps + 1),
            dtype=torch.float32,
        )

        for index in range(batch_size):
            sigma_schedule = torch.linspace(1, 0, num_steps + 1)
            sigma_schedule = self._sd3_time_shift(
                self.config.actor_rollout_ref.shift,
                sigma_schedule,
            )
            sigma_schedule_batch[index] = sigma_schedule
            input_latents[index] = torch.randn(latent_shape, dtype=torch.float32)

        gen_batch.batch["input_latents"] = input_latents
        gen_batch.batch["sigma_schedule"] = sigma_schedule_batch

    @staticmethod
    def _sd3_time_shift(shift, value):
        return (shift * value) / (1 + (shift - 1) * value)


class RayWanTrainer(WanGenerationMixin, RayTeleBoostTrainer):
    """Wan generation policy composed with the shared GRPO phase driver."""


__all__ = ["RayWanTrainer", "WanGenerationMixin"]
