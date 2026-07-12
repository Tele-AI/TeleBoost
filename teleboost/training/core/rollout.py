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
"""Generation-batch and rollout-adapter support for prompt-only families."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from teleboost.config.access import select as _select
from teleboost.training.core.payload import RolloutRuntimeConfig

if TYPE_CHECKING:
    from verl import DataProto

PROMPT_ONLY_BATCH_KEYS = (
    "caption",
    "prompt",
    "text",
    "raw_prompt",
    "prior",
    "index_prompt",
    "id",
)


def build_prompt_only_gen_batch(new_batch: DataProto) -> DataProto:
    """Copy prompt metadata into a tensor-free generation batch.

    Prompt-native rollout workers consume the non-tensor plane directly. They
    do not need token tensors or preallocated diffusion latents on the driver.
    """

    # Lazy: keep this module importable in verl-less environments (only the
    # prompt-only gen-batch construction needs the runtime stack).
    from tensordict import TensorDict
    from verl import DataProto

    non_tensor_batch = {}
    for key in PROMPT_ONLY_BATCH_KEYS:
        if key not in new_batch.non_tensor_batch:
            continue
        values = list(new_batch.non_tensor_batch[key])
        non_tensor_batch[key] = np.asarray(values, dtype=object)
    first_values = next(iter(non_tensor_batch.values()), ())
    return DataProto(
        batch=TensorDict({}, batch_size=[len(first_values)]),
        non_tensor_batch=non_tensor_batch,
    )


class PromptOnlyGenerationMixin:
    """Use a raw-prompt generation batch instead of token/latent inputs."""

    def _build_gen_batch(self, new_batch: DataProto) -> DataProto:
        return build_prompt_only_gen_batch(new_batch)


def resolve_sampling_steps(config: Any, *, label: str) -> int | None:
    """Resolve the canonical denoising schedule length from verl config."""

    for dead_key in ("num_timesteps", "actor.num_timesteps", "actor_rollout_ref.actor.num_timesteps"):
        dead = _select(config, dead_key, None)
        if dead is not None:
            raise ValueError(f"{dead_key}={dead} is not a knob; use actor_rollout_ref.sampling_steps (points; transitions = points-1). Set sampling_steps={dead} instead.")
    value = _select(config, "sampling_steps", _select(config, "actor_rollout_ref.sampling_steps", None))
    return None if value is None else int(value)


def rollout_runtime_config_from_verl(config: Any, *, label: str = "rollout") -> RolloutRuntimeConfig:
    """Build model-neutral rollout runtime knobs from an actor_rollout_ref config slice."""

    group_size = int(_select(config, "rollout.n", _select(config, "actor_rollout_ref.rollout.n", _select(config, "n", 1))))
    guard_enabled = bool(_select(config, "actor.grpo_guard.enable", _select(config, "actor_rollout_ref.actor.grpo_guard.enable", False)))
    ratio_norm = bool(
        _select(
            config,
            "ratio_norm",
            _select(
                config,
                "actor.grpo_guard.ratio_norm",
                _select(config, "actor_rollout_ref.actor.grpo_guard.ratio_norm", guard_enabled),
            ),
        )
    )
    return RolloutRuntimeConfig(
        group_size=group_size,
        sigma_form=str(_select(config, "sigma_form", _select(config, "actor_rollout_ref.sigma_form", "flow_grpo"))),
        cfg_text_scale=float(_select(config, "cfg_text_scale", _select(config, "actor_rollout_ref.cfg_text_scale", 1.0))),
        cfg_img_scale=float(_select(config, "cfg_img_scale", _select(config, "actor_rollout_ref.cfg_img_scale", 1.0))),
        num_timesteps=resolve_sampling_steps(config, label=label),
        timestep_shift=_select(
            config,
            "timestep_shift",
            _select(config, "shift", _select(config, "actor_rollout_ref.shift", None)),
        ),
        ratio_norm=ratio_norm,
    )


def as_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        for key in ("prompt", "text", "caption"):
            if key in prompt:
                return str(prompt[key])
    return str(prompt)


def extract_prompt_texts(prompts: Any, *, label: str = "rollout") -> list[str]:
    """Extract prompt strings from a verl-like prompt ``DataProto`` or sequence."""

    non_tensor = getattr(prompts, "non_tensor_batch", None)
    if non_tensor:
        for key in ("caption", "prompt", "text", "raw_prompt"):
            if key in non_tensor:
                return [as_prompt_text(x) for x in list(non_tensor[key])]

    batch = getattr(prompts, "batch", None)
    if isinstance(batch, dict):
        for key in ("prompts", "input_ids"):
            if key in batch:
                return [as_prompt_text(x) for x in list(batch[key])]

    if isinstance(prompts, (list, tuple)):
        return [as_prompt_text(x) for x in prompts]

    raise ValueError(f"{label} could not extract prompt text from DataProto")


__all__ = [
    "PROMPT_ONLY_BATCH_KEYS",
    "PromptOnlyGenerationMixin",
    "as_prompt_text",
    "build_prompt_only_gen_batch",
    "extract_prompt_texts",
    "resolve_sampling_steps",
    "rollout_runtime_config_from_verl",
]
