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
"""Wan video-diffusion model config — verl-native ``HFModelConfig``
subclass that bypasses HF AutoConfig loading.

Why this exists
---------------
``verl.workers.config.model.HFModelConfig.__post_init__`` always runs
``AutoConfig.from_pretrained`` against ``self.path`` to populate
``self.hf_config``. Wan is not an HF-registered architecture — neither
``diffusers``-style nor ``transformers``-style — so this call fails
unconditionally on a real Wan checkpoint directory.

Earlier phases worked around this by pointing ``model.path`` at a stub
HF-config directory (``/tmp/stub_hf/`` with a fake ``Qwen2`` config)
purely to satisfy ``HFModelConfig.__post_init__``. That stub leaks
into the pod's filesystem and lies about the model architecture
throughout the verl init chain.

Verl's design intends downstream subclasses to extend ``HFModelConfig``
via dataclass inheritance — that's why all the post-init-mutated
fields are listed in ``HFModelConfig._mutable_fields``. We override
``__post_init__`` to skip the AutoConfig / tokenizer / processor
loads while keeping the field-population semantics downstream verl
code expects (``self.local_path``, ``self.architectures``,
``self.hf_config``, ``self.share_embeddings_and_output_weights``).

Pair this with ``teleboost.patches.wan_weight_saver`` which registers
a no-op weight saver for the ``WanForWorldModel`` arch in verl's
``weight_loader_registry`` — verl's ``MegatronCheckpointManager``
looks one up at init time, and DPO never actually triggers it (no
rollout sync, no HF export path).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from verl.workers.config.model import HFModelConfig

# Single source of truth for the synthetic architecture name. Used by
# both ``WanModelConfig.__post_init__`` (default ``architectures`` list)
# and ``teleboost.patches.wan_weight_saver`` (registry key).
from teleboost.models import WAN_ARCH


@dataclass
class WanModelConfig(HFModelConfig):
    """Wan/Teletron video-diffusion model config.

    Differences from ``HFModelConfig``:

    * ``__post_init__`` does NOT call ``AutoConfig.from_pretrained``.
      Wan is not HF-registered; the call would always fail. The real
      transformer config comes from ``set_config()['model_config']['dit']``
      via ``MegatronEngineWanVideo._build_tf_config``.
    * ``hf_config`` is a ``SimpleNamespace`` carrying just the fields
      verl downstream code reads on this object: ``architectures``,
      ``model_type``, ``tie_word_embeddings``.
    * ``tokenizer`` / ``processor`` / ``generation_config`` are left
      ``None`` (Wan has no tokenizer). ``load_tokenizer`` defaults
      stay irrelevant.
    * ``share_embeddings_and_output_weights = False`` — Wan has
      separate ``patch_emb`` / ``head`` layers, no weight tie.

    Identical to ``HFModelConfig`` everywhere else (LoRA fields,
    activation offload, override config, etc.).
    """

    # Default the architecture so yaml callers can omit ``architectures``.
    # Verl's MegatronCheckpointManager calls ``get_weight_saver(arch)``;
    # the patches module registers a no-op for ``WAN_ARCH``.
    architectures: Optional[list[str]] = None

    def __post_init__(self):
        # Mirror HFModelConfig.__post_init__ external-lib import; harmless
        # if external_lib is None (default).
        from verl.workers.config.model import import_external_libs

        import_external_libs(self.external_lib)

        # ``path`` is an opaque label for Wan — the real Wan weights
        # come from the pre-converted teletron checkpoint at
        # ``teletron_args.load``, loaded by megatron's
        # ``load_checkpoint`` inside ``_build_megatron_module``. We
        # still surface ``local_path`` / ``local_hf_config_path`` as
        # mirrors of ``path`` so any downstream code that reads these
        # for metadata logging doesn't NPE.
        if self.local_path is None:
            self.local_path = self.path
        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.local_hf_config_path is None:
            self.local_hf_config_path = self.path

        # No tokenizer / processor / generation_config — Wan video
        # diffusion has none. (yaml should set ``load_tokenizer: false``
        # but we ignore it defensively here.)
        self.tokenizer = None
        self.processor = None
        self.generation_config = None
        self.local_tokenizer_path = None

        # Default architectures to Wan; yaml may override.
        if self.architectures is None:
            self.architectures = [WAN_ARCH]

        # Stub ``hf_config`` carrying just the attributes downstream
        # verl code reads on it. ``MegatronEngineWanVideo`` overrides
        # ``_build_tf_config`` / ``_build_megatron_module`` so this is
        # never used for the real model build; it's only here to keep
        # any non-overridden verl call site (e.g. ``HFModelConfig
        # .get_processor``, ``share_embeddings_and_output_weights``
        # propagation) happy.
        self.hf_config = SimpleNamespace(
            architectures=list(self.architectures),
            model_type="wan_video_diffusion",
            tie_word_embeddings=False,
        )

        # Wan has no tied input/output embeddings.
        self.share_embeddings_and_output_weights = False
