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
"""No-op the upstream async-rollout / checkpoint-engine plumbing.

Upstream verl v0.7.1 ``RayPPOTrainer.init_workers`` unconditionally builds
an ``AgentLoopManager`` (vLLM / SGLang / TRTLLM async server fleet) and a
``CheckpointEngineManager`` keyed off rollout replicas. Neither applies to
TeleBoost's diffusion path:

* the rollout lives **inside** the actor worker (hybrid_engine) and is
  driven synchronously from ``fit()`` via
  ``self.actor_rollout_wg.generate_sequences(...)``, so there is no async
  request scheduler;
* there are no separate inference replicas to checkpoint/sleep — the FSDP
  module *is* the rollout.

The recipes is diffusion-only, so we replace both managers with no-ops at
patch-apply time. Other recipes never load ``teleboost.patches``, so they
keep upstream behaviour.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

# Match the patch-layer contract at MODULE import time: verl absent ⇒ keep this
# module import-light (object base; the manager is never applied without verl
# anyway); verl present ⇒ import the exact target UNGUARDED so a drifted symbol
# fails LOUDLY at startup instead of silently degrading the base to ``object``.
import importlib.util

logger = logging.getLogger(__name__)


def _install_ray_state_api_alias() -> None:
    """Bridge verl 0.7.1's deprecated Ray state import to the stable API.

    verl imports ``ray.experimental.state.api`` while the pinned Ray 2.56
    exposes the same public functions from ``ray.util.state`` and warns that
    the experimental path will be removed. Installing the module aliases
    before verl is imported avoids both the warning and a future hard failure;
    it does not modify Ray's implementation.
    """
    import ray.experimental
    import ray.util.state as stable_state

    sys.modules["ray.experimental.state"] = stable_state
    sys.modules["ray.experimental.state.api"] = stable_state
    ray.experimental.state = stable_state


if importlib.util.find_spec("verl") is None:
    _BaseRewardLoopManager = object
else:
    _install_ray_state_api_alias()
    from verl.experimental.reward_loop.reward_loop import (
        RewardLoopManager as _BaseRewardLoopManager,
    )


class _NoOpAgentLoopManager:
    """Stub satisfying ``RayPPOTrainer.init_workers`` async-rollout call.

    Upstream uses the returned object's ``rollout_replicas`` attribute when
    wiring ``CheckpointEngineManager``; an empty list is the correct value
    when no async replicas exist.
    """

    @classmethod
    def create(cls, **_kwargs):
        return SimpleNamespace(rollout_replicas=[])


class _NoOpCheckpointEngineManager:
    def __init__(self, **_kwargs):
        pass

    def sleep_replicas(self):
        pass


async def _teleboost_mark_vllm_teardown_expected(self):
    """Mark vLLM engine-core death as expected during Ray teardown.

    vLLM's MPClient monitor logs any EngineCore process exit as an ERROR unless
    ``resources.engine_dead`` is already set. At the end of a successful
    TeleBoost smoke, Ray tears down the reward-server actors after the trainer
    returns, so that monitor message is cleanup noise. Do not call
    ``AsyncLLM.shutdown()`` here: in this colocated reward path it can enter a
    vLLM/C++ destructor path that segfaults. This method only flips the
    monitor's guard bit.
    """
    engine = getattr(self, "engine", None)
    engine_core = getattr(engine, "engine_core", None)
    resources = getattr(engine_core, "resources", None)
    if resources is None:
        return {"marked": False, "reason": "missing_engine_resources"}

    resources.engine_dead = True
    return {"marked": True}


def _install_vllm_tokenizer_regex_fix() -> None:
    """Make vLLM pass Transformers' tokenizer-regex compatibility flag.

    vLLM 0.14 does not expose arbitrary ``AutoTokenizer.from_pretrained``
    kwargs through ``EngineArgs``. Some otherwise valid Qwen3-VL tokenizer
    artifacts contain the legacy Mistral pre-tokenizer regex; Transformers
    4.57 detects it and requires ``fix_mistral_regex=True``. Patch the two
    loader boundaries inside the vLLM server process so both the engine
    tokenizer and multimodal processor see the flag.
    """
    from transformers.processing_utils import ProcessorMixin
    from vllm.tokenizers.hf import CachedHfTokenizer

    if not getattr(CachedHfTokenizer, "_teleboost_regex_fix", False):
        original_tokenizer_loader = CachedHfTokenizer.from_pretrained.__func__

        @classmethod
        def tokenizer_loader(cls, path_or_repo_id, *args, **kwargs):
            kwargs.setdefault("fix_mistral_regex", True)
            return original_tokenizer_loader(cls, path_or_repo_id, *args, **kwargs)

        CachedHfTokenizer.from_pretrained = tokenizer_loader
        CachedHfTokenizer._teleboost_regex_fix = True

    if not getattr(ProcessorMixin, "_teleboost_regex_fix", False):
        original_processor_loader = ProcessorMixin.from_pretrained.__func__

        @classmethod
        def processor_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
            kwargs.setdefault("fix_mistral_regex", True)
            return original_processor_loader(
                cls,
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )

        ProcessorMixin.from_pretrained = processor_loader
        ProcessorMixin._teleboost_regex_fix = True


def _install_vllm_teardown_marker() -> None:
    try:
        from verl.workers.rollout.vllm_rollout import vllm_async_server as _vllm_server
    except Exception:
        logger.debug("vLLM teardown marker unavailable; leaving cleanup logging unchanged", exc_info=True)
        return

    if not hasattr(_vllm_server, "vLLMHttpServer"):
        return

    server_cls = _vllm_server.vLLMHttpServer
    server_cls.teleboost_mark_teardown_expected = _teleboost_mark_vllm_teardown_expected
    if not getattr(server_cls, "_teleboost_regex_fix", False):
        original_launch_server = server_cls.launch_server

        async def launch_server(self, *args, **kwargs):
            _install_vllm_tokenizer_regex_fix()
            return await original_launch_server(self, *args, **kwargs)

        server_cls.launch_server = launch_server
        server_cls._teleboost_regex_fix = True


class _TeleBoostRewardLoopManager(_BaseRewardLoopManager):
    """RewardLoopManager subclass for TeleBoost diffusion paths.

    Upstream ``RewardLoopManager.__init__`` sets up the ``RewardModelManager``
    (colocated vLLM judge + router address) and THEN spawns ``RewardLoopWorker``
    ray actors via ``_init_reward_loop_workers``. Those workers assume LLM token
    tensors and init tokenizers from the actor model path — invalid for Wan
    diffusion checkpoints.

    So we INHERIT the upstream ``__init__`` (the reward-model-manager/router
    setup stays single-sourced in verl) and override exactly two seams:
      * ``_init_reward_loop_workers`` -> no-op (skip the LLM-token actors),
      * ``compute_rm_score`` -> route video batches through TeleBoost's VLM
        adapter instead of the upstream worker fan-out.
    The wake/sleep lifecycle of ``RewardModelManager`` is reused as-is.

    (verl offers no injection seam for this manager — ``init_workers`` hard
    ``from verl... import RewardLoopManager`` — so the name is still swapped in
    at patch-apply time; that swap now points at this subclass, not a rewrite.)
    """

    def __init__(self, config, rm_resource_pool=None):
        # Importing verl's vLLM server module pulls the complete vLLM tokenizer
        # stack (including native SentencePiece) into every TeleBoost process.
        # Install the hook only when the upstream constructor will actually
        # create a reward server.
        if bool(config.reward.reward_model.enable):
            _install_vllm_teardown_marker()
        super().__init__(config, rm_resource_pool)

    def _init_reward_loop_workers(self):
        # diffusion routes batches through the VLM adapter (compute_rm_score),
        # never through upstream RewardLoopWorker actors.
        self.reward_loop_workers = []

    def compute_rm_score(self, data):
        if self.reward_model_manager is None:
            raise RuntimeError("TeleBoost VLM reward requires reward.reward_model.enable=True")
        if not self.reward_router_address:
            raise RuntimeError("vLLM reward router is not available; check reward.reward_model.adapter and reward.reward_model rollout config")

        from teleboost.reward.adapters.video_vlm import compute_video_vlm_reward as _score

        self.reward_model_manager.wake_up()
        try:
            return _score(self.config, data, reward_router_address=self.reward_router_address)
        finally:
            self.reward_model_manager.sleep()

    def mark_vllm_teardown_expected(self) -> None:
        reward_model_manager = getattr(self, "reward_model_manager", None)
        if reward_model_manager is None:
            return

        mark_refs = []
        for replica in getattr(reward_model_manager, "rollout_replicas", []) or []:
            for server in getattr(replica, "servers", []) or []:
                mark = getattr(server, "teleboost_mark_teardown_expected", None)
                if mark is not None:
                    mark_refs.append(mark.remote())

        if not mark_refs:
            return

        import ray

        ray.get(mark_refs, timeout=10)


def _wrap_reward_loop_manager_with_gate(_rl_mod) -> None:
    """Replace ``RewardLoopManager`` with the teleboost no-spawn variant.

    Real ``RewardModelManager`` still starts on VLM judge paths; only the
    upstream ``RewardLoopWorker`` actors are skipped.
    """
    _rl_mod.RewardLoopManager = _TeleBoostRewardLoopManager
    # ``verl.trainer.ppo.ray_trainer.init_workers`` does
    # ``from verl.experimental.reward_loop import RewardLoopManager``
    # at call time — patch the package attr too so the fresh ``from``
    # picks up our class.
    # Unguarded: verl is known-present here (patches/__init__._verl_available).
    # Failing to re-point the package attr would leave the bypass HALF-applied
    # (the call-time ``from`` picks the upstream class, silently dropping the
    # multimodal adapter) — worse than crashing at startup.
    import verl.experimental.reward_loop as _rl_pkg

    _rl_pkg.RewardLoopManager = _TeleBoostRewardLoopManager


_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    # Unguarded: the patch-layer gate already established verl is installed,
    # so an import failure here is upstream layout drift under the 0.7.1 pin —
    # fail loudly instead of silently skipping the async bypass.
    import verl.experimental.agent_loop as _alm
    import verl.experimental.reward_loop.reward_loop as _rl
    import verl.trainer.ppo.ray_trainer as _rt

    # Assert the exact targets exist before overriding — otherwise a renamed
    # upstream symbol would silently create a shadow attribute and the real
    # (renamed) manager would still run. Loud-on-drift, per the patch contract.
    assert hasattr(_rt, "CheckpointEngineManager"), "verl drift: ray_trainer.CheckpointEngineManager gone"
    assert hasattr(_alm, "AgentLoopManager"), "verl drift: agent_loop.AgentLoopManager gone"
    _rt.CheckpointEngineManager = _NoOpCheckpointEngineManager
    _alm.AgentLoopManager = _NoOpAgentLoopManager
    _wrap_reward_loop_manager_with_gate(_rl)
    _APPLIED = True
