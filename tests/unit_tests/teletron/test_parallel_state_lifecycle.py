# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light lifecycle tests for TeleTron's process-global adapters.

The core test profile intentionally does not require Megatron-Core.  Load the
two implementation files against small fake Megatron modules so these tests can
pin monkey-patch identity, return-value, and teardown semantics without a GPU.
The real two-rank path is covered by
``tests/special_distributed/teletron_lifecycle_smoke.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_source(monkeypatch, *, name: str, path: Path, fake_parallel_state=None):
    fake_ps = fake_parallel_state or ModuleType("megatron.core.parallel_state")
    fake_core = ModuleType("megatron.core")
    fake_core.__path__ = []
    fake_core.parallel_state = fake_ps
    fake_megatron = ModuleType("megatron")
    fake_megatron.__path__ = []
    fake_megatron.core = fake_core
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", fake_ps)

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, fake_core, fake_ps


@pytest.fixture
def parallel_state_module(monkeypatch):
    module, _, _ = _load_source(
        monkeypatch,
        name="_teleboost_parallel_state_lifecycle_test",
        path=_REPO_ROOT / "teleboost/engines/teletron/parallel_state.py",
    )
    yield module
    module.restore_distributed_op_patches()


def _fake_collectives(monkeypatch):
    calls = {}
    async_work = object()

    def barrier(group=None, async_op=False, device_ids=None):
        calls["barrier"] = (group, async_op, device_ids)
        return async_work if async_op else None

    def get_rank(group=None):
        calls["get_rank"] = group
        return 0

    def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        calls["all_reduce"] = (tensor, op, group, async_op)
        return async_work if async_op else None

    def all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
        calls["all_gather_base"] = (output_tensor, input_tensor, group, async_op)
        return async_work if async_op else None

    def get_world_size(group=None):
        calls.setdefault("get_world_size", []).append(group)
        return 2 if group == "model-group" else 4

    def broadcast(tensor, src=None, group=None, async_op=False, group_src=None):
        calls["broadcast"] = (tensor, src, group, async_op, group_src)
        return async_work if async_op else None

    def broadcast_object_list(object_list, src=None, group=None, device=None, group_src=None):
        calls["broadcast_object_list"] = (object_list, src, group, device, group_src)
        return "object-broadcast-result"

    originals = {
        "barrier": barrier,
        "get_rank": get_rank,
        "all_reduce": all_reduce,
        "_all_gather_base": all_gather_base,
        "get_world_size": get_world_size,
        "broadcast": broadcast,
        "broadcast_object_list": broadcast_object_list,
    }
    for name, fn in originals.items():
        monkeypatch.setattr(torch.distributed, name, fn)
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: [2, 3] if group == "model-group" else [0, 1, 2, 3],
    )
    return originals, calls, async_work


def test_collective_wrappers_preserve_async_results_and_restore(
    parallel_state_module,
    monkeypatch,
):
    module = parallel_state_module
    originals, calls, async_work = _fake_collectives(monkeypatch)
    module.get_transformer_model_group = lambda check_initialized=True: "model-group"

    module.apply_distributed_op_patches(models_num=1)
    first_barrier_wrapper = torch.distributed.barrier
    module.apply_distributed_op_patches(models_num=1)
    assert torch.distributed.barrier is first_barrier_wrapper

    assert torch.distributed.barrier(async_op=True, device_ids=[0]) is async_work
    assert calls["barrier"] == ("model-group", True, [0])

    tensor = object()
    assert (
        torch.distributed.broadcast(
            tensor,
            group_src=0,
            async_op=True,
        )
        is async_work
    )
    assert calls["broadcast"] == (tensor, None, "model-group", True, 0)

    module.restore_distributed_op_patches()
    for name in ("barrier", "all_reduce", "_all_gather_base", "get_world_size", "broadcast"):
        assert getattr(torch.distributed, name) is originals[name]

    # A complete teardown permits a clean second installation rather than
    # wrapping the previous generation of wrappers.
    module.apply_distributed_op_patches(models_num=1)
    assert getattr(torch.distributed.barrier, "_teleboost_original") is originals["barrier"]
    module.restore_distributed_op_patches()


def test_multi_model_translation_and_topology_change_guard(
    parallel_state_module,
    monkeypatch,
):
    module = parallel_state_module
    originals, calls, async_work = _fake_collectives(monkeypatch)
    module.get_this_transformer_model_group = lambda check_initialized=True: "model-group"

    module.apply_distributed_op_patches(models_num=2)
    tensor = object()
    assert torch.distributed.broadcast(tensor, src=0, async_op=True) is async_work
    assert calls["broadcast"] == (tensor, 2, "model-group", True, None)
    assert torch.distributed.broadcast_object_list(["x"], src=0) == "object-broadcast-result"
    assert calls["broadcast_object_list"][1:3] == (2, "model-group")

    with pytest.raises(RuntimeError, match="destroy model parallelism"):
        module.apply_distributed_op_patches(models_num=1)

    module.restore_distributed_op_patches()
    assert torch.distributed.get_rank is originals["get_rank"]
    assert torch.distributed.broadcast_object_list is originals["broadcast_object_list"]


def test_destroy_wrapper_restores_collectives_resets_state_and_returns_result(
    parallel_state_module,
    monkeypatch,
):
    module = parallel_state_module
    originals, _, _ = _fake_collectives(monkeypatch)
    module.get_transformer_model_group = lambda check_initialized=True: "model-group"
    module.apply_distributed_op_patches(models_num=1)

    for name in (
        "_TENSOR_CONTEXT_PARALLEL_GROUP",
        "_MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE",
        "_MPU_TENSOR_CONTEXT_PARALLEL_RANK",
        "_TRANSFORMER_MODEL_GROUP",
        "_TRANSFORMER_THIS_MODEL_GROUP",
        "WORLD_GROUP",
        "_DATA_PRODUCER_CONSUMER_GROUP",
    ):
        setattr(module, name, object())
    module._DATA_TRANSMIT_GROUP = [object()]

    calls = []

    def original_destroy(marker=None):
        calls.append(marker)
        return "destroy-result"

    wrapped = module.destroy_model_parallel_wrapper(original_destroy)
    assert callable(wrapped)
    assert wrapped(marker="first") == "destroy-result"
    assert calls == ["first"]
    assert torch.distributed.barrier is originals["barrier"]
    assert module._DISTRIBUTED_OP_WRAPPERS == {}
    assert module._DATA_TRANSMIT_GROUP == []
    for name in (
        "_TENSOR_CONTEXT_PARALLEL_GROUP",
        "_MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE",
        "_MPU_TENSOR_CONTEXT_PARALLEL_RANK",
        "_TRANSFORMER_MODEL_GROUP",
        "_TRANSFORMER_THIS_MODEL_GROUP",
        "WORLD_GROUP",
        "_DATA_PRODUCER_CONSUMER_GROUP",
    ):
        assert getattr(module, name) is None


def test_destroy_wrapper_restores_state_when_native_destroy_raises(
    parallel_state_module,
    monkeypatch,
):
    module = parallel_state_module
    originals, _, _ = _fake_collectives(monkeypatch)
    module.get_transformer_model_group = lambda check_initialized=True: "model-group"
    module.apply_distributed_op_patches(models_num=1)
    module._TRANSFORMER_MODEL_GROUP = object()

    def failing_destroy():
        raise RuntimeError("native teardown failed")

    wrapped = module.destroy_model_parallel_wrapper(failing_destroy)
    with pytest.raises(RuntimeError, match="native teardown failed"):
        wrapped()

    assert torch.distributed.barrier is originals["barrier"]
    assert module._DISTRIBUTED_OP_WRAPPERS == {}
    assert module._TRANSFORMER_MODEL_GROUP is None


def test_megatron_adaptor_install_destroy_reinstall_does_not_stack(monkeypatch):
    fake_ps = ModuleType("megatron.core.parallel_state")
    calls = {"initialize": 0, "destroy": 0, "wrap_initialize": 0, "wrap_destroy": 0}

    def original_initialize():
        calls["initialize"] += 1
        return "initialized"

    def original_destroy():
        calls["destroy"] += 1
        return "destroyed"

    fake_ps.initialize_model_parallel = original_initialize
    fake_ps.destroy_model_parallel = original_destroy

    fake_impl = ModuleType("teleboost.engines.teletron.parallel_state")

    def initialize_decorator(fn):
        calls["wrap_initialize"] += 1

        @wraps(fn)
        def wrapped():
            return fn()

        wrapped._teleboost_initialize_wrapper = True
        wrapped._teleboost_original = fn
        return wrapped

    def destroy_decorator(fn):
        calls["wrap_destroy"] += 1

        @wraps(fn)
        def wrapped():
            return fn()

        wrapped._teleboost_destroy_wrapper = True
        wrapped._teleboost_original = fn
        return wrapped

    fake_impl.initialize_model_parallel_decorators = initialize_decorator
    fake_impl.destroy_model_parallel_wrapper = destroy_decorator
    for name in (
        "get_tensor_and_context_parallel_src_rank",
        "get_tensor_context_parallel_group",
        "get_tensor_context_parallel_rank",
        "get_tensor_context_parallel_src_rank",
        "get_tensor_context_parallel_world_size",
    ):
        setattr(fake_impl, name, lambda: None)

    fake_teleboost = ModuleType("teleboost")
    fake_teleboost.__path__ = []
    fake_engines = ModuleType("teleboost.engines")
    fake_engines.__path__ = []
    fake_teletron = ModuleType("teleboost.engines.teletron")
    fake_teletron.__path__ = []
    fake_teleboost.engines = fake_engines
    fake_engines.teletron = fake_teletron
    fake_teletron.parallel_state = fake_impl
    monkeypatch.setitem(sys.modules, "teleboost", fake_teleboost)
    monkeypatch.setitem(sys.modules, "teleboost.engines", fake_engines)
    monkeypatch.setitem(sys.modules, "teleboost.engines.teletron", fake_teletron)
    monkeypatch.setitem(sys.modules, "teleboost.engines.teletron.parallel_state", fake_impl)

    adaptor, fake_core, _ = _load_source(
        monkeypatch,
        name="_teleboost_megatron_adaptor_lifecycle_test",
        path=_REPO_ROOT / "teleboost/engines/teletron/megatron_adaptor.py",
        fake_parallel_state=fake_ps,
    )

    adaptor.install()
    first_initialize = fake_ps.initialize_model_parallel
    first_destroy = fake_ps.destroy_model_parallel
    assert fake_core.mpu is fake_ps
    assert first_initialize() == "initialized"
    assert first_destroy() == "destroyed"

    # install is valid both immediately and after a complete initialize/destroy
    # generation; neither call may wrap the wrappers again.
    adaptor.install()
    assert fake_ps.initialize_model_parallel is first_initialize
    assert fake_ps.destroy_model_parallel is first_destroy
    assert first_initialize() == "initialized"
    assert first_destroy() == "destroyed"
    assert calls == {
        "initialize": 2,
        "destroy": 2,
        "wrap_initialize": 1,
        "wrap_destroy": 1,
    }


def test_non_default_mcore_options_fail_before_group_creation(parallel_state_module):
    with pytest.raises(NotImplementedError, match="hybrid_context_parallel=True"):
        parallel_state_module._validate_mcore_initialize_options(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_comm_backend=None,
            hierarchical_context_parallel_sizes=None,
            hybrid_context_parallel=True,
            num_distributed_optimizer_instances=1,
            expert_tensor_parallel_size=None,
            order="tp-cp-ep-dp-pp",
            get_embedding_ranks=None,
            get_position_embedding_ranks=None,
            create_gloo_process_groups=True,
            high_priority_stream_groups=None,
            sharp_enabled_group=None,
            create_all_gather_group=False,
        )
