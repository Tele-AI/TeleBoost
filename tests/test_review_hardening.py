# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from teleboost.algorithms.vipo import _normalize_tensor_or_ones
from teleboost.datasets.samplers.default_sampler import DefaultSampler
from teleboost.datasets.preprocessing.transforms.formatting import PackInputs
from teleboost.models import offline_clip


def test_vipo_constant_map_keeps_scalar_grpo_weight():
    fallback = torch.ones(3, 4, 5)
    normalized = _normalize_tensor_or_ones(fallback)
    assert torch.equal(normalized, fallback)


def test_vipo_nonconstant_map_still_normalizes_to_unit_interval():
    normalized = _normalize_tensor_or_ones(torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(normalized, torch.tensor([0.0, 0.5, 1.0]))


def test_pack_inputs_transforms_string_conditioning_keys():
    image = torch.arange(8 * 16, dtype=torch.float32).reshape(1, 8, 16).repeat(3, 1, 1)
    conditioning = image.clone()
    mask = torch.ones(1, 8, 16)
    pack = PackInputs(
        image_keys=["video"],
        embedding_keys=["conditioning", "ref_mask"],
        deterministic=True,
    )

    packed = pack(
        {
            "video": image,
            "conditioning": conditioning,
            "ref_mask": mask,
            "video_height": 8,
            "video_width": 16,
            "video_info": (8, 8),
            "struct_prompt": "s",
            "short_prompt": "p",
            "dense_prompt": "d",
            "frame_interval": 1,
        }
    )

    assert packed["video"].shape == (3, 8, 8)
    assert packed["conditioning"].shape == (3, 8, 8)
    assert torch.equal(packed["conditioning"], conditioning[:, :, 4:12])
    assert packed["ref_mask"].shape == (1, 1, 1)


def test_default_sampler_rejects_empty_drop_last_epoch():
    with pytest.raises(ValueError, match="would produce no samples"):
        DefaultSampler(
            [0],
            consumed_samples=0,
            micro_batch_size=2,
            data_parallel_rank=0,
            data_parallel_size=2,
            global_batch_size=4,
            drop_last=True,
        )


def test_default_sampler_can_pad_small_nonempty_dataset():
    sampler = DefaultSampler(
        [0],
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=3,
        data_parallel_size=4,
        global_batch_size=8,
        drop_last=False,
        shuffle=False,
        infinite=False,
    )
    assert list(sampler) == [[0, 0]]


def test_offline_clip_incomplete_visual_checkpoint_fails_fast(monkeypatch):
    class TinyVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 1024, 14, stride=14, bias=False)
            self.required_weight = torch.nn.Parameter(torch.zeros(1))

    class FakeJitModel:
        @staticmethod
        def state_dict():
            return {"visual.conv1.weight": torch.zeros(1024, 3, 14, 14)}

    monkeypatch.setattr(offline_clip, "PreciseViTL14", TinyVisual)
    monkeypatch.setattr(torch.jit, "load", lambda *_args, **_kwargs: FakeJitModel())

    with pytest.raises(RuntimeError, match="incomplete or incompatible"):
        offline_clip.load_vit_l14_weights_from_jit("incomplete.pt")


def test_reward_registry_declares_builtins_without_importing_implementations(monkeypatch):
    """Exercise registry.py without importing BaseRewardModel's verl stack."""
    root = Path(__file__).resolve().parents[1]
    package_name = "_teleboost_lazy_reward_test"
    package = types.ModuleType(package_name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package_name, package)

    path = root / "teleboost/reward/registry.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    registry = module.RewardRegistry
    assert "random" in registry.list_available()
    assert registry.is_registered("videoclip")
    assert registry._registry == {}

    class DummyReward:
        pass

    def fake_import(name):
        assert name == "teleboost.reward.providers.debug.random"
        registry.register("random")(DummyReward)
        return object()

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    assert registry.get("random") is DummyReward


def test_reward_package_init_has_no_eager_model_imports():
    root = Path(__file__).resolve().parents[1]
    init_path = root / "teleboost/reward/__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    eager_imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    assert eager_imports == []


def test_parallel_wan_cp_grad_hook_has_no_synchronous_debug_output():
    path = Path(__file__).resolve().parents[1] / "teleboost/models/wan/teletron/parallel_wan_teletron_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print" for node in ast.walk(tree))
    assert "BEFORE all_reduce" not in path.read_text(encoding="utf-8")
