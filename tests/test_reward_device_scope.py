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
"""Batch-scope device placement for reward scoring nets.

``BaseRewardModel._models_on_device`` replaces the per-sample
``finally: model.to("cpu")`` round-trip that every reward model used to do
inside the scoring loop (a multi-GB PCIe transfer per sample). Contract:
modules from ``_reward_modules()`` are placed once on entry, restored to CPU
on exit — including on exceptions — and ``None`` entries are skipped.
"""

from types import SimpleNamespace

import pytest
import torch

from teleboost.reward.contract import BaseRewardModel


class _Tracking(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moves = []

    def to(self, device, *args, **kwargs):
        self.moves.append(str(device))
        return self


def _duck(modules, device="cpu"):
    return SimpleNamespace(
        _reward_modules=lambda: modules,
        get_device=lambda: device,
    )


def test_places_once_per_batch_and_restores():
    a, b = _Tracking(), _Tracking()
    duck = _duck([a, b], device="cuda:0")
    with BaseRewardModel._models_on_device(duck):
        assert a.moves == ["cuda:0"] and b.moves == ["cuda:0"]
    assert a.moves == ["cuda:0", "cpu"] and b.moves == ["cuda:0", "cpu"]


def test_restores_on_exception():
    a = _Tracking()
    with pytest.raises(RuntimeError):
        with BaseRewardModel._models_on_device(_duck([a])):
            raise RuntimeError("scoring blew up")
    assert a.moves == ["cpu", "cpu"]


def test_none_modules_skipped_and_default_empty():
    a = _Tracking()
    with BaseRewardModel._models_on_device(_duck([None, a])):
        pass
    assert a.moves == ["cpu", "cpu"]
    # base default declares no modules — the scope is a clean no-op
    with BaseRewardModel._models_on_device(_duck([])):
        pass


def test_every_reward_model_declares_its_modules():
    # The point of the hook: no reward model keeps a per-sample offload.
    import inspect

    from teleboost.reward.providers.external import aesthetic, hps, raft, videoclip, videophy

    for mod, cls_name in [
        (hps, "HPSRewardModel"),
        (aesthetic, "AestheticRewardModel"),
        (raft, "RaftRewardModel"),
        (videoclip, "VideoClipRewardModel"),
        (videophy, "VideophyRewardModel"),
    ]:
        cls = getattr(mod, cls_name, None)
        if cls is None:  # class name drifted — find the BaseRewardModel subclass
            candidates = [c for _, c in inspect.getmembers(mod, inspect.isclass) if issubclass(c, BaseRewardModel) and c is not BaseRewardModel]
            assert candidates, f"{mod.__name__}: no reward model class found"
            cls = candidates[0]
        assert "_reward_modules" in cls.__dict__, f"{cls.__name__} must declare _reward_modules"
        src = inspect.getsource(cls)
        assert 'to("cpu")' not in src, f"{cls.__name__} still offloads per sample"
