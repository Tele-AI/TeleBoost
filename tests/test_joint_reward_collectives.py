# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for joint reward collection.

The distributed calls are mocked deliberately: these regressions exercise the
padding/order and fail-fast protocols without creating Ray actors, a process
group, NCCL, or a CUDA context.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordict import TensorDict
from verl import DataProto

from teleboost.reward.contract import BaseRewardModel, RewardConfig
from teleboost.reward.execution import worker as worker_module
from teleboost.reward.execution.worker import JointRewardModelWorker


class _MetaReward(BaseRewardModel):
    REWARD_KEY = "meta_rewards"

    def get_device(self) -> torch.device:
        # A meta tensor gives this CPU-only test a non-CPU device whose device
        # identity survives TensorDict construction.
        return torch.device("meta")

    def init_model(self) -> None:
        pass

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        del video_frames, caption
        return 1.0


class _CaptionReward(BaseRewardModel):
    REWARD_KEY = "caption_rewards"

    def init_model(self) -> None:
        pass

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        del video_frames
        return float(caption)


def test_joint_batch_result_stays_on_scoring_device_without_cuda():
    model = _MetaReward(
        RewardConfig(name="meta", normalize=False),
        global_rank=0,
        world_size=1,
    )
    data = DataProto(
        batch=TensorDict(
            {"video_frames": torch.empty((1, 3, 2, 2, 2), device="meta")},
            batch_size=1,
        ),
        non_tensor_batch={"caption": np.asarray(["prompt"], dtype=object)},
    )

    result = model.compute_batch_score_for_joint(data)

    assert result.batch[model.REWARD_KEY].device.type == "meta"


def test_joint_base_returns_raw_uneven_rank_slice_before_global_normalization():
    model = _CaptionReward(
        RewardConfig(name="caption", normalize=True),
        global_rank=0,
        world_size=2,
    )
    data = DataProto(
        batch=TensorDict(
            {"video_frames": torch.empty((5, 3, 2, 2, 2))},
            batch_size=5,
        ),
        non_tensor_batch={
            "caption": np.asarray(["1", "3", "10", "20", "24"], dtype=object),
        },
    )

    result = model.compute_batch_score_for_joint(data)

    # Rank 0 receives ceil(5/2)==3 samples. These must still be raw; local
    # z-score would instead return approximately [-0.76, -0.37, 1.13].
    assert result.batch[model.REWARD_KEY].tolist() == [1.0, 3.0, 10.0]


def test_variable_length_allgather_trims_padding_and_preserves_rank_order(monkeypatch):
    worker = object.__new__(JointRewardModelWorker)
    calls = 0
    rank_chunks = [
        torch.tensor([10.0, 11.0]),
        torch.tensor([20.0]),
        torch.tensor([30.0, 31.0]),
    ]

    monkeypatch.setattr(worker_module.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(worker_module.dist, "get_backend", lambda: "gloo")

    def fake_all_gather(outputs, value):
        nonlocal calls
        calls += 1
        if value.dtype == torch.int64:
            for output, chunk in zip(outputs, rank_chunks, strict=True):
                output.fill_(len(chunk))
            return
        for output, chunk in zip(outputs, rank_chunks, strict=True):
            output.zero_()
            output[: len(chunk)].copy_(chunk)

    monkeypatch.setattr(worker_module.dist, "all_gather", fake_all_gather)

    gathered = worker._allgather_rewards(rank_chunks[1], expected_size=5)

    assert calls == 2
    assert gathered.tolist() == [10.0, 11.0, 20.0, 30.0, 31.0]


def test_uneven_shards_are_normalized_once_after_global_gather(monkeypatch):
    worker = object.__new__(JointRewardModelWorker)
    rank_chunks = [
        torch.tensor([1.0, 3.0]),
        torch.tensor([10.0]),
        torch.tensor([20.0, 24.0]),
    ]

    monkeypatch.setattr(worker_module.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(worker_module.dist, "get_backend", lambda: "gloo")

    def fake_all_gather(outputs, value):
        if value.dtype == torch.int64:
            for output, chunk in zip(outputs, rank_chunks, strict=True):
                output.fill_(len(chunk))
            return
        for output, chunk in zip(outputs, rank_chunks, strict=True):
            output.zero_()
            output[: len(chunk)].copy_(chunk)

    monkeypatch.setattr(worker_module.dist, "all_gather", fake_all_gather)

    raw = worker._allgather_rewards(rank_chunks[1], expected_size=5)
    normalized = worker._normalize_gathered_rewards(raw, enabled=True)
    expected = (raw - raw.mean()) / raw.std()

    assert raw.tolist() == [1.0, 3.0, 10.0, 20.0, 24.0]
    assert torch.allclose(normalized, expected)
    assert worker._normalize_gathered_rewards(raw, enabled=False) is raw


def test_variable_length_allgather_rejects_missing_samples_before_data_gather(monkeypatch):
    worker = object.__new__(JointRewardModelWorker)
    calls = 0

    monkeypatch.setattr(worker_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(worker_module.dist, "get_backend", lambda: "gloo")

    def fake_all_gather(outputs, value):
        nonlocal calls
        calls += 1
        assert value.dtype == torch.int64
        outputs[0].fill_(2)
        outputs[1].fill_(1)

    monkeypatch.setattr(worker_module.dist, "all_gather", fake_all_gather)

    with pytest.raises(RuntimeError, match="expected full batch size 4"):
        worker._allgather_rewards(torch.tensor([2.0]), expected_size=4)

    assert calls == 1


def test_peer_model_failure_is_synchronized_before_reward_gather(monkeypatch):
    worker = object.__new__(JointRewardModelWorker)
    reduced = []

    monkeypatch.setattr(worker_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(worker_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(worker_module.dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(value, op):
        del op
        reduced.append(value.clone())
        # This rank succeeded, but its peer reported a scoring error.
        value.fill_(1)

    monkeypatch.setattr(worker_module.dist, "all_reduce", fake_all_reduce)

    with pytest.raises(
        RuntimeError,
        match="failed on 1/2 rank.*before reward all_gather",
    ):
        worker._raise_if_any_rank_failed(
            "aesthetic",
            local_error=None,
            local_rewards=torch.tensor([0.5]),
        )

    assert len(reduced) == 1
    assert reduced[0].item() == 0
