# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the exact dependency-light joint reward protocol."""

from __future__ import annotations

import torch

from teleboost.reward.execution import collectives


def _rank_chunks():
    return [
        torch.tensor([1.0, 3.0]),
        torch.tensor([10.0]),
        torch.tensor([20.0, 24.0]),
    ]


def _mock_gather(monkeypatch, chunks):
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda: len(chunks))

    def fake_all_gather(outputs, value):
        if value.dtype == torch.int64:
            for output, chunk in zip(outputs, chunks, strict=True):
                output.fill_(len(chunk))
            return
        for output, chunk in zip(outputs, chunks, strict=True):
            output.zero_()
            output[: len(chunk)].copy_(chunk)

    monkeypatch.setattr(collectives.dist, "all_gather", fake_all_gather)


def test_uneven_rank_shards_preserve_order_then_normalize_once(monkeypatch):
    chunks = _rank_chunks()
    _mock_gather(monkeypatch, chunks)

    raw = collectives.allgather_variable_batch(
        chunks[1],
        collective_device=torch.device("cpu"),
        expected_size=5,
    )
    normalized = collectives.normalize_gathered_rewards(raw, enabled=True)

    assert raw.tolist() == [1.0, 3.0, 10.0, 20.0, 24.0]
    assert torch.allclose(normalized, (raw - raw.mean()) / raw.std())
    assert collectives.normalize_gathered_rewards(raw, enabled=False) is raw


def test_gather_rejects_missing_samples_before_payload_collective(monkeypatch):
    calls = 0
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda: 2)

    def gather_lengths_only(outputs, value):
        nonlocal calls
        calls += 1
        assert value.dtype == torch.int64
        outputs[0].fill_(2)
        outputs[1].fill_(1)

    monkeypatch.setattr(collectives.dist, "all_gather", gather_lengths_only)
    try:
        collectives.allgather_variable_batch(
            torch.tensor([2.0]),
            collective_device=torch.device("cpu"),
            expected_size=4,
        )
    except RuntimeError as exc:
        assert "expected full batch size 4" in str(exc)
    else:
        raise AssertionError("missing gathered sample count must fail")
    assert calls == 1


def test_peer_failure_is_synchronized_before_reward_gather(monkeypatch):
    reduced = []
    monkeypatch.setattr(collectives.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda: 3)

    def fake_all_reduce(value, op):
        del op
        reduced.append(value.clone())
        value.fill_(2)

    monkeypatch.setattr(collectives.dist, "all_reduce", fake_all_reduce)
    count = collectives.synchronized_failure_count(
        local_failed=False,
        device=torch.device("cpu"),
    )
    assert count == 2
    assert reduced[0].item() == 0


def test_zscore_degenerate_batches_are_finite():
    singleton = torch.tensor([7.0])
    constant = torch.tensor([3.0, 3.0, 3.0])
    assert collectives.zscore_normalize(singleton) is singleton
    assert torch.equal(collectives.zscore_normalize(constant), torch.zeros_like(constant))
