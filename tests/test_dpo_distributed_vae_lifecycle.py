# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Lightweight multiprocess tests for the DPO distributed-VAE lifecycle.

The workers use Gloo and tiny CPU tensors: these tests exercise the exact
READY/DATA/DONE-or-ERROR/terminal protocol without loading Wan, a VAE, Ray, or
any checkpoint.  A final matched barrier proves no unmatched production
barrier or point-to-point operation remains after each outcome.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import pytest
from tests.unit_tests.test_utils import spawn


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_process_group(rank: int, world_size: int, port: int):
    backend = os.environ.get("TELEBOOST_DPO_VAE_TEST_BACKEND", "gloo")
    if backend not in {"gloo", "nccl"}:
        raise ValueError(f"unsupported lifecycle test backend: {backend!r}")
    if backend == "nccl":
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(f"NCCL lifecycle test needs {world_size} visible GPUs")
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo") if backend == "nccl" else dist.group.WORLD
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    return device, control_group


def _producer_protocol(device="cpu", control_group=None):
    from teleboost.engines.teletron.distributed.distributed_encoder import (
        DistributedVAEProducerProtocol,
    )

    return DistributedVAEProducerProtocol(
        [1],
        device,
        control_group=control_group,
        data_group=dist.group.WORLD,
        timeout_seconds=20,
    )


def _consumer_channel(device="cpu", control_group=None):
    from teleboost.training.dit_batch_loader import DistributedVAEConsumerChannel

    channel = DistributedVAEConsumerChannel(
        0,
        device,
        control_group=control_group,
        data_group=dist.group.WORLD,
    )
    channel.send_ready(
        iteration=7,
        consumed_train_samples=11,
        consumed_valid_samples=13,
    )
    return channel


def _send_one_batch(protocol, device) -> None:
    tensor = torch.arange(6, dtype=torch.bfloat16, device=device)
    protocol.send_data(
        1,
        {
            "paths": ["chosen/latents"],
            "shapes": {"chosen/latents": [2, 3]},
            "dtypes": {"chosen/latents": "torch.bfloat16"},
        },
        tensor,
    )


def _assert_one_batch(channel, device) -> None:
    batch = channel.receive_batch()
    expected = torch.arange(6, dtype=torch.bfloat16, device=device).view(2, 3)
    assert torch.equal(batch["chosen"]["latents"], expected)


def _normal_worker(rank, world_size, result_queue, port):
    from teleboost.engines.teletron.distributed.distributed_encoder import (
        DISTRIBUTED_VAE_DONE,
        DISTRIBUTED_VAE_STOP,
    )

    device, control_group = _init_process_group(rank, world_size, port)
    try:
        if rank == 0:
            protocol = _producer_protocol(device, control_group)
            assert protocol.receive_ready_states() == [(7, 11, 13)]
            _send_one_batch(protocol, device)
            protocol.send_terminal(DISTRIBUTED_VAE_STOP)
            statuses = protocol.wait_for_terminal_statuses()
            assert statuses[1] == (DISTRIBUTED_VAE_DONE, None)
        else:
            channel = _consumer_channel(device, control_group)
            _assert_one_batch(channel, device)
            channel.close()

        dist.barrier()
        result_queue.put((rank, "normal"))
    finally:
        dist.destroy_process_group()


def _producer_error_worker(rank, world_size, result_queue, port):
    from teleboost.engines.teletron.distributed.distributed_encoder import (
        DISTRIBUTED_VAE_CONSUMER_ERROR,
        DISTRIBUTED_VAE_ERROR,
        DistributedVAEProducerError,
    )

    device, control_group = _init_process_group(rank, world_size, port)
    try:
        if rank == 0:
            protocol = _producer_protocol(device, control_group)
            protocol.receive_ready_states()
            protocol.send_terminal(
                DISTRIBUTED_VAE_ERROR,
                error=RuntimeError("producer encode exploded"),
            )
            statuses = protocol.wait_for_terminal_statuses()
            assert statuses[1][0] == DISTRIBUTED_VAE_CONSUMER_ERROR
            assert "producer encode exploded" in (statuses[1][1] or "")
        else:
            channel = _consumer_channel(device, control_group)
            try:
                channel.receive_batch()
            except DistributedVAEProducerError as exc:
                assert "producer encode exploded" in str(exc)
                channel.close(error=exc)
            else:  # pragma: no cover - a protocol regression
                raise AssertionError("producer error was not propagated")

        dist.barrier()
        result_queue.put((rank, "producer_error"))
    finally:
        dist.destroy_process_group()


def _consumer_error_worker(rank, world_size, result_queue, port):
    from teleboost.engines.teletron.distributed.distributed_encoder import (
        DISTRIBUTED_VAE_CONSUMER_ERROR,
        DISTRIBUTED_VAE_ERROR,
        DistributedVAEConsumerError,
    )

    device, control_group = _init_process_group(rank, world_size, port)
    try:
        if rank == 0:
            protocol = _producer_protocol(device, control_group)
            protocol.receive_ready_states()
            _send_one_batch(protocol, device)

            while not protocol.poll_statuses():
                time.sleep(0.01)
            statuses = protocol.poll_statuses()
            assert statuses[1][0] == DISTRIBUTED_VAE_CONSUMER_ERROR
            assert "consumer train exploded" in (statuses[1][1] or "")

            failure = DistributedVAEConsumerError(statuses[1][1] or "")
            protocol.send_terminal(DISTRIBUTED_VAE_ERROR, error=failure)
            protocol.wait_for_terminal_statuses()
        else:
            channel = _consumer_channel(device, control_group)
            _assert_one_batch(channel, device)
            channel.close(error=RuntimeError("consumer train exploded"))

        dist.barrier()
        result_queue.put((rank, "consumer_error"))
    finally:
        dist.destroy_process_group()


def _run_case(worker, label: str) -> None:
    results = spawn(2, worker, _free_port(), timeout_seconds=90)
    assert sorted(results.get() for _ in range(2)) == [
        (0, label),
        (1, label),
    ]


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def exception(self, *_args, **_kwargs):
        return None


class _FakeProducerProtocol:
    def __init__(self):
        self.terminals = []
        self.wait_calls = 0

    def raise_for_status(self):
        return None

    def send_terminal(self, kind, *, error=None):
        self.terminals.append((kind, error))

    def wait_for_terminal_statuses(self):
        self.wait_calls += 1
        return {1: (2, None)}


def _bare_producer(step):
    import threading

    from teleboost.engines.teletron.distributed.distributed_encoder import DistDataProducer

    producer = DistDataProducer.__new__(DistDataProducer)
    producer.logger = _FakeLogger()
    producer.train_iteration = 0
    producer.target_train_iters = 1
    producer.iteration = 0
    producer.profiler = None
    producer.stop_event = threading.Event()
    producer.protocol = _FakeProducerProtocol()
    producer._main_loop_step = step
    producer.train_epilogue = lambda: None
    return producer


def test_producer_run_sends_stop_without_world_barrier():
    from teleboost.engines.teletron.distributed.distributed_encoder import (
        DISTRIBUTED_VAE_STOP,
    )

    holder = {}

    def _step():
        holder["producer"].train_iteration += 1

    producer = _bare_producer(_step)
    holder["producer"] = producer
    producer.run()

    assert producer.protocol.terminals == [(DISTRIBUTED_VAE_STOP, None)]
    assert producer.protocol.wait_calls == 1


def test_producer_run_reports_exception_through_protocol():
    from teleboost.engines.teletron.distributed.distributed_encoder import (
        DISTRIBUTED_VAE_ERROR,
    )

    failure = RuntimeError("encoder exploded")

    def _step():
        raise failure

    producer = _bare_producer(_step)
    with pytest.raises(RuntimeError, match="encoder exploded") as caught:
        producer.run()

    assert caught.value is failure
    assert producer.protocol.terminals == [(DISTRIBUTED_VAE_ERROR, failure)]
    assert producer.protocol.wait_calls == 1


def test_distributed_vae_normal_exit_is_symmetric():
    _run_case(_normal_worker, "normal")


def test_distributed_vae_producer_error_reaches_consumer():
    _run_case(_producer_error_worker, "producer_error")


def test_distributed_vae_consumer_error_reaches_producer():
    _run_case(_consumer_error_worker, "consumer_error")
