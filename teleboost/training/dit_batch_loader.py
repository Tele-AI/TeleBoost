# Copyright (c) 2025 TeleAI-infra Team (TeleTron)
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
"""DiT-side batch loaders for TeleTron DPO training."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist

from teleboost.engines.teletron import (
    get_args,
    print_rank_0,
    set_config,
)
from teleboost.engines.teletron.distributed.distributed_encoder import (
    DISTRIBUTED_VAE_CONSUMER_ERROR,
    DISTRIBUTED_VAE_DATA,
    DISTRIBUTED_VAE_DONE,
    DISTRIBUTED_VAE_ERROR,
    DISTRIBUTED_VAE_PROTOCOL_VERSION,
    DISTRIBUTED_VAE_READY,
    DISTRIBUTED_VAE_STOP,
    DistributedVAELifecycleError,
    DistributedVAEProducerError,
    _DISTRIBUTED_VAE_READY_TAG,
    _DISTRIBUTED_VAE_STATUS_TAG,
    _format_remote_error,
)


_BATCH_ENVELOPE_KEY = "__teleboost_batch_state__"
_BATCH_DATA = "data"
_BATCH_STOP = "stop"
_BATCH_ERROR = "error"


def _mpu():
    from megatron.core import mpu

    return mpu


def unpack_tensors(packed_tensor, intervals, producer_tensors=None):
    features = [packed_tensor[intervals[i - 1] : intervals[i]] for i in range(1, len(intervals))]
    if producer_tensors is not None:
        assert len(producer_tensors) == len(features)
    return features


class BaseBatchLoader(ABC):
    """Batch iterator shared by DiT ranks inside one TCP group."""

    def __init__(self, data_iterator):
        self.data_iterator = data_iterator
        mpu = _mpu()
        from teleboost.engines.teletron.parallel_state import (
            get_tensor_and_context_parallel_src_rank,
        )

        self.rank = mpu.get_tensor_and_context_parallel_rank()
        self.src_rank = get_tensor_and_context_parallel_src_rank()
        self.group = mpu.get_tensor_and_context_parallel_group()
        self.iteration = 0

        if self.rank == 0 and self.data_iterator is None:
            print_rank_0("Warning: data_iterator is None on the batch root rank.")

    def _get_debug_dump_path(self):
        base_dir = getattr(get_args(), "profile_path", None) or "."
        return os.path.join(base_dir, f"dit_batch/batch_debug_rank_{self.rank}.jsonl")

    def _broadcast_tensor(self, tensor):
        if tensor is not None:
            dist.broadcast(tensor.contiguous(), self.src_rank, group=self.group)

    def _broadcast_object(self, obj_list):
        dist.broadcast_object_list(obj_list, self.src_rank, group=self.group)

    @abstractmethod
    def _prepare_batch_on_rank_zero(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        device = torch.cuda.current_device()
        from .utils import allocate_from_meta, broadcast_tensor_tree, build_meta_tree, recv_tensor_tree

        if self.rank == 0:
            try:
                batch = self._prepare_batch_on_rank_zero()
            except BaseException as exc:
                # The remaining TP/CP ranks are already blocked in the object
                # broadcast below.  Always release them with the same failure
                # instead of leaving a model-parallel subgroup hanging.
                self._broadcast_object(
                    [
                        {
                            _BATCH_ENVELOPE_KEY: _BATCH_ERROR,
                            "error": _format_remote_error(exc),
                        }
                    ]
                )
                raise

            if batch is None:
                self._broadcast_object([{_BATCH_ENVELOPE_KEY: _BATCH_STOP}])
                raise StopIteration

            meta_tree = build_meta_tree(batch)
            self._broadcast_object([{_BATCH_ENVELOPE_KEY: _BATCH_DATA, "meta": meta_tree}])
            broadcast_tensor_tree(batch, self._broadcast_tensor)

            self.iteration += 1
            return batch

        meta_list = [None]
        self._broadcast_object(meta_list)
        envelope = meta_list[0]
        if not isinstance(envelope, dict) or _BATCH_ENVELOPE_KEY not in envelope:
            raise DistributedVAELifecycleError(f"invalid DiT batch broadcast envelope: {envelope!r}")
        state = envelope[_BATCH_ENVELOPE_KEY]

        if state == _BATCH_STOP:
            raise StopIteration
        if state == _BATCH_ERROR:
            raise DistributedVAELifecycleError("DiT batch root failed: " + str(envelope.get("error", "unknown error")))
        if state != _BATCH_DATA:
            raise DistributedVAELifecycleError(f"unknown DiT batch broadcast state: {state!r}")
        meta_tree = envelope["meta"]

        batch = allocate_from_meta(meta_tree, device)
        recv_tensor_tree(batch, self._broadcast_tensor)

        self.iteration += 1
        return batch


def _unflatten_tensor_tree(paths, tensors):
    root = {}
    for path, tensor in zip(paths, tensors, strict=False):
        cur = root
        parts = path.split("/")
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = tensor
    return root


class DistVAEConsumerBatchLoader(BaseBatchLoader):
    """DiT loader that receives encoded latents from distributed-VAE ranks."""

    def __init__(self, data_iterator, *, consumer_channel=None):
        super().__init__(data_iterator)
        self.consumer_channel = consumer_channel
        if self.rank == 0 and self.consumer_channel is None:
            raise ValueError("DistVAEConsumerBatchLoader requires a consumer_channel on the tensor/context-parallel root rank")

    def _prepare_batch_on_rank_zero(self):
        return self.consumer_channel.receive_batch()


class DistributedVAEConsumerChannel:
    """Consumer-side protocol with idempotent normal/error shutdown.

    ``close(error=...)`` reports the local outcome first, then drains any
    producer send that was already in flight until STOP/ERROR arrives.  The
    drain is what lets a producer blocked in its current send observe the
    pre-posted reverse ERROR and terminate every peer coherently.
    """

    def __init__(
        self,
        producer_rank: int,
        device: torch.device | str | int,
        *,
        control_group=None,
        data_group=None,
    ) -> None:
        self.producer_rank = int(producer_rank)
        self.device = device
        self.control_group = control_group
        self.data_group = data_group
        self.control_device = torch.device("cpu") if dist.get_backend(control_group) == "gloo" else device
        self._ready_sent = False
        self._status_sent = False
        self._terminal_kind: str | None = None
        self._terminal_error: str | None = None

    def send_ready(
        self,
        *,
        iteration: int,
        consumed_train_samples: int,
        consumed_valid_samples: int,
    ) -> None:
        if self._ready_sent:
            return
        packet = torch.tensor(
            [
                DISTRIBUTED_VAE_READY,
                int(iteration),
                int(consumed_train_samples),
                int(consumed_valid_samples),
            ],
            dtype=torch.int64,
            device=self.control_device,
        )
        dist.send(
            packet,
            dst=self.producer_rank,
            group=self.control_group,
            tag=_DISTRIBUTED_VAE_READY_TAG,
        )
        self._ready_sent = True

    def _send_status(self, error: BaseException | str | None) -> None:
        if self._status_sent:
            return
        kind = DISTRIBUTED_VAE_DONE if error is None else DISTRIBUTED_VAE_CONSUMER_ERROR
        packet = torch.tensor(
            [kind],
            dtype=torch.int64,
            device=self.control_device,
        )
        dist.send(
            packet,
            dst=self.producer_rank,
            group=self.control_group,
            tag=_DISTRIBUTED_VAE_STATUS_TAG,
        )
        if error is not None:
            dist.send_object_list(
                [_format_remote_error(error)],
                dst=self.producer_rank,
                group=self.control_group,
            )
        self._status_sent = True

    @staticmethod
    def _validate_envelope(envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, dict):
            raise DistributedVAELifecycleError(f"distributed-VAE message must be a dict, got {type(envelope).__name__}")
        if envelope.get("protocol_version") != DISTRIBUTED_VAE_PROTOCOL_VERSION:
            raise DistributedVAELifecycleError(f"distributed-VAE protocol version mismatch: producer={envelope.get('protocol_version')!r}, consumer={DISTRIBUTED_VAE_PROTOCOL_VERSION}")
        return envelope

    def _receive_envelope(self) -> dict[str, Any]:
        payload = [None]
        dist.recv_object_list(
            payload,
            src=self.producer_rank,
            group=self.control_group,
        )
        return self._validate_envelope(payload[0])

    @staticmethod
    def _intervals(meta_info: dict[str, Any]) -> tuple[list[str], dict, list[int]]:
        paths = list(meta_info["paths"])
        shapes = meta_info["shapes"]
        intervals = [0]
        for path in paths:
            size = 1
            for dim in shapes[path]:
                size *= int(dim)
            intervals.append(intervals[-1] + size)
        return paths, shapes, intervals

    def _receive_data_tensor(self, meta_info: dict[str, Any]) -> dict[str, Any]:
        paths, shapes, intervals = self._intervals(meta_info)
        recv_tensor = torch.empty(
            (intervals[-1],),
            device=self.device,
            dtype=torch.bfloat16,
        )
        dist.recv(
            recv_tensor,
            self.producer_rank,
            group=self.data_group,
            tag=0,
        )
        flat_tensors = unpack_tensors(recv_tensor, intervals)
        flat_named = [tensor.view(*shapes[path]) for path, tensor in zip(paths, flat_tensors, strict=False)]
        return _unflatten_tensor_tree(paths, flat_named)

    def receive_batch(self) -> dict[str, Any]:
        if self._terminal_kind is not None:
            raise DistributedVAELifecycleError("cannot receive a batch after distributed-VAE terminal message")
        envelope = self._receive_envelope()
        kind = envelope.get("kind")
        if kind == DISTRIBUTED_VAE_DATA:
            return self._receive_data_tensor(envelope)
        self._record_terminal(envelope)
        if kind == DISTRIBUTED_VAE_ERROR:
            raise DistributedVAEProducerError(self._terminal_error or "distributed-VAE producer failed")
        raise DistributedVAELifecycleError("distributed-VAE producer stopped before the consumer requested its next batch")

    def _record_terminal(self, envelope: dict[str, Any]) -> None:
        kind = envelope.get("kind")
        if kind not in {DISTRIBUTED_VAE_STOP, DISTRIBUTED_VAE_ERROR}:
            raise DistributedVAELifecycleError(f"unknown distributed-VAE producer message kind: {kind!r}")
        self._terminal_kind = str(kind)
        self._terminal_error = str(envelope.get("error", "distributed-VAE producer failed")) if kind == DISTRIBUTED_VAE_ERROR else None

    def close(self, *, error: BaseException | str | None = None) -> None:
        """Report local completion/failure and drain through producer terminal."""

        if not self._ready_sent:
            raise DistributedVAELifecycleError("cannot close distributed-VAE consumer before READY")
        self._send_status(error)
        while self._terminal_kind is None:
            envelope = self._receive_envelope()
            kind = envelope.get("kind")
            if kind == DISTRIBUTED_VAE_DATA:
                # A batch may already be in flight when the consumer's reverse
                # DONE/ERROR reaches the producer. Drain exactly that payload;
                # the producer checks status before its next send.
                self._receive_data_tensor(envelope)
                continue
            self._record_terminal(envelope)

        if error is None and self._terminal_kind == DISTRIBUTED_VAE_ERROR:
            raise DistributedVAEProducerError(self._terminal_error or "distributed-VAE producer failed")


def create_dit_batch_loader(args, data_iterator, *, consumer_channel=None):
    model_name_lower = set_config().model_config.dit.type.lower()
    is_distributed_vae = args.distributed_vae

    if "teletron" in model_name_lower:
        if is_distributed_vae:
            print_rank_0("Info: Creating DistVAEConsumerBatchLoader.")
            return DistVAEConsumerBatchLoader(
                data_iterator,
                consumer_channel=consumer_channel,
            )
        raise NotImplementedError("A non-distributed VAE loader for TeleTron DiT is not implemented.")
    raise ValueError(f"Unknown model type '{model_name_lower}' for batch loader creation.")


__all__ = [
    "DistributedVAEConsumerChannel",
    "DistVAEConsumerBatchLoader",
    "create_dit_batch_loader",
]
