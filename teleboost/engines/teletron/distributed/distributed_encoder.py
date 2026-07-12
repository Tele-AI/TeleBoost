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
from __future__ import annotations

import collections
import logging
import os
import random
import threading
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import torch
import torch.distributed as dist

from teleboost.engines.teletron import get_args, get_timers

if TYPE_CHECKING:
    from teleboost.engines.teletron.parallel_state import CommPair

# --- Constants ---
NUM_ITEMS_PER_CONSUMER = 100000
MAX_QUEUE_PER_CONSUMER_ON_PRODUCER = 2

TRAIN_MODE = "train"
VALID_MODE = "valid"

# Producer -> consumer messages.  DATA is followed by one packed tensor; the
# terminal messages contain no tensor payload.
DISTRIBUTED_VAE_PROTOCOL_VERSION = 1
DISTRIBUTED_VAE_DATA = "data"
DISTRIBUTED_VAE_STOP = "stop"
DISTRIBUTED_VAE_ERROR = "error"

# Consumer -> producer fixed-size control packets.  READY is the initial
# state handshake; DONE/ERROR are sent exactly once when the consumer leaves
# ``real_train_step``.  Tags matter for Gloo tests.  NCCL preserves the same
# point-to-point ordering even on versions where tags are ignored.
DISTRIBUTED_VAE_READY = 1
DISTRIBUTED_VAE_DONE = 2
DISTRIBUTED_VAE_CONSUMER_ERROR = 3
_DISTRIBUTED_VAE_READY_TAG = 29101
_DISTRIBUTED_VAE_STATUS_TAG = 29102


class DistributedVAELifecycleError(RuntimeError):
    """Base error for the distributed-VAE producer/consumer protocol."""


class DistributedVAEProducerError(DistributedVAELifecycleError):
    """Raised on a consumer when its producer reports a fatal error."""


class DistributedVAEConsumerError(DistributedVAELifecycleError):
    """Raised on a producer when a consumer reports a fatal error."""


class _ConsumerRequestedStop(Exception):
    """Internal normal-control-flow signal raised after a consumer DONE."""


def _protocol_timeout_seconds() -> float:
    raw = os.environ.get("TELEBOOST_DPO_VAE_PROTOCOL_TIMEOUT_SECONDS", "300")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"TELEBOOST_DPO_VAE_PROTOCOL_TIMEOUT_SECONDS must be a number; got {raw!r}") from exc
    if timeout <= 0:
        raise ValueError(f"TELEBOOST_DPO_VAE_PROTOCOL_TIMEOUT_SECONDS must be > 0; got {timeout}")
    return timeout


def _format_remote_error(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return str(error)


class DistributedVAEProducerProtocol:
    """Producer-side half of the distributed-VAE lifecycle protocol.

    Every reverse-direction terminal receive is posted immediately after the
    READY handshake.  Consequently a consumer that fails while the producer
    is sending the next batch can report ERROR and drain the in-flight batch;
    the producer observes that status before starting another send and emits
    a terminal ERROR to every peer.  This avoids the classic send-vs-send
    deadlock without destroying the process group from a background thread.
    """

    def __init__(
        self,
        consumer_ranks: list[int],
        device: torch.device | str | int,
        *,
        control_group=None,
        data_group=None,
        timeout_seconds: float | None = None,
    ) -> None:
        if not consumer_ranks:
            raise ValueError("distributed-VAE producer has no consumer ranks")
        if len(set(consumer_ranks)) != len(consumer_ranks):
            raise ValueError(f"distributed-VAE consumer ranks must be unique: {consumer_ranks}")
        self.consumer_ranks = list(consumer_ranks)
        self.device = device
        self.control_group = control_group
        self.data_group = data_group
        self.control_device = torch.device("cpu") if dist.get_backend(control_group) == "gloo" else device
        self.timeout_seconds = _protocol_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
        self._status_tensors: dict[int, torch.Tensor] = {}
        self._status_works: dict[int, Any] = {}
        self._status_waiters: dict[int, threading.Thread] = {}
        self._status_wait_errors: dict[int, BaseException] = {}
        self._statuses: dict[int, tuple[int, str | None]] = {}
        self._terminal_sent = False

    def _wait_for_works(self, works: dict[int, Any], *, what: str) -> None:
        # Gloo's Work.is_completed() does not drive point-to-point progress in
        # every supported torch release. All receives are already posted, so a
        # direct wait is deadlock-free and inherits the process-group timeout.
        for rank, work in works.items():
            try:
                work.wait()
            except BaseException as exc:
                raise DistributedVAELifecycleError(f"failed while waiting for {what} from rank {rank}: {exc}") from exc

    def receive_ready_states(self) -> list[tuple[int, int, int]]:
        """Receive READY plus iteration/consumed-sample state from each peer."""

        tensors = {rank: torch.empty((4,), dtype=torch.int64, device=self.control_device) for rank in self.consumer_ranks}
        works = {
            rank: dist.irecv(
                tensor,
                src=rank,
                group=self.control_group,
                tag=_DISTRIBUTED_VAE_READY_TAG,
            )
            for rank, tensor in tensors.items()
        }
        self._wait_for_works(works, what="distributed-VAE READY")

        states = []
        for rank in self.consumer_ranks:
            packet = tensors[rank].cpu().tolist()
            if packet[0] != DISTRIBUTED_VAE_READY:
                raise DistributedVAELifecycleError(f"consumer rank {rank} sent invalid READY kind {packet[0]}")
            states.append((int(packet[1]), int(packet[2]), int(packet[3])))

        # Post the reverse terminal receives before encoder/dataloader setup.
        # A setup exception can therefore be propagated without a peer hang.
        for rank in self.consumer_ranks:
            tensor = torch.empty((1,), dtype=torch.int64, device=self.control_device)
            self._status_tensors[rank] = tensor
            work = dist.irecv(
                tensor,
                src=rank,
                group=self.control_group,
                tag=_DISTRIBUTED_VAE_STATUS_TAG,
            )
            self._status_works[rank] = work

            def _wait_status(peer_rank=rank, peer_work=work):
                try:
                    peer_work.wait()
                except BaseException as exc:
                    self._status_wait_errors[peer_rank] = exc

            waiter = threading.Thread(
                target=_wait_status,
                daemon=False,
                name=f"DistributedVAEStatus-rank{rank}",
            )
            self._status_waiters[rank] = waiter
            waiter.start()
        return states

    def poll_statuses(self) -> dict[int, tuple[int, str | None]]:
        """Materialize every completed DONE/ERROR receive without blocking."""

        for rank, waiter in list(self._status_waiters.items()):
            if waiter.is_alive():
                continue
            waiter.join()
            error = self._status_wait_errors.pop(rank, None)
            if error is not None:
                raise DistributedVAELifecycleError(f"consumer status receive from rank {rank} failed: {error}") from error
            kind = int(self._status_tensors[rank].item())
            message = None
            if kind == DISTRIBUTED_VAE_CONSUMER_ERROR:
                payload = [None]
                dist.recv_object_list(
                    payload,
                    src=rank,
                    group=self.control_group,
                )
                message = str(payload[0])
            elif kind != DISTRIBUTED_VAE_DONE:
                message = f"invalid consumer terminal status {kind}"
                kind = DISTRIBUTED_VAE_CONSUMER_ERROR
            self._statuses[rank] = (kind, message)
            del self._status_works[rank]
            del self._status_waiters[rank]
        return dict(self._statuses)

    def raise_for_status(self) -> None:
        statuses = self.poll_statuses()
        failures = [(rank, message) for rank, (kind, message) in statuses.items() if kind == DISTRIBUTED_VAE_CONSUMER_ERROR]
        if failures:
            details = " | ".join(f"rank {rank}: {message or 'unknown consumer failure'}" for rank, message in failures)
            raise DistributedVAEConsumerError(details)
        if any(kind == DISTRIBUTED_VAE_DONE for kind, _ in statuses.values()):
            raise _ConsumerRequestedStop

    def send_data(
        self,
        consumer_rank: int,
        meta_info: dict[str, Any],
        packed_tensor: torch.Tensor,
    ) -> None:
        """Send one batch and check reverse status on both sides of the send."""

        self.raise_for_status()
        envelope = dict(meta_info)
        envelope.update(
            {
                "protocol_version": DISTRIBUTED_VAE_PROTOCOL_VERSION,
                "kind": DISTRIBUTED_VAE_DATA,
            }
        )
        dist.send_object_list(
            [envelope],
            dst=consumer_rank,
            group=self.control_group,
        )
        dist.send(
            tensor=packed_tensor,
            dst=consumer_rank,
            group=self.data_group,
            tag=0,
        )
        # If the peer failed while this send was in flight, its close path
        # drained the payload above. Observe its pre-posted ERROR now, before
        # producing or sending another batch.
        self.raise_for_status()

    def send_terminal(
        self,
        kind: str,
        *,
        error: BaseException | str | None = None,
    ) -> None:
        if self._terminal_sent:
            return
        if kind not in {DISTRIBUTED_VAE_STOP, DISTRIBUTED_VAE_ERROR}:
            raise ValueError(f"invalid producer terminal kind: {kind!r}")
        envelope = {
            "protocol_version": DISTRIBUTED_VAE_PROTOCOL_VERSION,
            "kind": kind,
        }
        if error is not None:
            envelope["error"] = _format_remote_error(error)
        for rank in self.consumer_ranks:
            dist.send_object_list(
                [envelope],
                dst=rank,
                group=self.control_group,
            )
        self._terminal_sent = True

    def wait_for_terminal_statuses(self) -> dict[int, tuple[int, str | None]]:
        """Wait for one DONE/ERROR from every consumer after terminal send."""

        deadline = time.monotonic() + self.timeout_seconds
        while len(self._statuses) < len(self.consumer_ranks):
            self.poll_statuses()
            if len(self._statuses) == len(self.consumer_ranks):
                break
            if time.monotonic() >= deadline:
                missing = sorted(set(self.consumer_ranks) - set(self._statuses))
                raise TimeoutError(f"timed out waiting {self.timeout_seconds:.1f}s for distributed-VAE terminal status from ranks {missing}")
            time.sleep(0.01)
        return dict(self._statuses)


def merge_commpairs(commpairs: list) -> dict[int, CommPair]:
    """
    Merge a list of communication pairs (commpairs) that share the same producer and data-parallel settings.
    This groups consumers that need the same data.
    """
    from teleboost.engines.teletron.parallel_state import CommPair

    merge_dict = {}
    for cp in commpairs:
        key = (cp.producer, cp.dp_rank, cp.dp_size)
        if key not in merge_dict:
            merge_dict[key] = []

        consumers = cp.consumer if isinstance(cp.consumer, list) else [cp.consumer]
        merge_dict[key].extend(consumers)

    merged_list = {}
    for idx, (key, consumers_list) in enumerate(merge_dict.items()):
        new_cp = CommPair(producer=key[0], consumer=sorted(list(set(consumers_list))), dp_rank=key[1], dp_size=key[2])
        merged_list[idx] = new_cp
    return merged_list


def _set_random_seed_by_rank(seed_=1234):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different producer get different seeds.
        seed = seed_ + (10 * torch.distributed.get_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.device_count() > 0:
        #     tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


class DistDataProducer:
    """
    Distributed data producer.
    Loads and encodes data from the dataset and sends it synchronously to consumer processes via PyTorch Distributed.
    """

    def __init__(
        self,
        rank: int,
        encoder_name: str,
        device,
        build_train_valid_test_data_iterators: Callable,
        train_ds: Any = None,
        valid_ds: Any = None,
        target_train_iters: int | None = None,
        stop_event: threading.Event | None = None,
        encoder_factory: Callable[[str, Any], Any] | None = None,
        control_group=None,
        data_group=None,
    ):
        self.rank = rank
        self.device = device

        self._setup_logger()
        self.logger.info("Initialization started...")

        self.args = get_args()
        self.build_data_iterators_fn = build_train_valid_test_data_iterators
        self.encoder_factory = encoder_factory
        self.train_ds_preloaded = train_ds
        self.valid_ds_preloaded = valid_ds
        self.stop_event = stop_event or threading.Event()

        self.iteration = 0
        self.train_iteration = 0
        self.target_train_iters = int(self.args.train_iters) if target_train_iters is None else int(target_train_iters)
        if self.target_train_iters <= 0:
            raise ValueError(f"target_train_iters must be > 0; got {self.target_train_iters}")
        self.batch_size = 1

        self.same_data_group = {}

        self.modes = [TRAIN_MODE]
        if self.args.eval_iters > 0:
            self.modes.append(VALID_MODE)
        self.in_train_epilogue = False
        self.logger.info(f"Run modes: {self.modes}")

        from teleboost.engines.teletron.parallel_state import get_comm_pair

        self.comm_pairs = get_comm_pair()
        if not isinstance(self.comm_pairs, list):
            raise TypeError(f"distributed-VAE producer expected get_comm_pair() to return a list, got {type(self.comm_pairs).__name__}")
        self.merged_comm_pairs = merge_commpairs(self.comm_pairs)
        self.logger.info(f"Raw communication pairs: {self.comm_pairs}")
        self.logger.debug(f"Merged communication pairs: {self.merged_comm_pairs}")  # debug level: this output is verbose
        consumer_ranks = [int(cp.consumer) for cp in self.comm_pairs]
        self.protocol = DistributedVAEProducerProtocol(
            consumer_ranks,
            self.device,
            control_group=control_group,
            data_group=data_group,
        )

        try:
            self._initialize_consumer_state()
            _set_random_seed_by_rank(self.args.seed)
            if self.encoder_factory is None:
                raise RuntimeError("DistDataProducer requires encoder_factory from the training composition root.")

            self.encoder = self.encoder_factory(encoder_name, self.device)
            self.encoder.setup()
            self.logger.info("Encoder setup complete")

            self._create_data_iterators()
            self._initialize_queues_and_trackers()
            self._setup_profiler()

            # init timers
            self.timers = get_timers()
            self.timers.get_timer("encoder-once-time")
        except BaseException as exc:
            # READY has completed and terminal receives are already posted on
            # every consumer, so initialization failures are safe to report.
            try:
                self.protocol.send_terminal(DISTRIBUTED_VAE_ERROR, error=exc)
                self.protocol.wait_for_terminal_statuses()
            except BaseException as notify_exc:
                exc.add_note(f"Additionally failed to finish the distributed-VAE error handshake: {notify_exc!r}")
            raise

        self.logger.info("Initialization complete")

    def _setup_logger(self):
        self.logger = logging.getLogger(f"ProducerRank{self.rank}")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            # Define the handler's output format
            formatter = logging.Formatter(f"PRODUCER (Rank {self.rank}) [%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
            ch.setFormatter(formatter)

            self.logger.addHandler(ch)

        self.logger.propagate = False

    def _get_debug_dump_path(self):
        base_dir = getattr(self.args, "profile_path", None) or "."
        return os.path.join(base_dir, f"producer/raw_batch_debug_rank_{self.rank}.jsonl")

    def _get_raw_tensor_dump_path(self):
        base_dir = getattr(self.args, "profile_path", None) or "."
        return os.path.join(base_dir, f"producer/raw_batch_tensors_rank_{self.rank}.jsonl")

    def _collect_tensor_kv(self, obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
        items: dict[str, torch.Tensor] = {}
        if torch.is_tensor(obj):
            items[prefix or "root"] = obj
            return items
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                new_prefix = f"{prefix}/{k}" if prefix else str(k)
                items.update(self._collect_tensor_kv(obj[k], new_prefix))
            return items
        if isinstance(obj, list | tuple):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}/{i}" if prefix else str(i)
                items.update(self._collect_tensor_kv(v, new_prefix))
            return items
        return items

    def _get_gpu_memory_usage(self) -> str:
        """Get and format the current GPU memory usage."""
        try:
            if not torch.cuda.is_available():
                return "CUDA not available"
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            free, total = torch.cuda.mem_get_info(self.device)
            free_mb = free / 1024**2
            total_mb = total / 1024**2
            return f"GPU Mem: Alloc={allocated:.2f}MB, Reserv={reserved:.2f}MB, Free={free_mb:.2f}MB, Total={total_mb:.2f}MB"
        except Exception as e:
            return f"GPU Mem: Error getting info - {e}"

    def _get_shm_usage(self) -> str:
        """Get and format /dev/shm usage."""
        try:
            shm_usage = psutil.disk_usage("/dev/shm")
            used_mb = shm_usage.used / 1024**2
            total_mb = shm_usage.total / 1024**2
            return f"SHM Mem: Used={used_mb:.2f}MB, Total={total_mb:.2f}MB ({shm_usage.percent}%)"
        except (FileNotFoundError, AttributeError):
            return "SHM Mem: /dev/shm not found or psutil error."

    def _initialize_consumer_state(self):
        """Synchronize initial state (e.g. training iteration count) with the consumers."""
        self.logger.info("Fetching initial state from consumers...")
        states = self.protocol.receive_ready_states()
        first_state = states[0]
        if any(state != first_state for state in states[1:]):
            raise DistributedVAELifecycleError(f"distributed-VAE consumers disagree on initial state: {states}")
        self.args.iteration = first_state[0]
        self.args.consumed_train_samples = first_state[1] // self.args.distributed_vae_world_size
        self.args.consumed_valid_samples = first_state[2]
        self.logger.info(f"State sync complete. Iteration: {self.args.iteration}, Consumed Train: {self.args.consumed_train_samples}, Consumed Valid: {self.args.consumed_valid_samples}")

    def _create_data_iterators(self):
        """Create data iterators based on the merged communication pairs."""
        self.logger.info("Creating data iterators...")
        self.data_iterators = {}

        train_ds_current = self.train_ds_preloaded
        valid_ds_current = self.valid_ds_preloaded

        train_iter, valid_iter, _, train_ds_current, valid_ds_current = self.build_data_iterators_fn(is_tp_first=True, dp_rank=0, dp_size=1, train_ds_prev=train_ds_current, valid_ds_prev=valid_ds_current, return_ds=True)

        self.data_iterators[TRAIN_MODE] = train_iter
        if VALID_MODE in self.modes:
            self.data_iterators[VALID_MODE] = valid_iter

        self.logger.info("Data iterators created")

    def _initialize_queues_and_trackers(self):
        self.same_data_group = {}
        for idx, mcp in self.merged_comm_pairs.items():
            first_consumer = mcp.consumer[0]
            self.same_data_group[first_consumer] = mcp.consumer
        """Initialize data queues and send/produce counters."""
        self.logger.info("Initializing queues and counters...")
        all_consumer_ranks = [cp.consumer for cp in self.comm_pairs]
        self.data_queues = {}
        self.produced_count = {}
        self.sended_count = {}

        for mode in self.modes:
            self.data_queues[mode] = {rank: collections.deque() for rank in all_consumer_ranks}
            self.produced_count[mode] = {rank: 0 for rank in all_consumer_ranks}
            self.sended_count[mode] = {rank: 0 for rank in all_consumer_ranks}
        self.logger.info("Queues and counters initialized")

    def _setup_profiler(self):
        """Set up the PyTorch profiler if enabled in the config."""
        self.profiler = None
        if self.args.producer_profile:
            prof_save_path = os.path.join(self.args.profile_path, f"producer/rank_{self.rank}.json")
            os.makedirs(os.path.dirname(prof_save_path), exist_ok=True)
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True, on_trace_ready=lambda p: p.export_chrome_trace(prof_save_path), record_shapes=True)
            self.logger.info(f"Profiler configured; traces will be saved to: {prof_save_path}")

    def _produce_and_enqueue_data(self, idx: int, mcp: CommPair, mode: str, raw_batch=None):
        """Produce data from the data iterator, encode it, and enqueue it."""
        first_consumer = mcp.consumer[0]

        if raw_batch is None:
            try:
                self.logger.debug(f"PRE-GET-RAW-DATA for mode [{mode}] iter [{idx}]")
                raw_batch = next(self.data_iterators[mode])
            except StopIteration:
                self.logger.warning(f"Warning: data iterator {idx} for mode [{mode}] is exhausted")
                return

        self.logger.debug(f"POST-GET-RAW-DATA for mode [{mode}] iter [{idx}]")
        self.logger.debug(f"PRE-ENCODE: {self._get_gpu_memory_usage()}")

        self.timers.start_timer("encoder-once-time")
        tensors_to_send = self.encoder.encode(raw_batch)
        self.timers.stop_timer("encoder-once-time")
        encode_time = self.timers.get_elapsed_time("encoder-once-time")

        self.logger.debug(f"POST-ENCODE: {self._get_gpu_memory_usage()}")

        self.produced_count[mode][first_consumer] += 1
        item_index = self.produced_count[mode][first_consumer]

        self.logger.info(f"mode [{mode}] iter [{idx}]: produced {item_index} data, encoded {self._infer_batch_shape(raw_batch)} data cost {encode_time:.3f}s")

        for consumer_rank in self.same_data_group[first_consumer]:
            self.data_queues[mode][consumer_rank].append(tensors_to_send)
            self.logger.debug(f"QUEUE for Consumer {consumer_rank}: push {item_index} data")

    def _send_data_from_queue(self, cp: CommPair, mode: str):
        consumer_rank = cp.consumer

        self.sended_count[mode][consumer_rank] += 1
        item_index = self.sended_count[mode][consumer_rank]

        tensors_to_send = self.data_queues[mode][consumer_rank].popleft()
        self.logger.debug(f"QUEUE for Consumer {consumer_rank}: get {item_index} data for sending")

        # 1) flatten
        flat = self._flatten_tensor_tree(tensors_to_send)
        paths = [p for p, _ in flat]
        tensors = [t for _, t in flat]

        # 2) pack (reuses the existing pack logic)
        packed_tensor = self.encoder._pack_tensors(tensors)

        # 3) meta: contains paths + shapes (both required at minimum)
        meta_info = {
            "paths": paths,
            "shapes": {p: list(t.shape) for p, t in flat},
            # optional
            "dtypes": {p: str(t.dtype) for p, t in flat},
        }

        resource_status = f"{self._get_gpu_memory_usage()} | {self._get_shm_usage()}"
        self.logger.debug(f"PRE-SEND-META to Consumer {consumer_rank} (item {item_index}): num_tensors={len(tensors)}, keys={paths}. Status: {resource_status}")
        self.protocol.send_data(consumer_rank, meta_info, packed_tensor)
        self.logger.debug(f"POST-SEND-META to Consumer {consumer_rank} (item {item_index}): success")

        self.logger.debug(f"PRE-SEND-TENSOR to Consumer {consumer_rank} (item {item_index}): shape={packed_tensor.shape}, dtype={packed_tensor.dtype}")
        self.logger.debug(f"POST-SEND-TENSOR to Consumer {consumer_rank} (item {item_index}): success")

    def _get_mode_to_process(self, in_train_epilogue=False):
        """Determine which mode (train or valid) to process in the current step."""
        if self.in_train_epilogue:
            assert getattr(self, "train_or_valid_mode", None) is not None
            return self.train_or_valid_mode
        if VALID_MODE in self.modes:
            train_data_count = self.args.eval_interval
            eval_data_count = self.args.eval_iters
            num_sended_in_cycle = self.iteration % (train_data_count + eval_data_count)
            mode_to_process = TRAIN_MODE if num_sended_in_cycle < train_data_count else VALID_MODE
        else:
            mode_to_process = TRAIN_MODE
        return mode_to_process

    def _infer_batch_shape(self, raw_batch):
        if isinstance(raw_batch, dict):
            if "images" in raw_batch:
                return raw_batch["images"].shape
            elif "chosen" in raw_batch and "images" in raw_batch["chosen"]:
                return raw_batch["chosen"]["images"].shape
        return None

    def _main_loop_step(self):
        """Execute one main-loop step: produce and send data."""
        # 1. Determine which mode to process
        mode_to_process = self._get_mode_to_process()

        self.logger.debug(f"Start produce data for mode: {mode_to_process}")

        # 2. Produce data
        # Fetch one batch first
        raw_batch = next(self.data_iterators[mode_to_process])
        # raw_tensor_kv = self._collect_tensor_kv(raw_batch)
        # if raw_tensor_kv:

        self._infer_batch_shape(raw_batch)

        num_micro_batchs = self.args.global_batch_size // self.args.data_parallel_size
        for i in range(num_micro_batchs):
            self.logger.debug(f"Start num_micro_batch {i} produce data for mode: {mode_to_process}")
            for idx, mcp in self.merged_comm_pairs.items():
                if i == 0 and idx == 0:
                    self._produce_and_enqueue_data(idx, mcp, mode_to_process, raw_batch)
                else:
                    self._produce_and_enqueue_data(idx, mcp, mode_to_process)
            self.logger.debug(f"End num_micro_batch {i} produce data ")

            self.logger.debug(f"Start num_micro_batch {i} send data for mode: {mode_to_process}")
            for cp in self.comm_pairs:
                self._send_data_from_queue(cp, mode_to_process)
            self.logger.debug(f"End num_micro_batch {i} send data")

        # 3. Increment the iteration counter; each iteration corresponds to one global batch
        self.iteration += 1
        if mode_to_process == TRAIN_MODE:
            self.train_iteration += 1

    def train_epilogue(self):
        if VALID_MODE in self.modes:
            self.in_train_epilogue = True
            self.train_or_valid_mode = VALID_MODE
            for _ in range(self.args.eval_iters):
                self._main_loop_step()

    def run(self):
        """Run until completion, then finish the explicit peer handshake.

        This method deliberately does not call a world barrier, destroy the
        process group, or abort it.  The process group belongs to the Ray
        worker's main thread and remains usable after the producer thread has
        joined.
        """
        failure: BaseException | None = None
        try:
            self.logger.info("Main loop started")

            while self.train_iteration < self.target_train_iters:
                if self.stop_event.is_set():
                    self.logger.info("Producer stop requested by worker lifecycle")
                    break
                self.protocol.raise_for_status()
                if self.profiler:
                    if self.iteration == self.args.profile_step_start:
                        self.logger.info("Starting profiler...")
                        self.profiler.start()
                    if self.iteration == self.args.profile_step_end:
                        self.logger.info("Stopping profiler...")
                        self.profiler.stop()
                        self.logger.info("Profiling data saved")

                self._main_loop_step()

                time.sleep(0.001)

            self.logger.info("All consumers reached the target data volume. Main loop finished.")
            if not self.stop_event.is_set():
                self.train_epilogue()
        except _ConsumerRequestedStop:
            self.logger.info("Consumer completed; stopping producer normally")
        except BaseException as exc:
            failure = exc
            self.logger.exception(f"!!!--- Fatal exception in producer main loop ---!!! e = {exc}")
        finally:
            terminal_kind = DISTRIBUTED_VAE_ERROR if failure is not None else DISTRIBUTED_VAE_STOP
            try:
                self.protocol.send_terminal(terminal_kind, error=failure)
                self.protocol.wait_for_terminal_statuses()
            except BaseException as lifecycle_exc:
                if failure is None:
                    failure = lifecycle_exc
                else:
                    failure.add_note(f"Additionally failed to finish the distributed-VAE terminal handshake: {lifecycle_exc!r}")
            self.logger.info("Producer lifecycle handshake complete")

        if failure is not None:
            raise failure

    def _flatten_tensor_tree(self, obj: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
        """
        Flatten a nested dict into [(path, tensor), ...], with paths like "chosen/latents".
        Guarantees a stable order by sorting dict keys (insertion order would also work, but sorting is more robust).
        """
        items: list[tuple[str, torch.Tensor]] = []
        if isinstance(obj, torch.Tensor):
            items.append((prefix, obj))
            return items
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                new_prefix = f"{prefix}/{k}" if prefix else str(k)
                items.extend(self._flatten_tensor_tree(obj[k], new_prefix))
            return items
        raise TypeError(f"Unsupported type in tensors_to_send tree: {type(obj)} at prefix={prefix}")
