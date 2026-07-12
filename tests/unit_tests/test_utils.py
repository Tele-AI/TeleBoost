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
import logging
import multiprocessing as mp
import os
import queue
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def _run_worker(func, rank, nprocs, result_queue, args):
    """Run one worker while preserving a non-zero child exit status."""
    try:
        func(rank, nprocs, result_queue, *args)
    except BaseException:
        traceback.print_exc()
        raise


def spawn(nprocs, func, *args, timeout_seconds=None):
    """Run multiprocessing tests and return a reliable in-memory result queue.

    The previous helper joined children before reading ``multiprocessing.Queue``.
    Large NumPy/tensor payloads can fill the pipe and deadlock that ordering; it
    also ignored child exit codes, so the parent often failed later with a
    misleading "missing payload" assertion. Drain concurrently, enforce a
    bounded group timeout, and surface the actual failed ranks instead.
    """
    processes = []
    result_queue = mp.Queue()
    collected = []
    if isinstance(func, list):
        assert len(func) == nprocs
    else:
        func = [func] * nprocs
    for i in range(nprocs):
        p = mp.Process(
            target=_run_worker,
            args=(func[i], i, nprocs, result_queue, args),
        )
        p.start()
        processes.append(p)

    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("TELEBOOST_TEST_PROCESS_TIMEOUT_SECONDS", "300"))
    deadline = time.monotonic() + timeout_seconds
    while any(process.is_alive() for process in processes):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            alive_ranks = [rank for rank, process in enumerate(processes) if process.is_alive()]
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            result_queue.close()
            result_queue.join_thread()
            raise TimeoutError(f"multiprocessing test timed out after {timeout_seconds:g}s; alive ranks: {alive_ranks}")
        try:
            collected.append(result_queue.get(timeout=min(0.1, remaining)))
        except queue.Empty:
            pass

    for process in processes:
        process.join()

    # Child feeder threads have now flushed. A short blocking drain avoids the
    # documented unreliability of multiprocessing.Queue.empty().
    while True:
        try:
            collected.append(result_queue.get(timeout=0.1))
        except queue.Empty:
            break

    failures = [f"rank {rank} exitcode={process.exitcode}" for rank, process in enumerate(processes) if process.exitcode != 0]
    result_queue.close()
    result_queue.join_thread()
    if failures:
        raise RuntimeError("multiprocessing worker failure: " + ", ".join(failures))

    # Existing tests consume a queue. Returning a local Queue keeps that API,
    # while making empty()/get() deterministic after all children finish.
    output_queue = queue.Queue()
    for item in collected:
        output_queue.put(item)
    return output_queue
