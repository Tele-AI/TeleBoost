# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Two-rank TeleTron initialize/destroy/reinitialize lifecycle smoke.

Run explicitly (this is not a pytest module):

    torchrun --standalone --nproc-per-node=2 \
      tests/special_distributed/teletron_lifecycle_smoke.py
"""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=3),
        device_id=torch.device("cuda", local_rank),
    )

    import megatron.core.parallel_state as mpu

    from teleboost.engines.teletron import set_global_args
    from teleboost.engines.teletron.megatron_adaptor import install
    from teleboost.engines.teletron.parallel_state import restore_distributed_op_patches

    set_global_args(
        SimpleNamespace(
            distributed_vae=False,
            distributed_vae_world_size=0,
            consumer_models_num=1,
            dit_world_size=dist.get_world_size(),
        )
    )
    install()
    initialize_wrapper = mpu.initialize_model_parallel
    destroy_wrapper = mpu.destroy_model_parallel
    native_collectives = {
        name: getattr(dist, name)
        for name in (
            "barrier",
            "all_reduce",
            "_all_gather_base",
            "get_world_size",
            "broadcast",
        )
    }

    try:
        for generation in range(2):
            # Calling install again after the previous generation's destroy is
            # supported and must not stack lifecycle wrappers.
            install()
            assert mpu.initialize_model_parallel is initialize_wrapper
            assert mpu.destroy_model_parallel is destroy_wrapper

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                create_gloo_process_groups=True,
            )
            assert mpu.is_initialized()
            assert dist.barrier is not native_collectives["barrier"]

            barrier_work = dist.barrier(async_op=True)
            assert barrier_work is not None
            barrier_work.wait()

            value = torch.tensor(
                [17 + generation if dist.get_rank() == 0 else -1],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            broadcast_work = dist.broadcast(value, src=0, async_op=True)
            assert broadcast_work is not None
            broadcast_work.wait()
            assert value.item() == 17 + generation

            reduced = torch.tensor(
                [dist.get_rank() + 1],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            reduce_work = dist.all_reduce(reduced, async_op=True)
            assert reduce_work is not None
            reduce_work.wait()
            assert reduced.item() == 3

            mpu.destroy_model_parallel()
            assert not mpu.is_initialized()
            for name, original in native_collectives.items():
                assert getattr(dist, name) is original, name

        # Native WORLD barrier after the second teardown proves restoration is
        # usable, not just identity-equal.
        dist.barrier()
        if dist.get_rank() == 0:
            print("TeleTron lifecycle smoke passed: 2 generations, async collectives restored")
    finally:
        restore_distributed_op_patches()
        if mpu.is_initialized():
            mpu.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
