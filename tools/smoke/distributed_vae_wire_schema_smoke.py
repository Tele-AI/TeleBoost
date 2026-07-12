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
"""Regression guard for the distributed-VAE producer/consumer wire schema.

The distributed-VAE DPO path ships encoder outputs from VAE-producer ranks to
DiT-consumer ranks over raw ``dist.send``/``dist.recv``. Both ends MUST size
their buffers from the SAME schema or the wire desyncs. A past bug had the
consumer (``teleboost/training/dit_batch_loader.py``) size its receive from
``wan_native``'s hardcoded 4-field I2V schema while the producer emitted the
config-driven ``encoder_schema`` (3 fields for wan_dpo_i2v) — a silent
collective-size mismatch. The fix routes both ends through
``WanTeletronEncoder.get_output_schema()``.

This smoke reproduces the exact receive+unpack path with the REAL project
functions and the REAL i2v config, on a 2-rank gloo group — no model weights,
no video data. It is the runtime gate for that fix.

Run (any host with the megatron venv; 2 CPU ranks suffice):

    PYTHONPATH=<repo>:<repo>/third_party GLOO_SOCKET_IFNAME=lo \
      python -m torch.distributed.run --nproc_per_node=2 \
      tools/smoke/distributed_vae_wire_schema_smoke.py --mode postfix

Expected:
  --mode postfix (current code): "RESULT: OK (schema unified, data roundtrips)"
  --mode prefix  (old 4-field consumer): gloo raises "Received data size
      doesn't match expected size ... distributed collective mismatch"
"""

import argparse
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

# Inject the minimal global args so set_config()/get_args() resolve the i2v
# config without booting the full megatron arg machinery.
import teleboost.engines.teletron.runtime_state as gv

gv._GLOBAL_ARGS = SimpleNamespace(
    config_path="teleboost.programs.wan.dpo.wan_dpo_i2v.config",
    consumer_models_num=1,  # non-MoE
)

from teleboost.models.wan.teletron.wan_teletron_encoder import (  # noqa: E402
    PROPERTY_DIMS,
    WanTeletronEncoder,
)
from teleboost.training.utils import unpack_tensors  # noqa: E402


def _prefix_consumer_schema():
    """The old wan_native hardcoded (non-MoE) consumer schema."""
    try:
        from teleboost.models.wan_native.encoder.wan_encoder import WanVideoEncoder

        return WanVideoEncoder.get_output_schema()
    except ModuleNotFoundError:
        return ["context", "img_clip_feature", "img_emb_y", "latents"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefix", "postfix"], required=True)
    args = ap.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    producer_schema = WanTeletronEncoder.get_output_schema()

    if rank == 0:
        tensors, shapes = [], []
        for k in producer_schema:
            ndim = PROPERTY_DIMS[k]
            shape = [2] * ndim
            tensors.append(torch.arange(2**ndim, dtype=torch.float32).reshape(shape))
            shapes.extend(shape)
        tensors_info = torch.tensor(shapes, dtype=torch.int32)
        packed = torch.cat([t.flatten() for t in tensors])
        print(f"[producer] schema={producer_schema} info_size={tensors_info.numel()}", flush=True)
        dist.send(tensors_info, dst=1)
        dist.send(packed, dst=1)
        print(f"[producer] sent tensors_info({tensors_info.numel()}) + packed({packed.numel()})", flush=True)
        dist.barrier()
        return

    consumer_schema = producer_schema if args.mode == "postfix" else _prefix_consumer_schema()
    info_size = sum(PROPERTY_DIMS[k] for k in consumer_schema)
    print(f"[consumer mode={args.mode}] schema={consumer_schema} info_size={info_size}", flush=True)

    tensors_info = torch.empty(info_size, dtype=torch.int32)
    dist.recv(tensors_info, src=0)

    intervals, start = [0], 0
    for k in consumer_schema:
        dims = PROPERTY_DIMS[k]
        size = 1
        for d in tensors_info[start : start + dims].tolist():
            size *= d
        start += dims
        intervals.append(intervals[-1] + size)
    packed = torch.empty(intervals[-1], dtype=torch.float32)
    dist.recv(packed, src=0)

    prod_total = sum(2 ** PROPERTY_DIMS[k] for k in producer_schema)
    unpacked = unpack_tensors(packed, intervals)
    data_ok = consumer_schema == producer_schema and intervals[-1] == prod_total and len(unpacked) == len(producer_schema) and torch.equal(unpacked[0], torch.arange(2 ** PROPERTY_DIMS[producer_schema[0]], dtype=torch.float32))
    print(f"[consumer] total={intervals[-1]} producer_total={prod_total} fields={len(unpacked)}/{len(producer_schema)} roundtrip={data_ok}", flush=True)
    print(f"RESULT: {'OK (schema unified, data roundtrips)' if data_ok else 'MISMATCH'}", flush=True)
    dist.barrier()
    sys.exit(0 if data_ok else 4)


if __name__ == "__main__":
    main()
