# TeleBoost Quickstart

This walks through (1) building the image, (2) running a smoke test on
FakeDataset, (3) bringing up real DPO training, and (4) writing your
own dataset adapter.

---

## 0. Prerequisites

* **Hardware**: 8×NVIDIA H100 / H200 / H800 (SM 9.0) for the headline
  config. SM 8.0 GPUs (A100, etc.) work with `--build-arg BUILD_FA3=0`.
* **Driver**: CUDA 13.0-compatible NVIDIA driver (≥575.x recommended).
* **Disk**: 200 GB free for the image; more for checkpoints.
* **RAM**: 256 GB+ host memory recommended for distributed_vae mode.

---

## 1. Build the image

```bash
git clone https://github.com/Tele-AI/TeleBoost.git
cd TeleBoost

# Hopper (H100/H200/H800):
docker build -t teleboost:mc0.16.1 .
# ~80 min: pip deps (5m) + flash-attn 2 source build (35m) + flash-attn 3 source build (40m)

# Non-Hopper (skip flash-attn 3, ~35 min total):
docker build --build-arg BUILD_FA3=0 -t teleboost:mc0.16.1 .

# Behind GFW, set a pip mirror for the python deps stage:
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t teleboost:mc0.16.1 .
```

What's inside (verified ABI-aligned):

| Component | Version | Source |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch:25.09-py3` | NGC |
| torch | 2.9.0a0+nv25.09 | NGC bundled |
| CUDA | 13.0 | NGC bundled |
| transformer_engine | 2.7.0 | NGC bundled |
| apex (FusedAdam, etc.) | NGC build | NGC bundled |
| flash-attn 2 | 2.8.3 | source build |
| flash-attn 3 (Hopper) | 2.8.3 | source build |
| megatron-core | 0.16.0 (verl-aligned; DPO's `MEGATRON_LM_DIR` clones Megatron-LM `core_v0.16.1`) | source build |
| deepspeed | **0.17.5** (pinned) | pip |
| All other pythons | see `requirements.txt` | pip |

> **Why deepspeed 0.17.5 specifically?** 0.17.6+ replaced the simple
> multi-call epilogue that Gradient Decoupled DPO relies on with a
> state machine requiring the `DeepSpeedEngine` wrapper. Do not bump.

---

## 2. Run the container

```bash
docker run -it --rm --gpus all --shm-size 512G --network host \
  -v $(pwd):/workspace/TeleBoost \
  -v /your/data/dir:/data \
  teleboost:mc0.16.1 zsh

# inside the container:
cd /workspace/TeleBoost
nvidia-smi  # confirm 8 GPUs visible
```

`--shm-size 512G` is required for distributed-VAE producer/consumer
sharing across DataLoader workers. `--network host` is needed if you
plan to use multi-node torchrun via TCP.

---

## 3. Smoke test on FakeDataset (no real data needed)

The split-DPO micro-bench harness referenced by earlier revisions
(`tests/bench_dpo_split.py`) is part of the internal baseline tooling and is
NOT shipped in this release. The checks that ship are:

```bash
# Inside the container
cd /workspace/TeleBoost

# 1. Import sanity (see INSTALL.md "Smoke test" for the full module list).

# 2. CPU-only algorithm tests - no GPU, no data:
python3 -m pytest tests/test_sigma_schedule.py tests/test_solver_contract.py \
    tests/test_noise_weight.py tests/test_trajectory_branch.py -q

# 3. CP grad-reduce regression (4 GPUs, ~30 s):
torchrun --nproc_per_node=4 tests/special_distributed/test_cp_grad_reduce.py
```

If these pass, the stack is healthy; real training runs go through
`recipes/wan_grpo_fsdp/run.sh` (GRPO) or `recipes/wan_dpo_teletron/` (DPO).

---

## 4. Real DPO training

The canonical entry is `recipes/wan_dpo_teletron/run.sh`. It expects
two external dependencies on `PYTHONPATH`:

```bash
# Megatron-LM at the core_v0.16 tag
git clone -b core_v0.16.1 https://github.com/NVIDIA/Megatron-LM.git /megatron
export MEGATRON_LM_DIR=/megatron
```

Wan-Video upstream is vendored (a plain in-tree copy, not a submodule)
under `third_party/wan/` — a normal clone already has it.

Wan checkpoints — real-train uses 14B I2V; smoke / replay use 1.3B T2V:

```bash
# DiT weights — convert HF safetensors to a megatron 'release' checkpoint
# directory once, then pass via WAN_DIT_CKPT:
python tools/convert_wan_to_teletron.py \
    --src '/path/to/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors' \
    --dst /path/to/Wan2.1-I2V-14B-480P-teletron \
    --roundtrip-check       # verify rename rules are bijective; bit-exact

export WAN_DIT_CKPT=/path/to/Wan2.1-I2V-14B-480P-teletron  # pre-converted teletron-format
export WAN_HF_DIR=/path/to/Wan2.1-I2V-14B-480P           # upstream HF Wan ckpt
```

Launch the default 8-GPU distributed-VAE training (`real-train` mode):

```bash
export WAN_DPO_DATA_DIR=/path/to/dpo_csv                # 8-shard preference-pair CSVs
export WAN_T2V_1_3B_DIR=/path/to/Wan2.1-T2V-1.3B        # VAE + T5 weights
export WAN_I2V_14B_DIR=/path/to/Wan2.1-I2V-14B-480P     # T5 tokenizer + CLIP image encoder
N_PROC=8 N_VAE=2 REAL_TRAIN_ITERS=1 \
  bash recipes/wan_dpo_teletron/run.sh
```

> **DPO eval is not supported.** `dpo_loss.py`'s `forward_step`
> returns a 5-element list of losses; megatron's eval reducer can't
> divide that. The launcher ships with `--eval-iters 0`. Until a
> DPO-aware eval reducer lands, leave eval disabled.

Key knobs (override via env vars):

| Env var | Default | Notes |
|---|---|---|
| `N_PROC` | 4 (smoke) / 8 (real-train) | total GPUs on this node |
| `N_VAE` | 2 | VAE-producer rank count (out of `N_PROC`); the rest are DiT |
| `REAL_TRAIN_ITERS` | 1 | iter count for `real-train` mode (sets `trainer.total_training_steps`; `optim.total_training_steps` follows via yaml interpolation) |
| `WAN_DPO_DATA_DIR` | — | preference-pair CSV shards (required for real-train) |
| `WAN_T2V_1_3B_DIR` | — | upstream 1.3B T2V dir for VAE + T5 (required for real-train) |
| `WAN_I2V_14B_DIR` | — | upstream 14B I2V dir for T5 tok + CLIP (required for real-train) |
| `REAL_GBS` | `N_PROC - N_VAE` | global_batch_size for `real-train` (default matches DiT-DP) |
| `PHASE3_DUMP_DIR` | `/tmp/phase3_dumps` | where `phase3-replay` reads golden dumps from |

See [`recipes/wan_dpo_teletron/README.md`](recipes/wan_dpo_teletron/README.md) for the 3 supported
modes (`recipe-smoke`, `phase3-replay`, `real-train`).

---

## 5. Write your own dataset

`teleboost.datasets.DPODatasetBase` documents the schema your
`__getitem__` must return:

```python
import torch
from teleboost.datasets import DPODatasetBase, DATASETS


class MyDPODataset(DPODatasetBase):
    """Loads pre-encoded chosen/rejected latents from disk."""

    def __init__(self, manifest_csv, **kwargs):
        import pandas as pd
        self.rows = pd.read_csv(manifest_csv).to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ch = torch.load(row["chosen_pkl"])      # pre-encoded latents
        rj = torch.load(row["rejected_pkl"])
        ctx = torch.load(row["text_emb_pkl"])   # T5 embedding

        return {
            "context": ctx,
            "chosen": {
                "latents":          ch["latents"],
                "img_clip_feature": ch["clip_feature"],
                "img_emb_y":        ch["first_frame_latent"],
            },
            "rejected": {
                "latents":          rj["latents"],
                "img_clip_feature": rj["clip_feature"],
                "img_emb_y":        rj["first_frame_latent"],
            },
        }


# Register so build_dataset("MyDPODataset") works
DATASETS.register_module(MyDPODataset)
```

Then in your config (see `teleboost/programs/wan/dpo/wan_dpo_i2v.py` /
`teleboost/programs/wan/dpo/wan_dpo_t2v.py`):

```python
config = dict(
    dataset=dict(
        type="MyDPODataset",
        manifest_csv="/data/my_dpo_pairs.csv",
    ),
    # ... rest of config (model_config, encoder, etc.)
)
```

**Schema notes**:

* Tensors should be CPU + bf16 or fp32; teleboost auto-casts to model
  dtype.
* Batch dim (B) is added by the DataLoader; do not prepend it.
* `chosen` and `rejected` MAY have different `T/H/W` — each branch
  goes through its own `_run_branch` forward pass. Mismatched-shape
  support is regression-tested (T-scale up to 8×, H/W up to 2×).
* For testing without real data, just use `FakeDataset` (already
  registered):

  ```python
  config = dict(dataset=dict(type="FakeDataset"), ...)
  ```

---

## 6. Common issues

**`ImportError: cannot import name 'backward_prologue'`** — your
deepspeed got bumped past 0.17.5. `pip install deepspeed==0.17.5`
inside the container.

**`KeyError: torch.bfloat16` in `ipg_buckets`** — `lr_scheduler.py`
must pass `communication_data_type=torch.bfloat16` when `args.bf16`. The
shipped code does this; only triggers if you write a custom optimizer
setup. See `teleboost/training/lr_scheduler.py` for the canonical pattern.

**Verify the dataset registry imports:**

```bash
python3 -c "from teleboost.datasets import FakeDataset; print(FakeDataset)"
```

**OOM on Wan 14B 40-layer with `--use-distributed-optimizer` (megatron)
instead of `--use-zero2`** — the megatron distributed optimizer doesn't
implement Gradient Decoupled DPO; use `--use-zero2`.

Want to verify Gradient Decoupled DPO math equivalence locally? See
[`recipes/wan_dpo_teletron/README.md#precision-alignment`](recipes/wan_dpo_teletron/README.md#precision-alignment).
