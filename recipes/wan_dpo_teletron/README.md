# DPO recipe

Memory-efficient DPO for video diffusion models, built as a verl
`@EngineRegistry` plug-in. Drives a Wan-family DiT under DPO loss with
**Gradient Decoupled DPO** — per-branch backward + immediate
reduce-scatter — for ~40% peak memory cut and ~15× longer context
versus single-backward DPO.

## Layout

```
teleboost/programs/wan/dpo/
├── main.py                  verl Hydra entry point
├── megatron_wan.py          MegatronEngineWanVideo verl plug-in
├── args_adapter.py          verl OmegaConf → teleboost argparse.Namespace bridge
├── wan_model_config.py      WanModelConfig (HFModelConfig stub for Wan)
├── dpo_loss.py              forward_step + dpo_loss_func (DPO core math)
├── wan_t2v_arch.py   Wan-1.3B T2V model-arch config dict
├── wan_dpo_i2v.py           teleboost production config (Wan 14B I2V, env-driven paths)
├── wan_dpo_t2v.py           teleboost dev/smoke config (Wan 1.3B T2V, overrides i2v base)
recipes/wan_dpo_teletron/
├── config.yaml              declarative program identity
└── run.sh                   env-var-driven launcher (3 modes)
```

## Modes

The launcher `run.sh` dispatches on `TELEBOOST_DPO_MODE`:

| Mode | GPUs | What it does |
|------|------|--------------|
| `real-train` (default) | 8 (2 VAE + 6 DiT) | Full end-to-end with the real preference-pair dataloader (`distributed_vae=true` + Wan VAE producer thread + DiT-side `DistVAEConsumerBatchLoader`). |
| `recipe-smoke` | 4 | Stub preference-pair batch through the verl-recipes path; verifies init chain + split-DPO multi-backward fires. ~3 min wall. |
| `phase3-replay` | 4 | Replay pre-recorded baseline dumps through `model.forward` of the verl recipes and compare per-pair `noise_pred` + DPO loss. **Bit-exact precision anchor** vs a reference run. Requires `PHASE3_DUMP_DIR` to point at the dumps. |

## Quickstart

Required env (every mode):

```bash
# Build / install dependencies (see top-level INSTALL.md)
export MEGATRON_LM_DIR=/path/to/Megatron-LM  # exact constraints/upstreams/megatron-lm.txt revision

# Wan checkpoints (real-train uses 14B I2V; smoke / replay use 1.3B T2V)
export WAN_DIT_CKPT=/path/to/Wan2.1-I2V-14B-480P-teletron  # pre-converted teletron-format
export WAN_HF_DIR=/path/to/Wan2.1-I2V-14B-480P             # upstream HF Wan ckpt
```

Get the teletron-format checkpoint by running `teleboost-convert-wan-to-teletron`
once offline on the upstream Wan release.

Run the default 8-GPU distributed-VAE training (`real-train` mode):

```bash
# real-train extra required env (read by teleboost.programs.wan.dpo.wan_dpo_i2v at import time):
export WAN_DPO_DATA_DIR=/path/to/dpo_csv               # 8-shard preference-pair CSVs
export WAN_T2V_1_3B_DIR=/path/to/Wan2.1-T2V-1.3B       # VAE + T5 weights
export WAN_I2V_14B_DIR=/path/to/Wan2.1-I2V-14B-480P    # T5 tokenizer + CLIP image encoder

bash recipes/wan_dpo_teletron/run.sh
# → [real_train_step] iter=1/1 metrics={'loss': 0.6924, ...}
```

Run a 4-GPU smoke through the verl-recipes path (1.3B T2V, no dataset needed):

```bash
TELEBOOST_DPO_MODE=recipe-smoke N_PROC=4 \
  WAN_DIT_CKPT=/path/to/Wan2.1-T2V-1.3B-teletron \
  WAN_HF_DIR=/path/to/Wan2.1-T2V-1.3B \
  bash recipes/wan_dpo_teletron/run.sh
# → [teleboost-dpo] running smoke_train_step ...
# → [teleboost-dpo] smoke_train_step completed
```

Replay pre-recorded baseline dumps for precision alignment:

```bash
TELEBOOST_DPO_MODE=phase3-replay N_PROC=4 \
  WAN_DIT_CKPT=/path/to/Wan2.1-T2V-1.3B-teletron \
  WAN_HF_DIR=/path/to/Wan2.1-T2V-1.3B \
  PHASE3_DUMP_DIR=/path/to/golden/dumps \
  bash recipes/wan_dpo_teletron/run.sh
# → replay compares noise_pred + DPO loss against the reference dumps
```

## Feature constraints preserved bit-for-bit vs reference

| # | Constraint |
|---|------------|
| 1 | **Split-DPO backward** — list-loss → per-loss `zero_optimizer.backward(t) + overlapping_partition_gradients_reduce_epilogue()`. Full per-layer grads reduce-scatter before the next backward starts (~½ peak mem, math-equivalent to single backward(sum) within bf16 ULP). |
| 2 | **TCP** (teleboost tensor-context-parallel — ulysses-style spatial-temporal split for video). |
| 3 | **Separated VAE** (`distributed_vae=true` + `distributed_vae_world_size`). Producer (VAE) / consumer (DiT) rank split at mesh-init time; VAE ranks run an encoder loop and `dist.send` encoded latents; DiT ranks `dist.recv` via `DistVAEConsumerBatchLoader`. |
| 4 | **DPO eval disabled** (forward returns a 5-element list that megatron's eval reducer can't divide). |

## Precision alignment

The DPO recipes ships with a precision anchor against the reference
standalone-megatron implementation. The `phase3-replay` mode replays recorded
reference dumps through the verl-recipes `model.forward` to check the
alignment. The reference dumps and the internal baseline generator that
produced them are not part of this OSS release.

## FSDP backend

`MegatronEngineWanVideo` is the only engine registered today
(`backend="megatron"`). An `FSDPEngineWanVideo` plug-in would need to
preserve constraints #1–#3 above; the split-DPO backward in particular
does not port trivially because the current Megatron implementation uses
DeepSpeed-ZeRO's `overlapping_partition_gradients_reduce_epilogue` hook,
which has no direct FSDP analogue. The FSDP path can self-anchor via
two invariances (CP on/off invariance + split/no-split invariance) so
it ships independently of the megatron precision gate, but the
engineering investigation is non-trivial. Not in this release.

## Troubleshooting

| Failure | Cause |
|---------|-------|
| `ImportError: verl` | Dockerfile verl install step didn't run / `--no-deps` install missed a dep. |
| `ModuleNotFoundError: megatron.training` | `MEGATRON_LM_DIR` not exported or not on `PYTHONPATH` — use the exact checkout in `constraints/upstreams/megatron-lm.txt`. |
| `lock_error from tensordict` | `teleboost.patches.tensordict_compat` did not load — the DPO entrypoint must call `apply_runtime_patches()` before importing verl symbols. |
| `KeyError: 'video_diffusion'` from `EngineRegistry` | `teleboost.programs.wan.dpo.megatron_wan` not imported before worker init — `_TeleboostTrainingWorker.__init__` imports it explicitly; check that path. |
| Distributed-VAE terminal/error timeout | All ranks must enter the same `real_train_step` RPC so the dedicated Gloo control group and READY/DATA/DONE/ERROR protocol are created in identical order. Do not enable distributed VAE in replay/smoke modes. |
