# Support and validation matrix

TeleBoost distinguishes implemented code from validated product claims. A
passing unit test does not certify every checkpoint, GPU architecture, or
distributed topology.

## Validation levels

| Level | Meaning |
|---|---|
| Core | Dependency-light math, config, architecture, and package contracts. |
| Training integration | Pinned verl/Ray/Diffusers worker and trainer contracts without a production model. |
| Compact GPU | Real architecture and distributed kernels with compact/random weights. |
| Real checkpoint smoke | A named user checkpoint loaded for one bounded request or forward. |
| End-to-end training | A documented multi-step run, save/resume, and quality gate on a stated topology. |

## Wan model paths

| Path | Implemented workflows | Strongest repository gate | Boundary |
|---|---|---|---|
| Wan 2.1 T2V | FSDP GRPO-family training, BGPO, VIPO, TempFlow, rollout/recompute, Ulysses CP | Compact GPU plus separate real-checkpoint one-step smoke | Real checkpoint smoke is not full training certification. |
| Wan 2.2 dual-DiT T2V | FSDP rollout/recompute with high/low-noise model selection | Compact GPU | LoRA is rejected for the dual path; geometry is explicit config. |
| Wan 2.1 I2V 14B | TeleTron/Megatron DPO, separated VAE workers, TP/CP, checkpoint conversion | Distributed architecture/replay gates | Eight-GPU production validation is a separate gate. |

## Algorithms and rewards

| Capability | Status |
|---|---|
| DanceGRPO / Flow-GRPO | Built-in Wan FSDP program capability. |
| BGPO / VIPO | First-class TeleBoost algorithms and public program identities. |
| TempFlow | First-class program because it changes rollout topology. |
| GRPO-Guard | Composable algorithm/config capability. |
| DPO | Wan over the TeleTron/Megatron engine. |
| Registry rewards | Aesthetic, HPS, random/debug, temporal quality, RAFT, VideoCLIP, and VideoPhy adapters as documented. |
| vLLM judge | Video-VLM adapter; Qwen3-VL real request smoke. |
| Joint reward | Multi-reward weighting and collectives implemented. |

Optional reward source, weights, and their license obligations are not supplied
by the root distribution.

## Runtime and kernel contract

| Component | Validated contract |
|---|---|
| Python / Torch | Python 3.11, PyTorch 2.9.1, CUDA 12.8 reference stack |
| verl | Exact 0.7.1 immutable source pin in `constraints/upstreams/verl.txt` |
| vLLM | 0.14.0 with a Qwen3-VL real video request smoke |
| SGLang | Dependency profile only; not runtime-certified |
| Wan attention | Hopper FA3, then FA2, then mask-correct SDPA; explicit selection fails loud |
| Wan FA3 | SM90, head dimension 128, FP16/BF16, forward/backward, varlen/local/GQA |

## Checkpoint and artifact commands

- `teleboost-convert-wan-to-teletron` converts Wan HF transformer weights to a
  TeleTron Megatron release checkpoint. Current output is TP=1.
## Release boundary

This public branch, its sdist, and its wheel are Wan-only. The root artifacts
exclude `third_party`, tests, diagnostics, model weights, datasets, outputs,
and checkpoints. External backend plugins may use the exact-name lazy
`teleboost.programs` entry-point contract, but are not built into this source
tree or covered by its validation claims.
The external backend entry-point registry is an extension boundary, not an
additional built-in model-family claim.

The reconciliation baselines in `THIRD_PARTY_PROVENANCE.md` still require
maintainer/legal approval before the first public source tag; mechanical gates
cannot make that authorship and redistribution decision.
