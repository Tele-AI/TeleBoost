# Installation

TeleBoost is validated on a CUDA 12.8 / PyTorch 2.9 stack. Use a dedicated
environment; do not mutate a cluster base environment used by active jobs.

The dependency contract has three layers:

- `pyproject.toml` declares functional dependency groups.
- `constraints/torch2.9-cu128.txt` pins the package versions tested together.
- `constraints/upstreams/*.txt` pins source dependencies that pip must not
  resolve or update implicitly.

`requirements.txt` is only a compatibility entry point joining the standard
groups and constraints.

## 1. Install the CUDA-matched PyTorch stack first

```bash
conda create -n teleboost python=3.11 -y
conda activate teleboost
python -m pip install -c constraints/release.txt setuptools wheel
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

If the cluster supplies PyTorch, activate that environment and verify the
versions instead of reinstalling it.

## 2. Install the pinned verl API

```bash
bash tools/install_verl.sh --show
bash tools/install_verl.sh
```

The installer uses `--no-deps` intentionally. TeleBoost targets exactly the
version and immutable revision in `constraints/upstreams/verl.txt`.

## 3. Install TeleBoost

```bash
python -m pip install -r requirements.txt
```

This installs the editable checkout with `train`, `wan`, `dpo`, and `reward`
groups under the tested constraints. The root package does not bundle the Wan
upstream runtime. Make a compatible top-level `wan` package importable:

```bash
export WAN_RUNTIME_ROOT=/path/to/Wan-Video
export PYTHONPATH="${WAN_RUNTIME_ROOT}:${PYTHONPATH:-}"
python -c 'import wan; print(wan.__file__)'
```

For development, `WAN_RUNTIME_ROOT` may point at this checkout's
`third_party/` directory. Launchers never add it implicitly. The Wan video
reader uses `decord2` (import-compatible as `decord`) because it provides
modern CPython wheels.

## 4. Optional GPU and inference profiles

### Wan attention and context parallelism

Wan DiT, Ulysses, and TeleTron attention use
`auto = FA3 -> FA2 -> mask-correct SDPA`. Calls with attention dropout select
FA2 when available; explicit backends fail when they cannot implement the
requested call. Override with
`TELEBOOST_WAN_ATTN_BACKEND=flash_attn_3|flash_attn_2|sdpa|auto`.

Install FA2 against the already installed PyTorch/CUDA stack:

```bash
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
MAX_JOBS=8 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}" \
  python -m pip install -c constraints/torch2.9-cu128.txt \
  --no-build-isolation -e '.[flash]'
```

Hopper FA3 is pinned separately in
`constraints/upstreams/flash-attn-3.txt`. Its installer fetches the required
CUTLASS submodule, builds outside the checkout, and installs with `--no-deps`:

```bash
bash tools/install_flash_attn_3.sh --show
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" MAX_JOBS=4 NVCC_THREADS=2 \
  bash tools/install_flash_attn_3.sh
CUDA_VISIBLE_DEVICES=0 python tools/smoke/wan_flash_attention.py
```

Context parallelism also uses Megatron-Core 0.16.1. NVIDIA Apex and
Transformer Engine are optional accelerators, not correctness requirements.

### vLLM or SGLang rewards

Install one inference profile, not both:

```bash
# TeleBoost-tested profile.
python -m pip install -c constraints/torch2.9-cu128.txt -e '.[vllm]'

# Compatibility profile; not runtime-certified by the reference gate.
python -m pip install -c constraints/torch2.9-cu128.txt -e '.[sglang]'
```

vLLM 0.14.0 is an intentional TeleBoost override over verl 0.7.1's older
declared range. Re-run training and reward integration tests after changing
Torch, TensorDict, Ray, Transformers, diffusers, or the inference runtime.

### DPO / Megatron-LM

The `dpo` extra installs Megatron-Core 0.16.1 and DeepSpeed. The full
Megatron-LM source tree is also required because the core wheel omits
`megatron.training`:

```bash
git clone --filter=blob:none \
  https://github.com/NVIDIA/Megatron-LM.git /path/to/Megatron-LM
git -C /path/to/Megatron-LM checkout \
  55ac7082517c3878ae653c07c09c534b8aed49f6
export MEGATRON_LM_DIR=/path/to/Megatron-LM
PYTHONPATH="$MEGATRON_LM_DIR:${PYTHONPATH:-}" \
  python -c 'import megatron.training; print(megatron.training.__file__)'
```

The launcher verifies this revision and module path. A trusted rsync without
Git metadata requires the explicit
`TELEBOOST_ALLOW_UNVERIFIED_MEGATRON_SOURCE=1` opt-out.

### Optional reward providers

HPSv2 is best installed with `python -m pip install --no-deps hpsv2` because
its historical dependency pins conflict with the reference runtime. RAFT,
VideoCLIP, and VideoPhy are integration points and require user-supplied source
and checkpoints under their own licenses. See `MODEL_AND_DATA_LICENSES.md`.

## 5. Validate without starting training

```bash
python -m pip install -c constraints/dev.txt -e '.[test]'
python -m pip check
python -m compileall -q teleboost tools
pytest --profile=core
pytest --profile=training
pytest --profile=heavy --heavy-lane=wan
```

The heavy lane checks its GPU/runtime prerequisites before collection and uses
compact/random weights around the real architecture. It is not a production
checkpoint certification.

Standalone tools provide bounded real-checkpoint checks without starting a
training job:

```bash
python -B tools/smoke/qwen3vl_vllm_real_smoke.py \
  --preflight --model-path /path/to/Qwen3-VL --gpu 0
python -B tools/smoke/wan_diffusers_real_smoke.py \
  --preflight --model-path /path/to/Wan --gpu 0
```

Use `--run` only on an explicitly owned idle GPU. Add `--decode` to the Wan
tool to include VAE decode.

## 6. Validate release artifacts

The public source is already Wan-only. The release gate validates the checkout
before copying it and never rewrites product code or documentation:

```bash
python -m pip install -c constraints/release.txt -e '.[release]'
python tools/release/build_artifacts.py \
  --out-dir /tmp/teleboost-release-wan
```

It builds the wheel only from a safely extracted sdist, runs
`tools/release/check_wheel_contents.py`, strict Twine validation, and an
isolated install/CLI smoke, then writes `SHA256SUMS`. The output directory must
be empty and outside the checkout.

See `THIRD_PARTY_PROVENANCE.md`, `MODEL_AND_DATA_LICENSES.md`, and
`SECURITY.md` for redistribution and trusted-input boundaries.
