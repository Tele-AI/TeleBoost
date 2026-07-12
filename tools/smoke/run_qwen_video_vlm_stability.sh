#!/usr/bin/env bash
set -euo pipefail

# Qwen video_vlm end-to-end training smoke/stability runner.
#
# All paths are env-driven; model/reward paths are REQUIRED (no machine-
# specific defaults).
#
# Common overrides:
#   STEPS=50 CUDA_VISIBLE_DEVICES=2 bash tools/smoke/run_qwen_video_vlm_stability.sh
#   TELEBOOST_ROOT=/path/to/TeleBoost TELEBOOST_VENV=/path/to/venv ...

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd "${script_dir}/../.." && pwd)"

export TELEBOOST_ROOT="${TELEBOOST_ROOT:-${default_root}}"
cd "${TELEBOOST_ROOT}"

if [[ -n "${TELEBOOST_VENV:-}" ]]; then
  export VIRTUAL_ENV="${TELEBOOST_VENV}"
  export PATH="${TELEBOOST_VENV}/bin:${PATH}"
fi

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export PYTHONPATH="${TELEBOOST_ROOT}:${TELEBOOST_ROOT}/third_party:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export TRAIN_FILE="${TRAIN_FILE:-${TELEBOOST_ROOT}/data/smoke_1_3B/processed_wan_prompt.json}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
: "${WAN_MODEL_PATH:?set WAN_MODEL_PATH to the Wan-T2V-1.3B HF dir}"
export WAN_MODEL_PATH
export WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_MODEL_PATH}/Wan2.1_VAE.pth}"
export REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-qwen}"
: "${REWARD_MODEL_PATH:?set REWARD_MODEL_PATH to the Qwen-VL HF dir}"
export REWARD_MODEL_PATH

export TELEBOOST_METHOD="${TELEBOOST_METHOD:-default}"
export PROJECT_NAME="${PROJECT_NAME:-TeleBoostStability}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export TOTAL_TRAINING_STEPS="${STEPS:-${TOTAL_TRAINING_STEPS:-10}}"
export SAMPLING_STEPS="${SAMPLING_STEPS:-4}"
export VIDEO_HEIGHT="${VIDEO_HEIGHT:-64}"
export VIDEO_WIDTH="${VIDEO_WIDTH:-64}"
export NUM_FRAMES="${NUM_FRAMES:-5}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export FSDP_OFFLOAD="${FSDP_OFFLOAD:-True}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-console}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}"
export QWEN_GMU="${QWEN_GMU:-0.15}"
export QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-2048}"
export QWEN_ALLOWED_LOCAL_MEDIA_PATH="${QWEN_ALLOWED_LOCAL_MEDIA_PATH:-/tmp}"
export ACTOR_LR="${ACTOR_LR:-1e-7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

: "${TRAIN_FILE:?}"
: "${TEST_FILE:?}"
: "${WAN_MODEL_PATH:?}"
: "${REWARD_MODEL_PATH:?}"

for required_path in "${TRAIN_FILE}" "${TEST_FILE}" "${WAN_MODEL_PATH}" "${WAN_VAE_PATH}" "${REWARD_MODEL_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[qwen-video-vlm] missing path: ${required_path}" >&2
    exit 2
  fi
done

dataset_len="$(
  python3 - "${TRAIN_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if hasattr(data, "__len__"):
    print(len(data))
else:
    raise SystemExit(f"dataset has no length: {path}")
PY
)"

if [[ "${dataset_len}" -le 0 ]]; then
  echo "[qwen-video-vlm] empty dataset: ${TRAIN_FILE}" >&2
  exit 2
fi

total_epochs="${TOTAL_EPOCHS:-$(( (TOTAL_TRAINING_STEPS + dataset_len - 1) / dataset_len ))}"
timestamp="$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_video_vlm_${TOTAL_TRAINING_STEPS}step_${timestamp}}"

if [[ "${RAY_STOP_BEFORE_RUN:-1}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

echo "[qwen-video-vlm] root=${TELEBOOST_ROOT}"
echo "[qwen-video-vlm] gpu=${CUDA_VISIBLE_DEVICES} steps=${TOTAL_TRAINING_STEPS} dataset_len=${dataset_len} epochs=${total_epochs}"
echo "[qwen-video-vlm] experiment=${EXPERIMENT_NAME}"

exec bash recipes/wan_grpo_fsdp/run.sh \
  data.prompt_key=caption \
  trainer.test_freq="${TEST_FREQ:-999}" \
  trainer.save_freq="${SAVE_FREQ:-999}" \
  trainer.total_epochs="${total_epochs}" \
  "$@"
