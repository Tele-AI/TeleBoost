#!/usr/bin/env bash
set -euo pipefail

# Environment-driven 4-GPU Wan GRPO smoke. No host-specific paths or model
# locations are embedded in this script.
#
# Required:
#   WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B
#   TRAIN_FILE=/path/to/processed_wan_prompt.json
#
# Example:
#   WAN_MODEL_PATH=/models/Wan2.1-T2V-1.3B \
#   TRAIN_FILE=/datasets/wan/smoke.json \
#   bash tools/smoke/run_wan_grpo_4gpu.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
cd "${project_root}"

if [[ -n "${TELEBOOST_VENV:-}" ]]; then
  export VIRTUAL_ENV="${TELEBOOST_VENV}"
  export PATH="${TELEBOOST_VENV}/bin:${PATH}"
fi

: "${WAN_MODEL_PATH:?set WAN_MODEL_PATH to the Wan model directory}"
: "${TRAIN_FILE:?set TRAIN_FILE to a processed Wan prompt dataset}"

export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export WAN_VERSION="${WAN_VERSION:-wan21}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_MODEL_PATH}/Wan2.1_VAE.pth}"
export REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-random}"
# The random debug provider does not read model weights, but the public
# launcher deliberately requires a non-empty reward path for every provider.
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${WAN_MODEL_PATH}}"

export NNODES="${NNODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export SP_SIZE="${SP_SIZE:-1}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-4}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-4}"
export VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
export VIDEO_WIDTH="${VIDEO_WIDTH:-832}"
export NUM_FRAMES="${NUM_FRAMES:-49}"
export SAMPLING_STEPS="${SAMPLING_STEPS:-16}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FSDP_OFFLOAD="${FSDP_OFFLOAD:-False}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-console}"
export PROJECT_NAME="${PROJECT_NAME:-TeleBoostSmoke}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-wan_grpo_4gpu_smoke}"
export TELEBOOST_OUTPUT_DIR="${TELEBOOST_OUTPUT_DIR:-${project_root}/outputs}"

if [[ "${RAY_STOP_BEFORE_RUN:-1}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

exec bash recipes/wan_grpo_fsdp/run.sh \
  "data.prompt_key=${PROMPT_KEY:-caption}" \
  "trainer.save_freq=${SAVE_FREQ:-999}" \
  "trainer.test_freq=${TEST_FREQ:-999}" \
  "$@"
