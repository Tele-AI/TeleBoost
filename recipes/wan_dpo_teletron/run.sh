#!/usr/bin/env bash
set -euo pipefail

# Public launcher for TeleBoost-DPO (verl-recipes path).
#
# Mirrors the env-var-driven shape of
# recipes/wan_grpo_fsdp/run.sh so DPO and GRPO share
# a single launch convention across this monorepo.
#
# === Required (every mode) ===
#   WAN_DIT_CKPT=/path/to/<Wan-DiT>-teletron
#       Pre-converted teletron-format Wan DiT checkpoint
#       (run ``teleboost-convert-wan-to-teletron`` once offline).
#       real-train default → Wan2.1-I2V-14B-480P-teletron.
#       recipe-smoke / phase3-replay → Wan2.1-T2V-1.3B-teletron.
#   WAN_HF_DIR=/path/to/<Wan-upstream-dir>
#       Upstream HF Wan ckpt dir — VAE + T5 paths derive from it.
#
# === real-train extra required ===
#   WAN_DPO_DATA_DIR=/path/to/dpo_csv
#       Directory holding the 8-shard preference-pair CSVs
#       (``prompt_video_pairs_enhanced_part{0..7}.csv``). Read by
#       ``teleboost.programs.wan.dpo.wan_dpo_i2v`` at import time.
#   WAN_T2V_1_3B_DIR=/path/to/Wan2.1-T2V-1.3B
#       Wan2.1-T2V-1.3B upstream release dir — VAE + T5 weight paths
#       derive from it.
#   WAN_I2V_14B_DIR=/path/to/Wan2.1-I2V-14B-480P
#       Upstream 14B I2V dir — T5 tokenizer + CLIP image encoder
#       paths derive from it.
#
# === Mode dispatch ===
#   TELEBOOST_DPO_MODE = real-train | recipe-smoke | phase3-replay
#       Default: real-train (8-GPU 2-VAE+6-DiT preference-pair training).
#       recipe-smoke is a 4-GPU stub-batch smoke (~3 min wall); use
#       phase3-replay for precision compare vs reference dumps.

: "${WAN_DIT_CKPT:?Set WAN_DIT_CKPT=/path/to/<Wan-DiT>-teletron (pre-converted)}"
: "${WAN_HF_DIR:?Set WAN_HF_DIR=/path/to/<Wan-upstream-dir>}"
: "${MEGATRON_LM_DIR:?Set MEGATRON_LM_DIR=/path/to/Megatron-LM (core_v0.16 checkout)}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

require_directory() {
    local name="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "[run_teleboost_dpo] ERROR: ${name} is not an existing directory: ${path}" >&2
        exit 1
    fi
}

require_directory "WAN_DIT_CKPT" "${WAN_DIT_CKPT}"
require_directory "WAN_HF_DIR" "${WAN_HF_DIR}"
require_directory "MEGATRON_LM_DIR" "${MEGATRON_LM_DIR}"

mode="${TELEBOOST_DPO_MODE:-real-train}"
phase3_dumps="${PHASE3_DUMP_DIR:-/tmp/phase3_dumps}"
noise_seed="${NOISE_SEED:-42}"

export WAN13B_DIR="${WAN_HF_DIR}"

# PYTHONPATH must include the project root so ``teleboost.programs.wan.dpo.*`` imports work.
# The upstream Wan runtime itself must already be installed/importable.
export PYTHONPATH="${project_root}:${MEGATRON_LM_DIR}:${PYTHONPATH:-}"
python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[run_teleboost_dpo] ERROR: PYTHON_BIN is not executable or not on PATH: ${python_bin}" >&2
    exit 1
fi

if ! "${python_bin}" - <<'PY'
import sys

try:
    import wan  # noqa: F401
except Exception as exc:
    print(
        "[run_teleboost_dpo] ERROR: the upstream Wan runtime is not importable "
        "as the top-level 'wan' package.",
        file=sys.stderr,
    )
    print(
        "Install Wan or add its installation root to PYTHONPATH before "
        "invoking this launcher.",
        file=sys.stderr,
    )
    print(f"Original import error: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
PY
then
    exit 1
fi

# The pip megatron-core distribution does not contain megatron.training.
# Pin and verify the separate source checkout before Ray starts any workers.
required_megatron="${project_root}/constraints/upstreams/megatron-lm.txt"
expected_megatron_commit="$(sed -n 's/^UPSTREAM_COMMIT=//p' "${required_megatron}")"
if current_megatron_commit="$(git -C "${MEGATRON_LM_DIR}" rev-parse HEAD 2>/dev/null)"; then
    if [[ "${current_megatron_commit}" != "${expected_megatron_commit}" ]]; then
        echo "[run_teleboost_dpo] ERROR: MEGATRON_LM_DIR commit ${current_megatron_commit} != required ${expected_megatron_commit}" >&2
        exit 1
    fi
elif [[ "${TELEBOOST_ALLOW_UNVERIFIED_MEGATRON_SOURCE:-0}" != "1" ]]; then
    echo "[run_teleboost_dpo] ERROR: cannot verify MEGATRON_LM_DIR revision (missing Git metadata)." >&2
    echo "Set TELEBOOST_ALLOW_UNVERIFIED_MEGATRON_SOURCE=1 only for a trusted rsync of ${expected_megatron_commit}." >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path

import megatron.training

expected = {"deepspeed": "0.18.6", "megatron-core": "0.16.1"}
mismatched = {
    name: (version(name), required)
    for name, required in expected.items()
    if version(name) != required
}
if mismatched:
    raise SystemExit(f"DPO dependency version mismatch: {mismatched}")

source_root = Path(os.environ["MEGATRON_LM_DIR"]).resolve()
loaded = Path(megatron.training.__file__).resolve()
if not loaded.is_relative_to(source_root):
    raise SystemExit(
        f"megatron.training loaded from {loaded}, outside MEGATRON_LM_DIR={source_root}"
    )
PY

# Helpful diagnostics.
echo "[run_teleboost_dpo] mode=${mode}"
echo "  project_root=${project_root}"
echo "  WAN_DIT_CKPT=${WAN_DIT_CKPT}"
echo "  WAN_HF_DIR=${WAN_HF_DIR}"
echo "  MEGATRON_LM_DIR=${MEGATRON_LM_DIR}"
echo "  PHASE3_DUMP_DIR=${phase3_dumps}"

case "${mode}" in
recipe-smoke)
    # 4-GPU stub-batch smoke through the verl-recipes path. Verifies the
    # full init chain (mesh + Wan load + DeepSpeed ZeRO + lr_scheduler
    # + ckpt manager) and the split-DPO multi-backward fires.
    # Overrides yaml production defaults back to a smoke profile:
    # distributed_vae=false, single 4-GPU world, model-arch-only
    # config_path (smoke_train_step synthesizes its batch inline).
    n_gpus="${N_PROC:-4}"
    cd "${project_root}"
    "${python_bin}" -m teleboost.programs.wan.dpo.main \
        trainer.n_gpus_per_node="${n_gpus}" \
        trainer.nnodes=1 \
        +trainer.run_smoke_train_step=true \
        teletron_args.config_path=teleboost.programs.wan.dpo.wan_t2v_arch.config \
        teletron_args.distributed_vae=false \
        teletron_args.distributed_vae_world_size=0 \
        teletron_args.global_batch_size="${n_gpus}" \
        teletron_args.load="${WAN_DIT_CKPT}" \
        "$@"
    ;;

phase3-replay)
    # Replay reference dumps through the verl-recipes model.forward
    # and compare per-pair noise_pred + loss. Each DiT rank in the
    # baseline run wrote one preference pair; replay covers all of them.
    if [ ! -d "${phase3_dumps}" ] || [ -z "$(ls -A "${phase3_dumps}" 2>/dev/null || true)" ]; then
        echo "ERROR: no dumps under ${phase3_dumps}. Point PHASE3_DUMP_DIR at a directory of baseline dumps." >&2
        exit 2
    fi
    n_gpus="${N_PROC:-4}"
    cd "${project_root}"
    "${python_bin}" -m teleboost.programs.wan.dpo.main \
        trainer.n_gpus_per_node="${n_gpus}" \
        trainer.nnodes=1 \
        +trainer.phase3_replay_dir="${phase3_dumps}" \
        teletron_args.config_path=teleboost.programs.wan.dpo.wan_t2v_arch.config \
        teletron_args.distributed_vae=false \
        teletron_args.distributed_vae_world_size=0 \
        teletron_args.global_batch_size="${n_gpus}" \
        teletron_args.load="${WAN_DIT_CKPT}" \
        "$@"
    ;;

real-train)
    # Real preference-pair data flow. 8-GPU 2-VAE + 6-DiT: VAE ranks run
    # DistDataProducer background thread; DiT ranks build
    # DistVAEConsumerBatchLoader + run engine.train_batch via the split-DPO
    # ZeRO path. ``teleboost.programs.wan.dpo.wan_dpo_i2v.config`` (yaml default) reads
    # WAN_DPO_DATA_DIR and WAN_I2V_14B_DIR at module import; enforce
    # both here so the failure is fast and named.
    : "${WAN_DPO_DATA_DIR:?Set WAN_DPO_DATA_DIR=/path/to/dpo_csv (preference-pair CSV shards)}"
    : "${WAN_T2V_1_3B_DIR:?Set WAN_T2V_1_3B_DIR=/path/to/Wan2.1-T2V-1.3B (VAE + T5)}"
    : "${WAN_I2V_14B_DIR:?Set WAN_I2V_14B_DIR=/path/to/Wan2.1-I2V-14B-480P}"
    require_directory "WAN_DPO_DATA_DIR" "${WAN_DPO_DATA_DIR}"
    require_directory "WAN_T2V_1_3B_DIR" "${WAN_T2V_1_3B_DIR}"
    require_directory "WAN_I2V_14B_DIR" "${WAN_I2V_14B_DIR}"
    export WAN_DPO_DATA_DIR WAN_T2V_1_3B_DIR WAN_I2V_14B_DIR
    # global_batch_size must be divisible by micro_batch_size *
    # data_parallel_size, where DP = n_gpus - n_vae. Default 6 matches
    # the canonical 8-GPU/2-VAE/6-DiT layout (micro=1, dp=6, microbatches=1).
    n_gpus="${N_PROC:-8}"
    n_vae="${N_VAE:-2}"
    real_iters="${REAL_TRAIN_ITERS:-1}"
    real_gbs="${REAL_GBS:-$(( n_gpus - n_vae ))}"
    cd "${project_root}"
    # trainer.total_training_steps drives the loop; optim picks up
    # the same value via the yaml interpolation
    # ``optim.total_training_steps: ${trainer.total_training_steps}``,
    # so the lr scheduler's lr_decay_steps stays in sync without a
    # double-set here.
    "${python_bin}" -m teleboost.programs.wan.dpo.main \
        trainer.n_gpus_per_node="${n_gpus}" \
        trainer.nnodes=1 \
        trainer.total_training_steps="${real_iters}" \
        teletron_args.distributed_vae_world_size="${n_vae}" \
        teletron_args.global_batch_size="${real_gbs}" \
        teletron_args.load="${WAN_DIT_CKPT}" \
        "$@"
    ;;

*)
    echo "Unknown TELEBOOST_DPO_MODE='${mode}'. Use recipe-smoke | phase3-replay | real-train." >&2
    exit 2
    ;;
esac
