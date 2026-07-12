#!/usr/bin/env bash
# VIPO recipe — Visual Preference Policy Optimization (arxiv 2511.18719).
#
# Enables the pixel-weighted dense advantage: a DINOv2-PCA allocation map M(p)
# turns the scalar GRPO advantage A into a dense A^p = M(p)·A. The algorithm
# math lives in teleboost/algorithms/vipo.py; this recipe just flips the
# `actor_rollout_ref.pixel_weight.enable` toggle on top of the shared GRPO
# launcher.
#
# Usage: set your model/data env first (see recipes/wan_grpo_fsdp/run.sh for the full
# WAN_*/N_GPUS/TRAIN_FILE preamble), then run this script. Extra Hydra overrides
# pass through, e.g.:
#   bash recipes/wan_vipo_fsdp/run.sh actor_rollout_ref.pixel_weight.sigma=1.5
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../wan_grpo_fsdp/run.sh" \
  actor_rollout_ref.pixel_weight.enable=true \
  "$@"
