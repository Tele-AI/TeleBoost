#!/usr/bin/env bash
# BGPO recipe — Bayesian-Prior Group Optimization (arxiv 2511.18919).
#
# Adds CRT reward-rearrangement + RAS adaptive advantage scaling on top of GRPO.
# The algorithm math lives in teleboost/algorithms/bgpo.py; this recipe flips
# `algorithm.bgpo.enable` on top of the
# shared GRPO launcher.
#
# Usage: set your model/data env first (see the env preamble in recipes/wan_grpo_fsdp/run.sh), then run.
# Extra Hydra overrides pass through, e.g.:
#   bash recipes/wan_bgpo_fsdp/run.sh algorithm.bgpo.regularization_term_alpha=0.5
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../wan_grpo_fsdp/run.sh" \
  algorithm.bgpo.enable=true \
  "$@"
