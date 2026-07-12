#!/usr/bin/env bash
# TempFlow-GRPO recipe (paper arxiv 2508.04324).
#
# Two separable, paper-faithful levers (both enabled here; override to use one):
#   * noise_reweight — per-step Eq. 8 weight Norm(σ_t·√Δt), mean-1 across the
#     denoise steps, realigning optimisation pressure with each step's actual
#     exploration capacity. Needs sigma_form=flow_grpo for the paper's σ_t.
#   * branch — trajectory branching (Def. 1 / Thm. 1 / Eq. 3): ODE → one SDE step
#     at the branch step k → ODE to x_0; the terminal reward becomes the process
#     reward for step k, with a group-relative advantage per branch point (the
#     rollout emits one DataProto row per branch).
# The math lives in teleboost/algorithms/{noise_weight,trajectory_branch}.py;
# this recipe flips the actor_rollout_ref.actor.tempflow.* toggles on top of the
# shared GRPO launcher.
#
# Usage: set your model/data env first (see recipes/wan_grpo_fsdp/run.sh for the full
# WAN_*/N_GPUS/TRAIN_FILE preamble), then run. Extra Hydra overrides pass
# through — e.g. noise-reweight only (no branching):
#   bash recipes/wan_tempflow_fsdp/run.sh actor_rollout_ref.actor.tempflow.branch.enable=false
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../wan_grpo_fsdp/run.sh" \
  actor_rollout_ref.actor.sigma_form=flow_grpo \
  actor_rollout_ref.actor.tempflow.noise_reweight_mode=tempflow_noise_norm \
  actor_rollout_ref.actor.tempflow.branch.enable=true \
  actor_rollout_ref.actor.tempflow.branch.branch_points=early_k \
  actor_rollout_ref.actor.tempflow.branch.early_k=2 \
  actor_rollout_ref.actor.tempflow.branch.exploration_k=2 \
  "$@"
