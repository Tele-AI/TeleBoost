# TempFlow-GRPO

Trainer: `trainer.py` in this directory composes the algorithm onto the
algorithm-agnostic base (`teleboost/training/core/trainer.py`); the backend
selects it automatically from the enable flag.

GRPO-family recipes (paper arxiv 2508.04324). Two separable, paper-faithful levers
that make GRPO **noise-aware** over the diffusion denoise schedule:

- **Noise-aware reweighting (Eq. 8):** scale each per-timestep clipped GRPO term
  by `Norm(σ_t·√Δt)` (mean-1 across steps), realigning optimisation pressure with
  each step's actual exploration capacity (high-noise early steps matter more
  than the near-deterministic late ones). Needs `sigma_form: flow_grpo` for the
  paper's `σ_t = a·√(t/(1−t))`.
- **Trajectory branching (Def. 1 / Thm. 1 / Eq. 3):** run the ODE deterministically
  to a branch step `k`, inject one SDE step, run the ODE to `x_0`, and score the
  final video — the terminal reward becomes the *process* reward for step `k`
  (Credit Localization), with a group-relative advantage computed **per branch
  point**. The rollout emits one DataProto row per branch.

- **Algorithm math:** [`noise_weight.py`](noise_weight.py),
  [`algorithm.py`](algorithm.py) + [`trainer.py`](trainer.py)
- **Config toggles:** `actor_rollout_ref.actor.tempflow.*` —
  `noise_reweight_mode`, `branch.enable`, `branch.branch_points`, `branch.early_k`,
  `branch.exploration_k` (see `teleboost/config/teleboost_trainer.yaml`). Both levers
  default **off** (≡ baseline GRPO bit-for-bit).
- **Base machinery:** reuses the shared Wan GRPO actor/rollout/trainer in
  `recipes/wan_grpo_fsdp/` and the system launcher `teleboost.programs.main`. The
  branched advantage is wired as a `TrajectoryBranchMixin` on the trainer, mirroring
  the bgpo/vipo pattern.

## Run

```bash
# set model/data env (see the env preamble in recipes/wan_grpo_fsdp/run.sh), then:
bash recipes/wan_tempflow_fsdp/run.sh
```

Both levers are on by default. Use one only — e.g. noise-reweight without
branching:

```bash
bash recipes/wan_tempflow_fsdp/run.sh actor_rollout_ref.actor.tempflow.branch.enable=false
```

Or tune the branch fan-out:

```bash
bash recipes/wan_tempflow_fsdp/run.sh \
  actor_rollout_ref.actor.tempflow.branch.early_k=4 \
  actor_rollout_ref.actor.tempflow.branch.exploration_k=4
```
