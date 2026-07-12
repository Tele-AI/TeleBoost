# Supported algorithm helpers

This directory is the home for pure algorithm math and small cross-cutting
guards. It is not a trainer or worker implementation directory.

## Current placement

| name | module | public switch | runtime wiring |
|---|---|---|---|
| GRPO base | `grpo/loss.py`, `grpo/advantage.py`, `grpo/sigma_schedule.py` | `algorithm.adv_estimator=grpo` | `teleboost.training.families.wan.actor` |
| BGPO | `bgpo.py` | `algorithm.bgpo.enable=true` | `teleboost.programs.wan.bgpo` |
| VIPO | `vipo.py` | `actor_rollout_ref.pixel_weight.enable=true` | `teleboost.programs.wan.vipo` plus Wan rollout pixel weights |
| TempFlow noise | `tempflow_noise.py` | `actor_rollout_ref.actor.tempflow.noise_reweight_mode=tempflow_noise_norm` | Wan actor loss body |
| TempFlow branching | `tempflow.py` | `actor_rollout_ref.actor.tempflow.branch.enable=true` | `teleboost.programs.wan.tempflow` plus rollout branch driver |
| GRPO-Guard | `grpo_guard.py` | `actor_rollout_ref.actor.grpo_guard.enable=true` | Wan actor loss body |

## Layering rules

- Pure math and validation helpers live here.
- Driver-phase trainer mixins live in `teleboost.programs.wan` for Wan-specific
  programs; they compose with `teleboost.training.families.wan.RayWanTrainer`.
- Actor loss-inner logic stays in the concrete program actor when it must run
  inside the denoise loop.
- Rollout topology changes stay in rollout/program wiring, not in this
  directory.
- `recipes/` contains launch scripts, config, and README files only.

## Cross-cutting contracts

These modules are safety scaffolding, not user-facing algorithms:

- `policy_scalars.py` guards per-sample scalar shape/finite boundaries before
  quantities enter GRPO loss math.
- `rollout_contract.py` defines rollout metadata contracts such as
  `VALID_LOGPROB_REDUCTIONS`.
- `solver_contract.py` verifies rollout solver and actor recompute consistency.

## Adding a new algorithm helper

1. Add pure compute to `teleboost/algorithms/<name>.py` when it is reusable or
   independently testable.
2. Add program-specific trainer/actor/rollout wiring under
   `teleboost/programs/<family>/` only when the algorithm needs runtime hooks.
3. Register program/backend selection in the relevant program backend.
4. Expose launcher/config knobs from `recipes/` without adding Python there.
5. Add unit tests for the pure math and one integration test for the program
   wiring.
