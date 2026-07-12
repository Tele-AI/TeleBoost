# BGPO — Bayesian-Prior Group Optimization

Trainer: `trainer.py` in this directory composes the algorithm onto the
algorithm-agnostic base (`teleboost/training/core/trainer.py`); the backend
selects it automatically from the enable flag.

GRPO-family recipes (arxiv 2511.18919). Two levers on top of GRPO, both
anchored on a per-prompt `prior`:

- **CRT** (`use_rerange`) — reward rearrangement. Rewrites each group's
  rewards around the prior,
  `R̃ = [a·(R − prior) + 𝟙{R > prior}] / (1 + exp(−R/τ)) · R`
  (`rerange_a`, `rerange_temperature`), overwriting them in place.
- **RAS** (`adaptive_weight_method=bayes`) — adaptive advantage scaling:
  `advantage *= clamp(1 + α·w, …)` with a Bayesian reliability weight `w`
  from the group's posterior vs the prior (`prior_var`, `bayes_weight_range`).

When `algorithm.bgpo.enable=false` it is a no-op and the trainer runs vanilla GRPO.

- **Algorithm math:** [`algorithm.py`](algorithm.py) (`binary_rerange_group_rewards` + `bayes_reliability_weight`) + [`trainer.py`](trainer.py) (`BGPOMixin` + `RayBGPOTrainer`)
- **Config toggle:** `algorithm.bgpo.enable=true` (`use_rerange`,
  `adaptive_weight_method`, `regularization_term_alpha`, … under the same
  block — see `teleboost/config/teleboost_trainer.yaml`)
- **Base machinery:** reuses the shared Wan GRPO actor/worker/trainer in
  `recipes/wan_grpo_fsdp/` and the system launcher `teleboost.programs.main`.

## Dataset — the `prior` field

BGPO requires a per-prompt numeric `prior` (`R_prior`) — the reward a
semantically clear, typical prompt is expected to earn. Both branches
measure against it, so add one field per training record:

```json
{ "caption": "...", "prior": 0.65, "context_path": "...", "context_null_path": "..." }
```

If a record has no `prior`, BGPO logs a warning and leaves rewards and
advantages unchanged (a no-op for that batch). Set it from the SFT model's
own reward per prompt (T2V), the first-frame text-alignment (I2V), or a
running mean of group rewards (T2I).

## Run

```bash
# set model/data env (see the env preamble in recipes/wan_grpo_fsdp/run.sh), then:
bash recipes/wan_bgpo_fsdp/run.sh
```
