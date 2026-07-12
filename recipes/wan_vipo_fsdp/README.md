# VIPO — Visual Preference Policy Optimization

Trainer: `trainer.py` in this directory composes the algorithm onto the
algorithm-agnostic base (`teleboost/training/core/trainer.py`); the backend
selects it automatically from the enable flag.

GRPO-family recipes (arxiv 2511.18719). Replaces the scalar advantage with a
**pixel-weighted dense advantage**: a DINOv2-PCA allocation map `M(p)` scales the
per-prompt GRPO advantage `A` into `A^p = M(p)·A`, focusing the policy gradient
on the visually salient regions.

- **Algorithm math:** [`algorithm.py`](algorithm.py) (math) + [`trainer.py`](trainer.py) (`RayVIPOTrainer`)
- **Config toggle:** `actor_rollout_ref.pixel_weight.enable=true`
  (plus `model_path` / `pca_method` / `sigma` under the same block —
  see `teleboost/config/teleboost_trainer.yaml`)
- **Base machinery:** reuses the shared Wan GRPO actor/worker/trainer in
  `recipes/wan_grpo_fsdp/` and the system launcher `teleboost.programs.main`.

## Run

```bash
# set model/data env (see the env preamble in recipes/wan_grpo_fsdp/run.sh), then:
bash recipes/wan_vipo_fsdp/run.sh
```

Extra Hydra overrides pass through, e.g. `actor_rollout_ref.pixel_weight.sigma=1.5`.
