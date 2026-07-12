# TeleBoost Testing

## Test profiles

The suite is split by executable environment.  Missing required dependencies
must not be counted as a passing test through broad `importorskip` use.

- Core (default): `pytest --profile=core`. Pure CPU/PyTorch logic and dependency-light
  contracts.  Modules that import the training runtime at collection time are
  excluded before import.
- Training integration: `pytest --profile=training`. This
  first checks for the pinned `verl`, TensorDict, Ray, datasets, diffusers,
  PEFT, and Hydra stack.  A missing dependency is a hard usage error, not a
  skipped test.
- Heavy GPU uses the Wan capability lane:
  `pytest --profile=heavy --heavy-lane=wan`. It performs GPU and dependency
  preflight before collection, so a completely skipped lane cannot report
  green. The lane uses compact/random weights around the real architecture
  code.

Real Qwen3-VL and Wan checkpoints are intentionally outside all pytest
profiles. Their Git/source-checkout-only standalone tools (excluded from the
sdist and wheel) default to a no-model-load preflight and require an explicitly
owned idle GPU for `--run`:

```bash
PYTHON=python  # run from the active TeleBoost environment
GPU=7         # choose/coordinate an idle GPU
QWEN_MODEL=/path/to/Qwen3-VL
WAN_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers

"$PYTHON" -B tools/smoke/qwen3vl_vllm_real_smoke.py \
  --preflight --model-path "$QWEN_MODEL" --gpu "$GPU"
"$PYTHON" -B tools/smoke/wan_diffusers_real_smoke.py \
  --preflight --model-path "$WAN_MODEL" --gpu "$GPU"

"$PYTHON" -B tools/smoke/qwen3vl_vllm_real_smoke.py \
  --run --model-path "$QWEN_MODEL" --gpu "$GPU"
"$PYTHON" -B tools/smoke/wan_diffusers_real_smoke.py \
  --run --model-path "$WAN_MODEL" --gpu "$GPU"
"$PYTHON" -B tools/smoke/wan_diffusers_real_smoke.py \
  --run --decode --model-path "$WAN_MODEL" --gpu "$GPU"
```

The Qwen run is one in-memory video request through vLLM 0.14 and the
production reward parser. The default Wan run checks a one-step latent; the
last command additionally checks the VAE decode. Neither command saves media,
touches `outputs`, starts Ray, or counts toward a pytest profile result. Model
paths can be overridden with `--model-path`,
`TELEBOOST_QWEN3VL_MODEL_PATH`, or
`TELEBOOST_WAN_DIFFUSERS_MODEL_PATH`.

The Wan lane leaves Megatron-Core's missing Apex/Transformer-Engine
`UserWarning` messages visible: they identify an environment running the
Torch fallback rather than the compiled production path. Pytest filters only
two exact Megatron-Core 0.16.1 compatibility-reexport deprecations; there is no
broad third-party `DeprecationWarning` suppression.

`tests/special_distributed/test_cp_grad_reduce.py` is a standalone torchrun
program rather than a pytest test. Run it with torchrun:

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/special_distributed/test_cp_grad_reduce.py
```

Two additional standalone torchrun programs are:

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/unit_tests/teletron/context_parallel/test_context_parallel_mixin_stateless.py
torchrun --standalone --nproc_per_node=8 \
  tests/unit_tests/teletron/context_parallel/test_forward_attn_precision.py
```

All three programs are excluded from normal pytest discovery. Explicitly
passing any of them to pytest is a usage error, so dependency or rendezvous
failures cannot be misreported as skipped pytest tests.

When reporting results, state the profile.  “Core passed” does not claim that
training or real-checkpoint profiles were executed.

## Testing distributed methods

Many TeleBoost code paths (split, gather, all-to-all, ...) only get meaningful
coverage in a distributed setting, so the test case itself must run multiple
processes. The suite's `spawn` helper launches those workers, drains their
result pipe while they run, prints child tracebacks, and turns every non-zero
child exit into a parent-side test failure that names the failed rank. The
concurrent drain also avoids deadlocking when a child returns a large NumPy or
tensor payload.

Interface:

```
spawn(
    nprocs: int,
    func: Callable,
    *args,
    timeout_seconds: float | None = None,
) -> queue.Queue
```

`nprocs` is the process count, `func` is what each process runs, and `args`
are its extra arguments. `spawn` passes `func` the rank and world size as the
first and second positional arguments and a `multiprocessing.Queue` object
`q` as the third; `args` (if any) follow from the fourth position on. The
whole process group has a 300-second timeout by default. Pass
`timeout_seconds=` for one call or set
`TELEBOOST_TEST_PROCESS_TIMEOUT_SECONDS` for the test environment.

The child queue is the result bridge: inside `func`, report values with
`q.put()`. After every child exits successfully, `spawn` returns a fully
materialized in-memory `queue.Queue`, so the parent can consume it with
deterministic `get()`/`empty()` behavior. Uncaught child assertions and other
exceptions already fail `spawn`; queue messages should carry test results or
diagnostics, not encode a hidden skip or substitute for an exception.
`tests/unit_tests/teletron/test_parallel_state.py` is a minimal example of a
child-process test built on this mechanism — one read and it should be clear.
In a real test you will usually also need to initialize the PyTorch process
group (`init_process_group`) and the megatron parallel groups
(`initialize_model_parallel`); use the rank / world-size arguments that were
passed in.
`tests/unit_tests/teletron/context_parallel/test_context_parallel_mixin.py`
shows how.

Note: bare `print` output is swallowed by pytest — log debug information via
`logging.info(...)` instead, and run pytest with
`-o log_cli=true -o log_cli_level=INFO` so the log output reaches the screen.
