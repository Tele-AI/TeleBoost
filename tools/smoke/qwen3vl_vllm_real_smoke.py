#!/usr/bin/env python3
# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Safe one-GPU Qwen3-VL/vLLM 0.14 real-checkpoint smoke.

The default action is preflight only.  ``--run`` is required to load the
model.  The tool never starts Ray or training, never writes media/checkpoints,
and redirects mutable caches to a unique directory below ``/tmp``.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# A direct script is not itself byte-compiled.  Set this before loading the
# local helper (and later TeleBoost/model modules) so the smoke never creates
# ``__pycache__`` inside the checkout even when the caller omits ``python -B``.
sys.dont_write_bytecode = True
_COMMON_MODULE = f"{__package__}._real_model_smoke_common" if __package__ else "_real_model_smoke_common"
_common = importlib.import_module(_COMMON_MODULE)
SmokePreflightError = _common.SmokePreflightError
assert_gpu_idle = _common.assert_gpu_idle
isolated_cache_environment = _common.isolated_cache_environment
print_result = _common.print_result
query_gpu_status = _common.query_gpu_status
read_json = _common.read_json
require_files = _common.require_files
require_versions = _common.require_versions
require_weight_index = _common.require_weight_index


DEFAULT_MODEL_PATH = ""
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "torch": "2.9.1",
    "transformers": "4.57.6",
    "vllm": "0.14.0",
}
EXPECTED_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_vl"
EXPECTED_VIDEO_SHAPE = (4, 64, 64, 3)
PRODUCTION_DIMENSIONS = (
    "dim1_aesthetics",
    "dim2_distortion",
    "dim3_artifacts",
    "dim4_sharpness",
    "dim5_consistency",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = REPO_ROOT / "teleboost" / "patches" / "vllm"


def _environment_model_path() -> str:
    return os.environ.get(
        "TELEBOOST_QWEN3VL_MODEL_PATH",
        os.environ.get("REWARD_MODEL_PATH", DEFAULT_MODEL_PATH),
    )


def _environment_gpu() -> int | None:
    value = os.environ.get("TELEBOOST_SMOKE_GPU")
    if value is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.isdigit():
            value = visible
    if value is None:
        return None
    try:
        gpu = int(value)
    except ValueError as exc:
        raise SmokePreflightError(f"invalid TELEBOOST_SMOKE_GPU: {value!r}") from exc
    if gpu < 0:
        raise SmokePreflightError("GPU index must be non-negative")
    return gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        dest="mode",
        action="store_const",
        const="preflight",
        help="validate local assets, exact package versions, and optional GPU idleness",
    )
    mode.add_argument(
        "--run",
        dest="mode",
        action="store_const",
        const="run",
        help="run one real offline vLLM video-judge request (requires --gpu)",
    )
    parser.set_defaults(mode="preflight")
    parser.add_argument(
        "--model-path",
        default=_environment_model_path(),
        help=("local Qwen3-VL directory (env TELEBOOST_QWEN3VL_MODEL_PATH or REWARD_MODEL_PATH)"),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=_environment_gpu(),
        help="physical GPU index; --run requires an idle GPU",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="vLLM reservation fraction on the selected idle GPU (default: 0.35)",
    )
    return parser


def inspect_model(model_path: str | Path) -> dict[str, Any]:
    if not str(model_path).strip():
        raise SmokePreflightError("set --model-path, TELEBOOST_QWEN3VL_MODEL_PATH, or REWARD_MODEL_PATH")
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise SmokePreflightError(f"Qwen3-VL model directory does not exist: {root}")
    require_files(
        root,
        (
            "chat_template.jinja",
            "config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
        ),
    )
    config = read_json(root / "config.json")
    architectures = config.get("architectures")
    if architectures != [EXPECTED_ARCHITECTURE]:
        raise SmokePreflightError(f"unexpected Qwen architecture {architectures!r}; expected {[EXPECTED_ARCHITECTURE]!r}")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise SmokePreflightError(f"unexpected Qwen model_type {config.get('model_type')!r}; expected {EXPECTED_MODEL_TYPE!r}")
    if config.get("dtype") != "bfloat16":
        raise SmokePreflightError(f"Qwen checkpoint dtype must be bfloat16, found {config.get('dtype')!r}")
    shards = require_weight_index(root, "model.safetensors.index.json")
    if not (BOOTSTRAP_PATH / "sitecustomize.py").is_file():
        raise SmokePreflightError(f"vLLM tokenizer bootstrap is missing: {BOOTSTRAP_PATH}")
    return {
        "architecture": EXPECTED_ARCHITECTURE,
        "dtype": "bfloat16",
        "model_path": str(root),
        "model_type": EXPECTED_MODEL_TYPE,
        "weight_shards": len(shards),
    }


def collect_preflight(
    *,
    model_path: str | Path,
    gpu_index: int | None,
    version_getter: Callable[[str], str] = version,
    gpu_query: Callable[[int], Any] = query_gpu_status,
) -> dict[str, Any]:
    model = inspect_model(model_path)
    versions = require_versions(EXPECTED_VERSIONS, version_getter=version_getter)
    gpu = None
    if gpu_index is not None:
        if gpu_index < 0:
            raise SmokePreflightError("GPU index must be non-negative")
        status = gpu_query(gpu_index)
        assert_gpu_idle(status)
        gpu = status.to_dict()
    return {
        "cache_policy": "unique /tmp directory, removed on exit",
        "gpu": gpu,
        "model": model,
        "production_parser": "teleboost.reward.adapters.video_vlm_score.parse_eval_score",
        "status": "preflight_ok",
        "tool": "qwen3vl_vllm_real_smoke",
        "versions": versions,
    }


def build_synthetic_video(np_module: Any) -> Any:
    frames = np_module.full(EXPECTED_VIDEO_SHAPE, 24, dtype=np_module.uint8)
    for frame_index in range(EXPECTED_VIDEO_SHAPE[0]):
        left = 8 + frame_index * 8
        frames[frame_index, 20:44, left : left + 16] = (240, 48, 48)
    return frames


def build_video_metadata() -> dict[str, Any]:
    return {
        "do_sample_frames": False,
        "duration": 2.0,
        "fps": 2.0,
        "frames_indices": [0, 1, 2, 3],
        "total_num_frames": 4,
        "video_backend": "opencv",
    }


def validate_judge_output(
    *,
    raw_text: str,
    token_count: int,
    frames_shape: tuple[int, ...],
    frames_finite: bool,
    parser: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    if frames_shape != EXPECTED_VIDEO_SHAPE:
        raise RuntimeError(f"synthetic video shape changed: {frames_shape} != {EXPECTED_VIDEO_SHAPE}")
    if not frames_finite:
        raise RuntimeError("synthetic video contains non-finite values")
    if token_count <= 0 or not raw_text.strip():
        raise RuntimeError("Qwen3-VL returned no generated tokens/text")
    parsed = parser(raw_text)
    if parsed.get("failed"):
        raise RuntimeError(f"production judge parser rejected output: {parsed}")
    score = parsed.get("score")
    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        raise RuntimeError(f"production judge score is invalid: {score!r}")
    missing_dimensions = [name for name in PRODUCTION_DIMENSIONS if name not in parsed]
    if missing_dimensions:
        raise RuntimeError("judge output did not satisfy the five-dimension production contract: " + ", ".join(missing_dimensions))
    return parsed


def _bootstrap_environment() -> dict[str, str]:
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [str(BOOTSTRAP_PATH), str(REPO_ROOT)]
    if current_pythonpath:
        entries.append(current_pythonpath)
    return {
        "PYTHONPATH": os.pathsep.join(entries),
        "TELEBOOST_VLLM_TOKENIZER_REGEX_FIX": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def _install_current_interpreter_bootstrap() -> None:
    path = BOOTSTRAP_PATH / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("_teleboost_smoke_sitecustomize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load vLLM tokenizer bootstrap: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install()


def _load_runtime() -> SimpleNamespace:
    import numpy as np
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from teleboost.reward.adapters.video_vlm_score import (
        EVAL_PROMPT_TEMPLATE,
        parse_eval_score,
    )

    return SimpleNamespace(
        AutoProcessor=AutoProcessor,
        EVAL_PROMPT_TEMPLATE=EVAL_PROMPT_TEMPLATE,
        LLM=LLM,
        SamplingParams=SamplingParams,
        np=np,
        parse_eval_score=parse_eval_score,
    )


def _shutdown_vllm(llm: Any) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        # Cleanup must not hide the inference/parser failure that triggered it.
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def run_real_smoke(
    *,
    model_path: str | Path,
    gpu_index: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    if not 0.20 <= gpu_memory_utilization <= 0.80:
        raise SmokePreflightError("--gpu-memory-utilization must be within [0.20, 0.80]")

    # Recheck immediately before importing a CUDA runtime.  This reduces, but
    # cannot eliminate, the nvidia-smi check/use race.
    assert_gpu_idle(query_gpu_status(gpu_index))
    root = Path(model_path).expanduser().resolve()
    llm = None
    runtime = None
    with isolated_cache_environment(
        prefix="teleboost-qwen3vl-smoke-",
        gpu_index=gpu_index,
        extra_environment=_bootstrap_environment(),
    ) as cache_root:
        try:
            _install_current_interpreter_bootstrap()
            runtime = _load_runtime()
            frames = build_synthetic_video(runtime.np)
            metadata = build_video_metadata()
            eval_text = runtime.EVAL_PROMPT_TEMPLATE.format(caption="A red square moves horizontally across a dark background.")
            processor = runtime.AutoProcessor.from_pretrained(
                str(root),
                fix_mistral_regex=True,
                local_files_only=True,
            )
            prompt = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": eval_text},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            llm = runtime.LLM(
                model=str(root),
                tensor_parallel_size=1,
                dtype="bfloat16",
                trust_remote_code=False,
                seed=0,
                gpu_memory_utilization=gpu_memory_utilization,
                swap_space=0,
                enforce_eager=True,
                max_model_len=2048,
                max_num_seqs=1,
                limit_mm_per_prompt={"video": 1},
                disable_log_stats=True,
            )
            outputs = llm.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"video": (frames, metadata)},
                    }
                ],
                runtime.SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=128,
                    seed=0,
                ),
                use_tqdm=False,
            )
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError(f"vLLM returned an unexpected request/completion count: {outputs!r}")
            completion = outputs[0].outputs[0]
            parsed = validate_judge_output(
                raw_text=completion.text,
                token_count=len(completion.token_ids),
                frames_shape=tuple(frames.shape),
                frames_finite=bool(runtime.np.isfinite(frames).all()),
                parser=runtime.parse_eval_score,
            )
            return {
                "cache_root": str(cache_root),
                "gpu": gpu_index,
                "input_finite": True,
                "input_shape": list(frames.shape),
                "model_type": EXPECTED_MODEL_TYPE,
                "parsed": parsed,
                "raw": completion.text,
                "status": "ok",
                "token_count": len(completion.token_ids),
                "tool": "qwen3vl_vllm_real_smoke",
                "versions": {name: version(name) for name in EXPECTED_VERSIONS},
            }
        finally:
            if llm is not None:
                _shutdown_vllm(llm)
                del llm
            gc.collect()
            if runtime is not None:
                try:
                    runtime.np  # Keep the namespace alive through vLLM shutdown.
                except Exception:
                    pass


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        if args.gpu is not None and args.gpu < 0:
            raise SmokePreflightError("GPU index must be non-negative")
        preflight = collect_preflight(model_path=args.model_path, gpu_index=args.gpu)
        if args.mode == "preflight":
            print_result("QWEN3VL_VLLM_REAL_SMOKE_PREFLIGHT_OK", preflight)
            return 0
        if args.gpu is None:
            raise SmokePreflightError("--run requires --gpu or TELEBOOST_SMOKE_GPU")
        result = run_real_smoke(
            model_path=args.model_path,
            gpu_index=args.gpu,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print_result("QWEN3VL_VLLM_REAL_SMOKE_OK", result)
        return 0
    except SmokePreflightError as exc:
        print(f"QWEN3VL_VLLM_REAL_SMOKE_PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
