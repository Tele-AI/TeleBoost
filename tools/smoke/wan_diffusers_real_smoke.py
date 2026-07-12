#!/usr/bin/env python3
# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Safe one-GPU Wan Diffusers real-checkpoint smoke.

The default action is preflight only.  ``--run`` executes one 64x64, five-
frame, one-step T5+DiT+scheduler forward and validates the latent in memory.
Add ``--decode`` to include the real VAE decode.  Nothing is saved to disk;
all mutable caches live in a unique, automatically removed ``/tmp`` tree.
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
# local helper (and later model modules) so the smoke never creates
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
    "diffusers": "0.39.0",
    "numpy": "1.26.4",
    "torch": "2.9.1",
    "transformers": "4.57.6",
}
EXPECTED_PIPELINE_CLASS = "WanPipeline"
EXPECTED_TRANSFORMER_CLASS = "WanTransformer3DModel"
EXPECTED_VAE_CLASS = "AutoencoderKLWan"
EXPECTED_LATENT_SHAPE = (1, 16, 2, 8, 8)
EXPECTED_DECODED_SHAPE = (1, 5, 3, 64, 64)


def _environment_model_path() -> str:
    return os.environ.get("TELEBOOST_WAN_DIFFUSERS_MODEL_PATH", DEFAULT_MODEL_PATH)


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
        help="run one real Wan forward (requires --gpu)",
    )
    parser.set_defaults(mode="preflight")
    parser.add_argument(
        "--model-path",
        default=_environment_model_path(),
        help="local Wan Diffusers directory (env TELEBOOST_WAN_DIFFUSERS_MODEL_PATH)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=_environment_gpu(),
        help="physical GPU index; --run requires an idle GPU",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="also run the VAE decode and validate a [1,5,3,64,64] video tensor",
    )
    return parser


def _require_component(
    model_index: dict[str, Any],
    component: str,
    expected_class: str,
) -> None:
    value = model_index.get(component)
    if not isinstance(value, list) or len(value) != 2 or value[1] != expected_class:
        raise SmokePreflightError(f"unexpected Wan component {component}={value!r}; expected class {expected_class}")


def inspect_model(model_path: str | Path) -> dict[str, Any]:
    if not str(model_path).strip():
        raise SmokePreflightError("set --model-path or TELEBOOST_WAN_DIFFUSERS_MODEL_PATH")
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise SmokePreflightError(f"Wan Diffusers model directory does not exist: {root}")
    require_files(
        root,
        (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/spiece.model",
            "tokenizer/tokenizer.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
    model_index = read_json(root / "model_index.json")
    if model_index.get("_class_name") != EXPECTED_PIPELINE_CLASS:
        raise SmokePreflightError(f"unexpected Wan pipeline {model_index.get('_class_name')!r}; expected {EXPECTED_PIPELINE_CLASS!r}")
    _require_component(model_index, "transformer", EXPECTED_TRANSFORMER_CLASS)
    _require_component(model_index, "vae", EXPECTED_VAE_CLASS)
    _require_component(model_index, "text_encoder", "UMT5EncoderModel")
    _require_component(model_index, "scheduler", "UniPCMultistepScheduler")

    transformer_config = read_json(root / "transformer" / "config.json")
    if transformer_config.get("_class_name") != EXPECTED_TRANSFORMER_CLASS:
        raise SmokePreflightError(f"unexpected transformer config class: {transformer_config.get('_class_name')!r}")
    vae_config = read_json(root / "vae" / "config.json")
    if vae_config.get("_class_name") != EXPECTED_VAE_CLASS:
        raise SmokePreflightError(f"unexpected VAE config class: {vae_config.get('_class_name')!r}")
    text_config = read_json(root / "text_encoder" / "config.json")
    if text_config.get("model_type") != "umt5":
        raise SmokePreflightError(f"unexpected text encoder model_type: {text_config.get('model_type')!r}")

    transformer_shards = require_weight_index(root, "transformer/diffusion_pytorch_model.safetensors.index.json")
    text_shards = require_weight_index(root, "text_encoder/model.safetensors.index.json")
    return {
        "model_path": str(root),
        "pipeline_class": EXPECTED_PIPELINE_CLASS,
        "scheduler_class": "UniPCMultistepScheduler",
        "text_encoder_class": "UMT5EncoderModel",
        "text_encoder_shards": len(text_shards),
        "transformer_class": EXPECTED_TRANSFORMER_CLASS,
        "transformer_shards": len(transformer_shards),
        "vae_class": EXPECTED_VAE_CLASS,
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
        "run_contract": {
            "decoded_shape": list(EXPECTED_DECODED_SHAPE),
            "latent_shape": list(EXPECTED_LATENT_SHAPE),
            "num_frames": 5,
            "num_inference_steps": 1,
            "resolution": [64, 64],
        },
        "status": "preflight_ok",
        "tool": "wan_diffusers_real_smoke",
        "versions": versions,
    }


def validate_wan_output(
    *,
    shape: tuple[int, ...],
    finite: bool,
    variation: float,
    decoded: bool,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    expected = EXPECTED_DECODED_SHAPE if decoded else EXPECTED_LATENT_SHAPE
    if shape != expected:
        raise RuntimeError(f"Wan output shape mismatch: {shape} != {expected}")
    if not finite:
        raise RuntimeError("Wan output contains NaN or infinity")
    if variation <= 0.0:
        raise RuntimeError(f"Wan output is constant (std={variation})")
    if decoded:
        if minimum is None or maximum is None:
            raise RuntimeError("decoded Wan validation requires min/max values")
        if minimum < 0.0 or maximum > 1.0:
            raise RuntimeError(f"decoded Wan output is outside the postprocessed [0,1] range: [{minimum}, {maximum}]")


def _load_runtime() -> SimpleNamespace:
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    return SimpleNamespace(
        AutoencoderKLWan=AutoencoderKLWan,
        WanPipeline=WanPipeline,
        torch=torch,
    )


def run_real_smoke(
    *,
    model_path: str | Path,
    gpu_index: int,
    decode: bool,
) -> dict[str, Any]:
    # Recheck immediately before importing torch/CUDA.  This reduces, but
    # cannot eliminate, the nvidia-smi check/use race.
    assert_gpu_idle(query_gpu_status(gpu_index))
    root = Path(model_path).expanduser().resolve()
    runtime = None
    pipeline = None
    vae = None
    with isolated_cache_environment(
        prefix="teleboost-wan-diffusers-smoke-",
        gpu_index=gpu_index,
    ) as cache_root:
        try:
            runtime = _load_runtime()
            torch = runtime.torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(f"Wan smoke requires exactly one visible CUDA GPU after isolation; found {torch.cuda.device_count()}")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("the selected GPU does not support bfloat16")
            torch.cuda.set_device(0)
            vae = runtime.AutoencoderKLWan.from_pretrained(
                str(root),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            pipeline = runtime.WanPipeline.from_pretrained(
                str(root),
                vae=vae,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=False,
            )
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to("cuda")
            output_type = "pt" if decode else "latent"
            output = pipeline(
                prompt="A small red square moves across a dark background.",
                negative_prompt=None,
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=1,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(0),
                output_type=output_type,
                max_sequence_length=32,
            ).frames
            torch.cuda.synchronize()
            shape = tuple(output.shape)
            finite = bool(torch.isfinite(output).all().item())
            float_output = output.float()
            variation = float(float_output.std().item())
            minimum = float(float_output.min().item())
            maximum = float(float_output.max().item())
            validate_wan_output(
                shape=shape,
                finite=finite,
                variation=variation,
                decoded=decode,
                minimum=minimum,
                maximum=maximum,
            )
            result = {
                "cache_root": str(cache_root),
                "decoded": decode,
                "dtype": str(output.dtype),
                "finite": finite,
                "gpu": gpu_index,
                "maximum": maximum,
                "minimum": minimum,
                "model_type": EXPECTED_TRANSFORMER_CLASS,
                "shape": list(shape),
                "status": "ok",
                "std": variation,
                "tool": "wan_diffusers_real_smoke",
                "versions": {name: version(name) for name in EXPECTED_VERSIONS},
            }
            del float_output, output
            return result
        finally:
            if pipeline is not None:
                del pipeline
            if vae is not None:
                del vae
            gc.collect()
            if runtime is not None:
                try:
                    runtime.torch.cuda.empty_cache()
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
            print_result("WAN_DIFFUSERS_REAL_SMOKE_PREFLIGHT_OK", preflight)
            return 0
        if args.gpu is None:
            raise SmokePreflightError("--run requires --gpu or TELEBOOST_SMOKE_GPU")
        result = run_real_smoke(
            model_path=args.model_path,
            gpu_index=args.gpu,
            decode=args.decode,
        )
        print_result("WAN_DIFFUSERS_REAL_SMOKE_OK", result)
        return 0
    except SmokePreflightError as exc:
        print(f"WAN_DIFFUSERS_REAL_SMOKE_PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
