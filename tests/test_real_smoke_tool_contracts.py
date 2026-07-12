"""CPU-only contracts for the standalone real-checkpoint smoke tools.

These tests use tiny manifests and mocked GPU/package data.  Importing this
module must never construct vLLM, Diffusers, a model, or a CUDA context.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.smoke import qwen3vl_vllm_real_smoke as qwen_smoke
from tools.smoke import wan_diffusers_real_smoke as wan_smoke
from tools.smoke._real_model_smoke_common import (
    SmokePreflightError,
    assert_gpu_idle,
    isolated_cache_environment,
    parse_gpu_rows,
    require_versions,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _fake_qwen_tree(root: Path) -> None:
    _write_json(
        root / "config.json",
        {
            "architectures": [qwen_smoke.EXPECTED_ARCHITECTURE],
            "dtype": "bfloat16",
            "model_type": qwen_smoke.EXPECTED_MODEL_TYPE,
        },
    )
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}},
    )
    for relative in (
        "chat_template.jinja",
        "model-00001-of-00001.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ):
        _touch(root / relative)


def _fake_wan_tree(root: Path) -> None:
    _write_json(
        root / "model_index.json",
        {
            "_class_name": wan_smoke.EXPECTED_PIPELINE_CLASS,
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", wan_smoke.EXPECTED_TRANSFORMER_CLASS],
            "vae": ["diffusers", wan_smoke.EXPECTED_VAE_CLASS],
        },
    )
    _write_json(
        root / "transformer" / "config.json",
        {"_class_name": wan_smoke.EXPECTED_TRANSFORMER_CLASS},
    )
    _write_json(
        root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        {"weight_map": {"layer.weight": "transformer.safetensors"}},
    )
    _touch(root / "transformer" / "transformer.safetensors")
    _write_json(
        root / "text_encoder" / "config.json",
        {"architectures": ["UMT5EncoderModel"], "model_type": "umt5"},
    )
    _write_json(
        root / "text_encoder" / "model.safetensors.index.json",
        {"weight_map": {"encoder.weight": "text.safetensors"}},
    )
    _touch(root / "text_encoder" / "text.safetensors")
    _write_json(
        root / "vae" / "config.json",
        {"_class_name": wan_smoke.EXPECTED_VAE_CLASS},
    )
    _touch(root / "vae" / "diffusion_pytorch_model.safetensors")
    _write_json(
        root / "scheduler" / "scheduler_config.json",
        {"_class_name": "UniPCMultistepScheduler"},
    )
    _touch(root / "tokenizer" / "spiece.model")
    _touch(root / "tokenizer" / "tokenizer.json")


def test_qwen_preflight_checks_manifest_without_loading_runtime(tmp_path, monkeypatch):
    _fake_qwen_tree(tmp_path)
    monkeypatch.setattr(
        qwen_smoke,
        "_load_runtime",
        lambda: pytest.fail("preflight must not import or construct the model runtime"),
    )
    result = qwen_smoke.collect_preflight(
        model_path=tmp_path,
        gpu_index=None,
        version_getter=qwen_smoke.EXPECTED_VERSIONS.__getitem__,
    )
    assert result["status"] == "preflight_ok"
    assert result["model"]["architecture"] == qwen_smoke.EXPECTED_ARCHITECTURE
    assert result["gpu"] is None


def test_qwen_cli_defaults_to_preflight_and_never_calls_real_runner(monkeypatch, capsys):
    monkeypatch.setattr(
        qwen_smoke,
        "collect_preflight",
        lambda **_kwargs: {"status": "preflight_ok"},
    )
    monkeypatch.setattr(
        qwen_smoke,
        "run_real_smoke",
        lambda **_kwargs: pytest.fail("default CLI mode must not load a model"),
    )
    assert qwen_smoke.main([]) == 0
    assert "QWEN3VL_VLLM_REAL_SMOKE_PREFLIGHT_OK" in capsys.readouterr().out


def test_qwen_preflight_rejects_a_missing_weight_shard(tmp_path):
    _fake_qwen_tree(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(SmokePreflightError, match="missing shards"):
        qwen_smoke.inspect_model(tmp_path)


@pytest.mark.parametrize(
    "inspect_model",
    [qwen_smoke.inspect_model, wan_smoke.inspect_model],
)
def test_real_smoke_requires_an_explicit_model_path(inspect_model):
    with pytest.raises(SmokePreflightError, match="model-path"):
        inspect_model("")


def test_qwen_video_metadata_matches_vllm_014_contract():
    frames = qwen_smoke.build_synthetic_video(np)
    metadata = qwen_smoke.build_video_metadata()
    assert frames.shape == qwen_smoke.EXPECTED_VIDEO_SHAPE
    assert frames.dtype == np.uint8
    assert metadata == {
        "do_sample_frames": False,
        "duration": 2.0,
        "fps": 2.0,
        "frames_indices": [0, 1, 2, 3],
        "total_num_frames": 4,
        "video_backend": "opencv",
    }


def test_qwen_output_validation_uses_the_production_parser_contract():
    calls = []

    def parser(raw):
        calls.append(raw)
        return {
            "score": 0.72,
            "score_raw": 72.0,
            **{name: 72.0 for name in qwen_smoke.PRODUCTION_DIMENSIONS},
        }

    parsed = qwen_smoke.validate_judge_output(
        raw_text="dim1:72分,dim2:72分,dim3:72分,dim4:72分,dim5:72分,合计:72分",
        token_count=24,
        frames_shape=qwen_smoke.EXPECTED_VIDEO_SHAPE,
        frames_finite=True,
        parser=parser,
    )
    assert calls and parsed["score"] == 0.72


@pytest.mark.parametrize(
    ("raw", "parsed", "message"),
    [
        ("", {"score": 0.5}, "no generated"),
        ("not parseable", {"score": 0.0, "failed": True}, "parser rejected"),
        ("合计:50分", {"score": 0.5}, "five-dimension"),
    ],
)
def test_qwen_output_validation_fails_closed(raw, parsed, message):
    with pytest.raises(RuntimeError, match=message):
        qwen_smoke.validate_judge_output(
            raw_text=raw,
            token_count=1,
            frames_shape=qwen_smoke.EXPECTED_VIDEO_SHAPE,
            frames_finite=True,
            parser=lambda _raw: parsed,
        )


def test_wan_preflight_checks_manifest_without_loading_runtime(tmp_path, monkeypatch):
    _fake_wan_tree(tmp_path)
    monkeypatch.setattr(
        wan_smoke,
        "_load_runtime",
        lambda: pytest.fail("preflight must not import or construct the model runtime"),
    )
    result = wan_smoke.collect_preflight(
        model_path=tmp_path,
        gpu_index=None,
        version_getter=wan_smoke.EXPECTED_VERSIONS.__getitem__,
    )
    assert result["status"] == "preflight_ok"
    assert result["model"]["pipeline_class"] == "WanPipeline"
    assert result["run_contract"]["latent_shape"] == [1, 16, 2, 8, 8]


def test_wan_cli_defaults_to_preflight_and_never_calls_real_runner(monkeypatch, capsys):
    monkeypatch.setattr(
        wan_smoke,
        "collect_preflight",
        lambda **_kwargs: {"status": "preflight_ok"},
    )
    monkeypatch.setattr(
        wan_smoke,
        "run_real_smoke",
        lambda **_kwargs: pytest.fail("default CLI mode must not load a model"),
    )
    assert wan_smoke.main([]) == 0
    assert "WAN_DIFFUSERS_REAL_SMOKE_PREFLIGHT_OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("shape", "finite", "variation", "decoded", "minimum", "maximum", "message"),
    [
        ((1, 16, 1, 8, 8), True, 1.0, False, None, None, "shape mismatch"),
        (wan_smoke.EXPECTED_LATENT_SHAPE, False, 1.0, False, None, None, "NaN"),
        (wan_smoke.EXPECTED_LATENT_SHAPE, True, 0.0, False, None, None, "constant"),
        (wan_smoke.EXPECTED_DECODED_SHAPE, True, 0.2, True, -0.1, 1.0, "outside"),
    ],
)
def test_wan_output_validation_fails_closed(
    shape,
    finite,
    variation,
    decoded,
    minimum,
    maximum,
    message,
):
    with pytest.raises(RuntimeError, match=message):
        wan_smoke.validate_wan_output(
            shape=shape,
            finite=finite,
            variation=variation,
            decoded=decoded,
            minimum=minimum,
            maximum=maximum,
        )


def test_wan_output_validation_accepts_latent_and_decoded_contracts():
    wan_smoke.validate_wan_output(
        shape=wan_smoke.EXPECTED_LATENT_SHAPE,
        finite=True,
        variation=0.25,
        decoded=False,
    )
    wan_smoke.validate_wan_output(
        shape=wan_smoke.EXPECTED_DECODED_SHAPE,
        finite=True,
        variation=0.25,
        decoded=True,
        minimum=0.0,
        maximum=1.0,
    )


def test_version_contract_is_a_hard_failure():
    with pytest.raises(SmokePreflightError, match="version contract failed"):
        require_versions(
            {"vllm": "0.14.0"},
            version_getter=lambda _name: "0.13.0",
        )


def test_gpu_parser_and_idle_guard_reject_foreign_compute_process():
    gpu_output = "7, GPU-abc, NVIDIA H100 80GB HBM3, 128, 0\n"
    process_output = "GPU-abc, 1234, python, 64\n"
    status = parse_gpu_rows(gpu_output, process_output, gpu_index=7)
    with pytest.raises(SmokePreflightError, match="pid=1234"):
        assert_gpu_idle(status)


def test_isolated_cache_environment_uses_and_removes_unique_tmp(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/original/hf")
    with isolated_cache_environment(prefix="teleboost-contract-test-", gpu_index=7) as root:
        assert root.parent == Path("/tmp")
        assert root.is_dir()
        assert Path(os.environ["HF_HOME"]).is_relative_to(root)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert not root.exists()
    assert os.environ["HF_HOME"] == "/original/hf"


def test_query_contract_uses_no_shell_or_training_runtime(monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        output = "0, GPU-safe, NVIDIA H100 80GB HBM3, 0, 0\n" if "--query-gpu=index,uuid,name,memory.used,utilization.gpu" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    from tools.smoke import _real_model_smoke_common as common

    status = common.query_gpu_status(0, command_runner=runner)
    assert status.index == 0 and not status.processes
    assert all(kwargs == {"check": True, "capture_output": True, "text": True} for _, kwargs in calls)
    assert all(command[0] == "nvidia-smi" for command, _ in calls)
