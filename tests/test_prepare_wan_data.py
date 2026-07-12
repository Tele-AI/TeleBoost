# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teleboost.datasets.preprocessing import wan as prepare_wan_data


def _args(input_path, output_dir):
    return SimpleNamespace(
        input=str(input_path),
        output_dir=str(output_dir),
        wan_model_path="unused-in-test",
        negative_prompt="negative",
    )


def test_wan_runtime_resolution_prefers_installed_distribution(monkeypatch):
    seen = []

    def fake_find_spec(name):
        seen.append(name)
        return object()

    monkeypatch.setattr(prepare_wan_data.importlib.util, "find_spec", fake_find_spec)
    assert prepare_wan_data._require_wan_runtime() is None
    assert seen == ["wan"]


def test_wan_runtime_resolution_fails_with_install_boundary(monkeypatch):
    monkeypatch.setattr(
        prepare_wan_data.importlib.util,
        "find_spec",
        lambda _name: None,
    )
    with pytest.raises(RuntimeError, match="top-level 'wan' package"):
        prepare_wan_data._require_wan_runtime()


def test_prepare_preserves_valid_per_row_null_embedding(tmp_path, monkeypatch):
    existing_context = tmp_path / "existing_context.npy"
    custom_null = tmp_path / "custom_null.npy"
    np.save(existing_context, np.array([1.0], dtype=np.float32))
    np.save(custom_null, np.array([2.0], dtype=np.float32))

    manifest = tmp_path / "prompts.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "caption": "already encoded",
                    "context_path": str(existing_context),
                    "context_null_path": str(custom_null),
                },
                {"caption": "needs encoding"},
            ]
        ),
        encoding="utf-8",
    )

    build_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        prepare_wan_data,
        "_build_t5",
        lambda *_args: build_calls.append(True) or object(),
    )
    monkeypatch.setattr(
        prepare_wan_data,
        "_encode",
        lambda _encoder, prompt, _device: np.array([len(prompt)], dtype=np.float32),
    )

    output_dir = tmp_path / "out"
    prepare_wan_data.prepare(_args(manifest, output_dir))
    rows = json.loads((output_dir / "processed_wan_prompt.json").read_text(encoding="utf-8"))

    assert build_calls == [True]
    assert rows[0]["context_path"] == str(existing_context)
    assert rows[0]["context_null_path"] == str(custom_null)
    assert rows[1]["context_path"] == str(output_dir / "context_000001.npy")
    assert rows[1]["context_null_path"] == str(output_dir / "context_null.npy")


def test_prepare_skips_t5_when_every_row_has_valid_custom_paths(tmp_path, monkeypatch):
    context = tmp_path / "context.npy"
    null = tmp_path / "negative.npy"
    np.save(context, np.array([1.0], dtype=np.float32))
    np.save(null, np.array([2.0], dtype=np.float32))
    manifest = tmp_path / "prompts.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "caption": "complete",
                    "context_path": str(context),
                    "context_null_path": str(null),
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def should_not_load(*_args):
        raise AssertionError("T5 must not load for a complete manifest")

    monkeypatch.setattr(prepare_wan_data, "_build_t5", should_not_load)
    output_dir = tmp_path / "out"
    prepare_wan_data.prepare(_args(manifest, output_dir))

    rows = json.loads((output_dir / "processed_wan_prompt.json").read_text(encoding="utf-8"))
    assert rows[0]["context_path"] == str(context)
    assert rows[0]["context_null_path"] == str(null)
    assert not (output_dir / "context_null.npy").exists()
