# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Static gates for the model-family-neutral trainer base."""

from __future__ import annotations

import ast
from pathlib import Path

_BASE_TRAINER = Path(__file__).parents[1] / "teleboost" / "training" / "core" / "trainer.py"


def test_base_trainer_source_has_no_family_dispatch_gates():
    source = _BASE_TRAINER.read_text(encoding="utf-8").lower()

    assert "trainer.type" not in source
    assert "vae_stride" not in source
    assert "latent_channels" not in source


def test_base_trainer_ast_has_no_model_family_imports():
    tree = ast.parse(_BASE_TRAINER.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)

    model_imports = []
    for module in imported_modules:
        if module.startswith("teleboost.models"):
            model_imports.append(module)
    assert model_imports == []


def test_base_trainer_ast_keeps_generation_as_a_required_seam():
    tree = ast.parse(_BASE_TRAINER.read_text(encoding="utf-8"))
    trainer_classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RayTeleBoostTrainer":
            trainer_classes.append(node)
    assert len(trainer_classes) == 1
    trainer_class = trainer_classes[0]

    methods = {}
    for node in trainer_class.body:
        if isinstance(node, ast.FunctionDef):
            methods[node.name] = node

    assert "_prepare_diffusion_inputs" not in methods
    assert "_sd3_time_shift" not in methods
    build_nodes = ast.walk(methods["_build_gen_batch"])
    assert any(isinstance(node, ast.Raise) for node in build_nodes)
