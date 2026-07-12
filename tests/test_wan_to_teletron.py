"""Artifact and naming contracts for the Wan → TeleTron converter."""

from __future__ import annotations

import tomllib
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from teleboost.artifacts.wan_conversion import (
    _expand_paths,
    load_state_dict,
    load_teletron_release,
    rename_teletron_to_wan,
    rename_wan_to_teletron,
    roundtrip_check,
    save_teletron_release,
)


def _state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "blocks.0.self_attn.k.weight": torch.arange(4).reshape(2, 2),
            "blocks.0.self_attn.norm_q.weight": torch.ones(2),
            "blocks.0.cross_attn.k_img.weight": torch.full((2,), 2),
            "patch_embedding.weight": torch.full((1,), 3),
            "time_projection.bias": torch.full((1,), 4),
        }
    )


def test_key_rename_and_release_checkpoint_are_bit_exact(tmp_path: Path) -> None:
    source = _state()
    converted = rename_wan_to_teletron(source)

    assert "blocks.0.self_attn.key.weight" in converted
    assert "blocks.0.self_attn.norm_query.weight" in converted
    assert "blocks.0.cross_attn.img_key.weight" in converted
    assert "patch_emb.weight" in converted
    assert "time_proj.bias" in converted

    destination = tmp_path / "teletron"
    save_teletron_release(converted, str(destination))
    reloaded = load_teletron_release(str(destination))
    ok, diagnostics = roundtrip_check(source, reloaded)

    assert ok, diagnostics
    assert rename_teletron_to_wan(reloaded).keys() == source.keys()
    assert (destination / "latest_checkpointed_iteration.txt").read_text() == "release"


def test_converter_rejects_rename_and_shard_collisions(tmp_path: Path) -> None:
    collision = OrderedDict(
        {
            "block.k.weight": torch.zeros(1),
            "block.key.weight": torch.ones(1),
        }
    )
    with pytest.raises(ValueError, match="rename collision"):
        rename_wan_to_teletron(collision)

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save({"same.weight": torch.zeros(1)}, first)
    torch.save({"same.weight": torch.ones(1)}, second)
    with pytest.raises(ValueError, match="repeats 1 parameter keys"):
        load_state_dict([str(first), str(second)])


def test_converter_rejects_duplicate_source_paths(tmp_path: Path) -> None:
    source = tmp_path / "weights.pt"
    torch.save({}, source)
    with pytest.raises(ValueError, match="duplicate paths"):
        _expand_paths([str(source), str(source)])


def test_only_explicit_converter_entry_point_is_published() -> None:
    scripts = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts["teleboost-convert-wan-to-teletron"] == "teleboost.cli.convert_checkpoint:main"
    assert "teleboost-convert-wan" not in scripts
