"""CPU-only contract tests for the production DPO CSV data path."""

from __future__ import annotations

import csv

import pytest
import torch
from PIL import Image

from teleboost.datasets.dpo import CSVPreferenceDPODataset
from teleboost.datasets.build import build_dataset


_TRANSFORMS = [
    {
        "type": "InjectRawFirstImageFromVideo",
        "video_key": "video",
        "output_key": "raw_first_image",
    },
    {
        "type": "PreprocessVideoToTensor",
        "input_key": "video",
        "output_key": "video",
        "torch_dtype": "bfloat16",
        "pattern": "B C T H W",
        "min_value": -1,
        "max_value": 1,
        "skip_if_tensor": True,
    },
    {
        "type": "InjectImagesFromVideoTensor",
        "video_key": "video",
        "output_key": "images",
    },
    {
        "type": "InjectPromptToTopLevel",
        "prompt_key": "caption",
    },
    {
        "type": "PackInputsNoResize",
        "normalize": False,
        "image_keys": ["images"],
        "embedding_keys": ["raw_first_image", "input_image"],
    },
]


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cpu_video_loader(path):
    offset = 20 if path.endswith("chosen.mp4") else 120
    return [Image.new("RGB", (20, 10), color=(offset + index, 2, 3)) for index in range(5)]


def test_csv_preference_dataset_reads_custom_columns_and_builds_encoder_raw_schema(tmp_path):
    (tmp_path / "chosen.mp4").touch()
    (tmp_path / "rejected.mp4").touch()
    manifest = tmp_path / "pairs.csv"
    prompt = "a quoted, harmless __import__('os') prompt"
    _write_csv(
        manifest,
        ["good_video", "bad_video", "caption"],
        [
            {
                "good_video": "chosen.mp4",
                "bad_video": "rejected.mp4",
                "caption": prompt,
            }
        ],
    )

    dataset = build_dataset(
        {
            "type": "CSVPreferenceDPODataset",
            "data_path_list": [manifest],
            "dataset_base_path": tmp_path,
            "dataset_repeat": 2,
            "chosen_video_key": "winner",
            "rejected_video_key": "loser",
            "chosen_path_key": "good_video",
            "rejected_path_key": "bad_video",
            "prompt_key": "caption",
            "transforms": _TRANSFORMS,
            "height": 8,
            "width": 12,
            "num_frames": 3,
            "time_division_factor": 2,
            "time_division_remainder": 1,
            "height_division_factor": 4,
            "width_division_factor": 4,
            "video_loader": _cpu_video_loader,
        }
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert set(item) == {"winner", "loser"}
    for branch in item.values():
        assert branch["images"].shape == (3, 3, 8, 12)
        assert branch["images"].dtype == torch.bfloat16
        assert branch["images"].device.type == "cpu"
        assert branch["raw_first_image"].shape == (1, 3, 8, 12)
        assert branch["input_image"].shape == (1, 3, 8, 12)
        assert branch["struct_prompt"] == [prompt]

    # This is the exact dimensional transition expected by WanTeletronEncoder:
    # DataLoader adds B to per-sample [T,C,H,W] raw video tensors.
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
    assert batch["winner"]["images"].shape == (1, 3, 3, 8, 12)
    assert batch["loser"]["images"].shape == (1, 3, 3, 8, 12)


def test_csv_preference_dataset_fails_fast_for_bad_schema_or_path(tmp_path):
    missing_column = tmp_path / "missing_column.csv"
    _write_csv(
        missing_column,
        ["good_video", "caption"],
        [{"good_video": "chosen.mp4", "caption": "prompt"}],
    )
    with pytest.raises(ValueError, match="missing required columns.*bad_video"):
        CSVPreferenceDPODataset(
            data_path_list=[missing_column],
            chosen_path_key="good_video",
            rejected_path_key="bad_video",
            prompt_key="caption",
        )

    missing_path = tmp_path / "missing_path.csv"
    _write_csv(
        missing_path,
        ["good_video", "bad_video", "caption"],
        [{"good_video": "does-not-exist.mp4", "bad_video": "also-missing.mp4", "caption": "prompt"}],
    )
    with pytest.raises(FileNotFoundError, match="good_video.*does not exist"):
        CSVPreferenceDPODataset(
            data_path_list=[missing_path],
            chosen_path_key="good_video",
            rejected_path_key="bad_video",
            prompt_key="caption",
        )


def test_production_wan_i2v_config_uses_real_dataset_and_image_conditioning():
    from teleboost.programs.wan.dpo.wan_dpo_i2v import config
    from teleboost.datasets.build import DATASETS

    assert config["dataset"]["type"] == "CSVPreferenceDPODataset"
    assert config["dataset"]["type"] in DATASETS
    assert config["dataset"]["chosen_video_key"] == "chosen"
    assert config["dataset"]["rejected_video_key"] == "rejected"
    assert config["dataset"]["chosen_path_key"]
    assert config["dataset"]["rejected_path_key"]
    assert config["dataset"]["prompt_key"]
    assert config["model_config"]["dit"]["config"]["has_image_input"] is True
    assert config["model_config"]["dit"]["config"]["in_dim"] == 36
    assert config["model_config"]["encoder"]["encoder_schema"] == [
        "context",
        "img_clip_feature",
        "img_emb_y",
        "latents",
    ]


def test_wan_t2v_override_maps_csv_columns_without_changing_dpo_branch_schema():
    from teleboost.programs.wan.dpo.wan_dpo_t2v import config

    assert config["dataset"]["type"] == "CSVPreferenceDPODataset"
    assert config["dataset"]["chosen_video_key"] == "chosen"
    assert config["dataset"]["rejected_video_key"] == "rejected"
    assert config["dataset"]["chosen_path_key"] == "positive_video_path"
    assert config["dataset"]["rejected_path_key"] == "negative_video_path"
    assert config["model_config"]["dit"]["config"]["has_image_input"] is False
    assert config["model_config"]["encoder"]["encoder_schema"] == ["context", "latents"]
