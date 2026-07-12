from __future__ import annotations

from types import SimpleNamespace

import cv2  # noqa: F401 - adapter runtime dependency
import pytest
import torch


def test_video_vlm_adapter_writes_mp4_and_scores(monkeypatch):
    from teleboost.reward.adapters import video_vlm
    from teleboost.reward.adapters import video_vlm_score

    seen: list[dict] = []

    async def fake_compute_score(
        data_source=None,
        solution_str=None,
        ground_truth=None,
        extra_info=None,
        *,
        reward_router_address=None,
        **_kwargs,
    ):
        del data_source, solution_str
        assert reward_router_address == "127.0.0.1:9999"
        assert ground_truth == {"caption": extra_info["caption"]}
        assert extra_info["video_url"].startswith("data:video/mp4;base64,")
        assert extra_info["media_uuid"].startswith("teleboost-video-")
        seen.append(dict(extra_info))
        return {"score": 0.75, "raw": "合计:75分"}

    def fake_reward_tensor(scores: torch.Tensor, batch_size: int):
        return SimpleNamespace(batch={"rewards": scores}, non_tensor_batch={}, batch_size=batch_size)

    monkeypatch.setattr(video_vlm_score, "compute_score", fake_compute_score)
    monkeypatch.setattr(video_vlm, "build_reward_tensor", fake_reward_tensor)

    data = SimpleNamespace(
        batch={"video_frames": torch.rand(2, 3, 2, 8, 8)},
        non_tensor_batch={"caption": ["red cube", "blue cube"]},
    )
    cfg = {
        "actor_rollout_ref": {"video_fps": 4},
        "reward": {
            "reward_model": {
                "enable": True,
                "adapter": "video_vlm",
                "model_path": "/models/video-vlm",
            }
        },
    }

    out = video_vlm.compute_video_vlm_reward(cfg, data, reward_router_address="127.0.0.1:9999")

    assert out.batch_size == 2
    assert torch.equal(out.batch["rewards"], torch.tensor([0.75, 0.75]))
    assert [item["caption"] for item in seen] == ["red cube", "blue cube"]
    assert [item["reward_model_name"] for item in seen] == ["/models/video-vlm", "/models/video-vlm"]


def test_video_vlm_adapter_accepts_bf16_frames(monkeypatch):
    from teleboost.reward.adapters import video_vlm
    from teleboost.reward.adapters import video_vlm_score

    async def fake_compute_score(*_args, **_kwargs):
        return {"score": 0.5, "raw": "50"}

    monkeypatch.setattr(video_vlm_score, "compute_score", fake_compute_score)
    monkeypatch.setattr(
        video_vlm,
        "build_reward_tensor",
        lambda scores, batch_size: SimpleNamespace(batch={"rewards": scores}, batch_size=batch_size),
    )
    data = SimpleNamespace(
        batch={"video_frames": torch.rand(1, 3, 2, 8, 8).to(torch.bfloat16)},
        non_tensor_batch={"caption": ["red cube"]},
    )
    cfg = {"reward": {"reward_model": {"enable": True, "adapter": "video_vlm"}}}

    out = video_vlm.compute_video_vlm_reward(cfg, data, reward_router_address="127.0.0.1:9999")

    assert out.batch_size == 1
    assert torch.equal(out.batch["rewards"], torch.tensor([0.5]))


def test_video_vlm_adapter_rejects_non_rgb(monkeypatch):
    from teleboost.reward.adapters import video_vlm

    monkeypatch.setattr(
        video_vlm,
        "build_reward_tensor",
        lambda scores, batch_size: SimpleNamespace(batch={"rewards": scores}, batch_size=batch_size),
    )
    data = SimpleNamespace(
        batch={"video_frames": torch.rand(1, 1, 2, 8, 8)},
        non_tensor_batch={"caption": ["gray"]},
    )
    cfg = {"reward": {"reward_model": {"enable": True, "adapter": "video_vlm"}}}

    with pytest.raises(ValueError, match="3-channel RGB"):
        video_vlm.compute_video_vlm_reward(cfg, data, reward_router_address="127.0.0.1:9999")
