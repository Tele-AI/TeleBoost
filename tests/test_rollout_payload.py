# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from teleboost.training.core.payload import (
    drop_batch_tensor,
    video_tensor_to_uint8_frames,
)


class _FakeProto:
    def __init__(self):
        self.batch = {
            "video_frames": torch.ones(2, 3, 1, 2, 2),
            "log_probs": torch.zeros(2, 1),
        }
        # A same-named metadata key proves cleanup targets the tensor plane.
        self.non_tensor_batch = {"video_frames": np.array(["metadata"])}
        self.pop_calls = []

    def pop(self, *, batch_keys=None, non_tensor_batch_keys=None):
        self.pop_calls.append((batch_keys, non_tensor_batch_keys))
        for key in batch_keys or ():
            self.batch.pop(key)
        for key in non_tensor_batch_keys or ():
            self.non_tensor_batch.pop(key)


def test_drop_batch_tensor_removes_tensor_plane_and_is_idempotent():
    data = _FakeProto()
    assert drop_batch_tensor(data, "video_frames") is True
    assert "video_frames" not in data.batch
    assert "video_frames" in data.non_tensor_batch
    assert data.pop_calls == [(["video_frames"], None)]

    assert drop_batch_tensor(data, "video_frames") is False
    assert len(data.pop_calls) == 1


def test_bfloat16_video_is_cast_before_numpy_conversion():
    video = torch.tensor(
        [[[[0.0, 0.5]], [[1.0, 2.0]]]],
        dtype=torch.bfloat16,
    )
    frames = video_tensor_to_uint8_frames(video, clamp=True)

    assert frames.dtype == np.uint8
    assert frames.shape == (2, 1, 2, 1)
    assert frames[..., 0].tolist() == [[[0, 127]], [[255, 255]]]
