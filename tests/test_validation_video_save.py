"""_save_video_and_prompt must handle bfloat16 frames (NumPy rejects them)."""

import sys
import types

import torch


def _install_fake_cv2(monkeypatch):
    writers = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            self.frames = []
            writers.append(self)

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            pass

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.VideoWriter = _FakeWriter
    fake_cv2.VideoWriter_fourcc = lambda *args: 0
    fake_cv2.cvtColor = lambda frame, flag: frame
    fake_cv2.COLOR_RGB2BGR = 0
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return writers


def test_save_video_and_prompt_accepts_bf16_frames(monkeypatch, tmp_path):
    from teleboost.training.core.trainer import _save_video_and_prompt

    writers = _install_fake_cv2(monkeypatch)
    monkeypatch.chdir(tmp_path)
    frames = torch.rand(3, 2, 4, 4).to(torch.bfloat16)  # (C, T, H, W), [0, 1]

    _save_video_and_prompt(frames, 0, 0)

    assert len(writers) == 1
    assert len(writers[0].frames) == 2
    assert all(f.dtype.name == "uint8" for f in writers[0].frames)
