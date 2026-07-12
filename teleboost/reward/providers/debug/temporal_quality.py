"""Temporal-quality DIAGNOSTIC reward — cheap, no-checkpoint, no-caption.

THIS IS A DIAGNOSTIC REWARD, NOT A FINAL/AESTHETIC OBJECTIVE. It exists to answer
exactly one question: *does any within-group-discriminative signal unfreeze Wan
GRPO?* HPS scores only frame-0 text-image alignment (a per-PROMPT property), so a
prompt's n rollouts get near-identical reward (group_std≈0.003) → advantages are
noise → the policy is frozen. An offline screen over rollout snapshots showed that
TEMPORAL sharpness structure — especially ``last_first_sharp_ratio`` (sharpness of
the last frame vs the first) — discriminates a prompt's n samples ~5× better than
HPS, on the exact dimension HPS-frame-0 is blind to (motion / last-frame collapse).

This reward exposes that signal. It is deliberately narrow and gameable (a policy
could reward-hack it with sharpening noise / flicker) — so it MUST be read as a
diagnostic: watch group_std↑ → grad_norm↑ → ratio unpins, AND watch the samples
for hacking. Keep HPS as a logging-only metric (weight 0) to watch whether
optimizing for quality sacrifices alignment.

Score (v1, strongest single signal): ``clamp(last_first_sharp_ratio, 0, RATIO_CAP)``
plus an optional ``w_sharp * mean_sharpness`` term (``w_sharp`` default 0 → pure
ratio). Run with ``reward_model.normalize=False`` so the raw per-sample score
reaches the driver's per-prompt z-score (a redundant worker-side z-score would
flatten the group signal before it reaches the driver).
"""

import logging

import torch

from teleboost.reward.contract import BaseRewardModel
from teleboost.reward.registry import RewardRegistry

logger = logging.getLogger(__name__)

RATIO_CAP = 10.0  # clamp last/first sharpness ratio so a near-zero first frame can't explode the reward


@RewardRegistry.register("temporal_quality")
class TemporalQualityRewardModel(BaseRewardModel):
    """Diagnostic within-group signal from temporal sharpness structure."""

    REWARD_KEY = "tempqual_rewards"

    def init_model(self) -> None:
        # No checkpoint, no network. Just read the (optional) sharpness weight.
        extra = self.config.extra_config or {}
        self._w_sharp = float(extra.get("w_sharp", 0.0))
        logger.info(f"[temporal_quality] DIAGNOSTIC reward (no checkpoint, ignores caption); score = clamp(last/first sharpness, 0, {RATIO_CAP}) + {self._w_sharp}*mean_sharpness")

    @staticmethod
    def _frame_sharpness(gray: torch.Tensor) -> torch.Tensor:
        # gray: (T, H, W) -> per-frame Laplacian (4-neighbour) variance = sharpness, (T,)
        lap = -4.0 * gray[:, 1:-1, 1:-1] + gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1] + gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:]
        return lap.reshape(lap.shape[0], -1).var(dim=1)

    def compute_single_score(self, video_frames: torch.Tensor, caption=None) -> float:
        # video_frames: (T, C, H, W). caption intentionally ignored (quality, not alignment).
        f = video_frames.detach().to(torch.float32)
        if f.dim() != 4 or f.shape[1] < 3:
            raise ValueError(f"temporal_quality expects (T,C,H,W) with C>=3, got {tuple(f.shape)}")
        if float(f.max()) > 1.5:  # accept 0..255 or 0..1
            f = f / 255.0
        gray = 0.299 * f[:, 0] + 0.587 * f[:, 1] + 0.114 * f[:, 2]  # (T,H,W)
        if gray.shape[0] < 2:
            return 1.0  # single frame: no temporal signal, neutral
        sharp = self._frame_sharpness(gray)  # (T,)
        eps = 1e-8
        last_first = float((sharp[-1] / (sharp[0] + eps)).clamp(0.0, RATIO_CAP).item())
        score = last_first
        if self._w_sharp:
            score += self._w_sharp * float(sharp.mean().item())
        return score
