# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VIPO: Visual Preference Policy Optimization for visual generation.

Paper: arxiv 2511.18719 — "Seeing What Matters: Visual Preference
Policy Optimization for Visual Generation".

Implementation scope
--------------------
TeleBoost implements VIPO's structured-advantage path: a DINOv2-derived
allocation map ``M(p)`` is used to form dense advantages
``A^p = M(p) · A``. The concrete map-construction defaults in this
repo — PCA pooling, reverse normalization, Gaussian smoothing, and
resize behavior — are TeleBoost implementation choices. Treat them as
this repository's reproducibility baseline, not as a claim of
bit-for-bit official paper reproduction. The map is normalized to
``[0, 1]`` on the success path; the baseline scalar GRPO path remains
recoverable by disabling VIPO or via the all-ones fallback when map
computation errors.

Paper-faithful pieces
---------------------
* DINOv2 feature extraction → mean-centered PCA → per-pixel semantic
  importance map (paper §3 "allocation map ``M(p)``", Eq. 5).
* Variance-weighted aggregation of the top-K PCs — each PC weighted by
  its explained-variance ratio (paper Eq. 6:
  ``S = Reshape(Σ_j λ_j z'_j)``).  ``pca_method="weighted"`` (the
  default) implements this.
* Dense advantage broadcast: ``A^p = M(p) · A`` (paper Eq. 8 — the
  spatially-resolved advantage is the allocation map times the scalar
  group advantage).  Implemented in ``_apply_vipo_broadcast``
  (``teleboost/programs/wan/vipo.py``) and consumed by the actor's per-pixel policy loss.

In-house design choices (NOT from paper)
----------------------------------------
* The ``"first_pc"`` and ``"average"`` pooling schemes are in-house
  ablations kept for parity with reference implementations; the paper's
  aggregation is the variance-weighted one (``"weighted"``, above).
* The reverse-mapping ``(max − x) / (max − min)`` over PC values, the
  top-K count (default K = 3), the default Gaussian σ = 1.0 smoothing,
  and the bilinear resize to latent resolution are convention-driven;
  the paper does not pin them.  They are documented as in-house choices,
  not paper claims.

Pipeline
--------
1. The diffusion rollout decodes generated videos to ``(B, C, T, H, W)``.
2. :func:`compute_batch_pixel_weight_maps` runs DINOv2 on each frame and
   collapses patch features into a per-pixel allocation map at the
   latent resolution ``(B, T_lat, H_lat, W_lat)``.
3. The trainer's ``_apply_vipo_broadcast`` (``teleboost/programs/wan/vipo.py``) multiplies the
   scalar GRPO advantage by that map, producing a dense advantage that
   the actor uses to weight the per-pixel policy loss.

When ``actor_rollout_ref.pixel_weight.enable`` is False the rollout
attaches no weight map and the mixin is a no-op, so the trainer falls
back to the baseline scalar GRPO path bit-for-bit identical to before.

Design notes
------------
* The DINOv2 model + processor are cached per ``(model_path, device)`` so
  each Ray worker pays the load cost once.
* All public entry points accept ``videos`` in shape ``(B, C, T, H, W)``
  (channels-first, batch-first). A ``ValueError`` is raised otherwise.
* ``pca_method`` controls how the top-K principal components are folded
  into a scalar weight per patch:
    - ``"weighted"``  : variance-weighted sum of reverse-normalized PCs
      (default; paper Eq. 6 aggregation)
    - ``"first_pc"``  : negated first PC only (matches several reference
      implementations; recommended when aligning with prior work)
    - ``"average"``   : negated mean of the top-K PCs
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _normalize_tensor_or_ones(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a weight map without destroying the GRPO fallback.

    A constant map carries no spatial preference.  VIPO defines that case as
    the baseline scalar-GRPO path, represented by an all-ones map.  The old
    ``(x - min) / (max - min + eps)`` expression turned an all-ones error
    fallback into all zeros and therefore silently removed the policy loss.
    """
    min_value = values.min()
    value_range = values.max() - min_value
    if not torch.isfinite(value_range) or float(value_range.detach().cpu()) <= eps:
        return torch.ones_like(values, dtype=torch.float32)
    return ((values - min_value) / value_range).to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# DINOv2 caching + helpers
# ---------------------------------------------------------------------------


_DINO_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _get_device(device):
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_dinov2(model_path: str, device: torch.device):
    cache_key = (model_path, str(device))
    cached = _DINO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    _DINO_CACHE[cache_key] = (processor, model)
    return processor, model


def _compute_top3_pca(features: np.ndarray):
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = centered @ right_vectors[:3].T
    if components.shape[1] < 3:
        pad_width = 3 - components.shape[1]
        components = np.pad(components, ((0, 0), (0, pad_width)))
        singular_values = np.pad(singular_values, (0, pad_width))
    explained = singular_values[:3] ** 2
    explained = explained / (explained.sum() + 1e-8)
    return components[:, :3], explained[:3]


def _normalize_unit_interval(values: np.ndarray):
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val <= min_val:
        return np.full_like(values, 0.5, dtype=np.float32)
    normalized = (values - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def _reverse_semantic_weights(pca_features: np.ndarray, explained: np.ndarray, method: str):
    if method == "weighted":
        semantic_weights = np.zeros(pca_features.shape[0], dtype=np.float32)
        for comp_idx in range(min(3, pca_features.shape[1])):
            comp_weights = pca_features[:, comp_idx]
            min_val = float(comp_weights.min())
            max_val = float(comp_weights.max())
            if max_val > min_val:
                remapped = (max_val - comp_weights) / (max_val - min_val)
            else:
                remapped = np.full_like(comp_weights, 0.5, dtype=np.float32)
            semantic_weights += explained[comp_idx] * remapped.astype(np.float32)
        return np.clip(semantic_weights, 0.0, 1.0)

    if method == "average":
        return _normalize_unit_interval(-np.mean(pca_features, axis=1))

    if method == "first_pc":
        return _normalize_unit_interval(-pca_features[:, 0])

    raise ValueError(f"Unknown pca_method: {method}")


def _best_patch_grid(num_patches: int, processed_height: int, processed_width: int):
    patch_size = 14
    grid_h = max(processed_height // patch_size, 1)
    grid_w = max(processed_width // patch_size, 1)
    if grid_h * grid_w == num_patches:
        return grid_h, grid_w

    target_ratio = processed_height / max(processed_width, 1)
    candidates = []
    limit = int(math.sqrt(num_patches)) + 32
    for height in range(1, max(limit, 2)):
        if num_patches % height == 0:
            width = num_patches // height
            candidates.append((height, width, abs((height / max(width, 1)) - target_ratio)))
    if not candidates:
        side = int(round(math.sqrt(num_patches)))
        return max(side, 1), max(side, 1)
    best_h, best_w, _ = min(candidates, key=lambda item: item[2])
    return best_h, best_w


def _reshape_patch_weights(semantic_weights: np.ndarray, grid_h: int, grid_w: int):
    if semantic_weights.size == grid_h * grid_w:
        return semantic_weights.reshape(grid_h, grid_w)

    temp_grid = max(int(round(math.sqrt(semantic_weights.size))), 1)
    padded = np.full(temp_grid * temp_grid, semantic_weights.mean(), dtype=np.float32)
    padded[: semantic_weights.size] = semantic_weights.astype(np.float32)
    temp_map = padded.reshape(temp_grid, temp_grid)
    temp_tensor = torch.from_numpy(temp_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        temp_tensor,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _smooth_weight_map(weight_map: np.ndarray, sigma: float):
    if sigma <= 0:
        return _normalize_unit_interval(weight_map)

    try:
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(weight_map, sigma=sigma)
        return _normalize_unit_interval(smoothed)
    except Exception:
        # scipy unavailable: average-pool approximation. Slightly different
        # semantics but never crashes training.
        weight_tensor = torch.from_numpy(weight_map).float().unsqueeze(0).unsqueeze(0)
        kernel = max(int(round(sigma * 4)) * 2 + 1, 3)
        smoothed = F.avg_pool2d(weight_tensor, kernel_size=kernel, stride=1, padding=kernel // 2)
        return _normalize_unit_interval(smoothed.squeeze(0).squeeze(0).cpu().numpy())


# ---------------------------------------------------------------------------
# Public compute API
# ---------------------------------------------------------------------------


def compute_dinov2_feature_map_reverse(
    video_frames: torch.Tensor,
    target_size=(64, 64),
    target_time=7,
    device=None,
    model_path="facebook/dinov2-large",
    pca_method="weighted",
    sigma=1.0,
):
    """Compute a DINOv2-based pixel-weight map for one video.

    Parameters
    ----------
    video_frames : torch.Tensor
        Shape ``(C, T, H, W)``.  Values in [-1, 1] or [0, 1].
    target_size : tuple of int
        Spatial size of the output map ``(H_lat, W_lat)``.
    target_time : int
        Temporal size of the output map ``T_lat``.
    device : str or torch.device, optional
        Device on which DINOv2 runs.  Defaults to the input tensor's device.
    model_path : str
        Hugging Face repo ID or local path for the DINOv2 model.
    pca_method : str
        ``"weighted"`` | ``"first_pc"`` | ``"average"``.
    sigma : float
        Gaussian smoothing sigma.  ``0`` disables smoothing.

    Returns
    -------
    torch.Tensor
        Shape ``(T_lat, H_lat, W_lat)``, values in [0, 1].
    """
    if video_frames.min() < 0:
        video_frames = (video_frames + 1.0) / 2.0
    video_frames = torch.clamp(video_frames, 0, 1).float()

    device = _get_device(device or video_frames.device)
    processor, model = _load_dinov2(model_path, device)

    _, time_steps, _, _ = video_frames.shape
    frame_feature_maps = []

    for frame_index in range(time_steps):
        frame = video_frames[:, frame_index]
        try:
            with torch.no_grad():
                from PIL import Image

                frame_np = frame.permute(1, 2, 0).detach().cpu().numpy()
                frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                original_width, original_height = frame_pil.size

                try:
                    inputs = processor(
                        images=frame_pil,
                        return_tensors="pt",
                        size={"height": original_height, "width": original_width},
                        do_center_crop=False,
                        do_resize=True,
                    ).to(device)
                except Exception:
                    shortest_edge = min(max(min(frame_pil.size), 224), 518)
                    inputs = processor(
                        images=frame_pil,
                        return_tensors="pt",
                        size={"shortest_edge": shortest_edge},
                        do_center_crop=False,
                        do_resize=True,
                    ).to(device)

                processed_tensor = inputs["pixel_values"][0]
                processed_height = int(processed_tensor.shape[1])
                processed_width = int(processed_tensor.shape[2])

                outputs = model(**inputs)
                features = outputs.last_hidden_state[0, 1:, :].detach().cpu().numpy()
                grid_h, grid_w = _best_patch_grid(features.shape[0], processed_height, processed_width)
                pca_features, explained = _compute_top3_pca(features)
                semantic_weights = _reverse_semantic_weights(pca_features, explained, pca_method)
                weight_map = _reshape_patch_weights(semantic_weights, grid_h, grid_w)
                smoothed = _smooth_weight_map(weight_map, sigma)

                target_h, target_w = target_size
                resized = F.interpolate(
                    torch.from_numpy(smoothed).float().unsqueeze(0).unsqueeze(0).to(device),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                feature_map = resized.squeeze(0).squeeze(0)
                feature_map = _normalize_tensor_or_ones(feature_map)
        except Exception as exc:
            # An all-ones map means "treat every pixel equally" and degrades
            # VIPO to baseline GRPO for this frame.  Keep the warning so a
            # broken DINO installation cannot remain invisible for a run.
            logger.warning("VIPO map failed for frame %s; using uniform weights: %s", frame_index, exc)
            feature_map = torch.ones(target_size, dtype=torch.float32, device=device)

        frame_feature_maps.append(feature_map.to(dtype=torch.float32))

    feature_maps = torch.stack(frame_feature_maps, dim=0)

    if feature_maps.shape[0] != target_time:
        feature_maps = (
            F.interpolate(
                feature_maps.unsqueeze(0).unsqueeze(0),
                size=(target_time, feature_maps.shape[1], feature_maps.shape[2]),
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    return _normalize_tensor_or_ones(feature_maps)


def compute_batch_pixel_weight_maps(
    videos: torch.Tensor,
    target_size,
    target_time,
    device=None,
    model_path="facebook/dinov2-large",
    pca_method="weighted",
    sigma=1.0,
):
    """Vectorize :func:`compute_dinov2_feature_map_reverse` over a batch.

    Parameters
    ----------
    videos : torch.Tensor
        Shape ``(B, C, T, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, T_lat, H_lat, W_lat)``, values in [0, 1].
    """
    if videos.ndim != 5:
        raise ValueError(f"Expected videos with shape (B, C, T, H, W), got {tuple(videos.shape)}")

    maps = []
    compute_device = _get_device(device or videos.device)
    for video in videos:
        maps.append(
            compute_dinov2_feature_map_reverse(
                video_frames=video,
                target_size=target_size,
                target_time=target_time,
                device=compute_device,
                model_path=model_path,
                pca_method=pca_method,
                sigma=sigma,
            )
        )
    return torch.stack(maps, dim=0)


# Backward-compatible aliases for code ported from the VIPO fork.
compute_dinov2_feature_map_reverse_pixel = compute_dinov2_feature_map_reverse
compute_batch_pixel_weight_maps_pixel = compute_batch_pixel_weight_maps


# ---------------------------------------------------------------------------
# Trainer mixin
# ---------------------------------------------------------------------------


__all__ = [
    "compute_batch_pixel_weight_maps",
    "compute_dinov2_feature_map_reverse",
    "compute_batch_pixel_weight_maps_pixel",
    "compute_dinov2_feature_map_reverse_pixel",
]
