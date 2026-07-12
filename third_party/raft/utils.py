# Copyright (c) 2020 princeton-vl (RAFT).
# Modifications Copyright 2025-2026 TeleAI and the TeleBoost contributors.
#
# The RAFT-authored flow utilities below are licensed under BSD-3-Clause;
# see LICENSE in this directory. TeleAI modifications are licensed under
# Apache-2.0; see LICENSE at the repository root.

"""Tensor utilities owned by the vendored RAFT runtime."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class InputPadder:
    """Pad images so their spatial dimensions are divisible by eight."""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, tensor):
        height, width = tensor.shape[-2:]
        crop = [
            self._pad[2],
            height - self._pad[3],
            self._pad[0],
            width - self._pad[1],
        ]
        return tensor[..., crop[0] : crop[1], crop[2] : crop[3]]


def bilinear_sampler(image, coords, mode="bilinear", mask=False):
    """Sample ``image`` with pixel-space coordinates."""

    height, width = image.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (width - 1) - 1
    ygrid = 2 * ygrid / (height - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    sampled = F.grid_sample(image, grid, align_corners=True)

    if mask:
        valid = (
            (xgrid > -1)
            & (ygrid > -1)
            & (xgrid < 1)
            & (ygrid < 1)
        )
        return sampled, valid.float()
    return sampled


def coords_grid(batch, height, width, device):
    coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(
        flow,
        size=new_size,
        mode=mode,
        align_corners=True,
    )


__all__ = ["InputPadder", "bilinear_sampler", "coords_grid", "upflow8"]
