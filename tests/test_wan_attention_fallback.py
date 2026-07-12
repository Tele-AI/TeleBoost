# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Wan must remain correct when the optional FlashAttention build is absent."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from teleboost import apply_runtime_patches
from teleboost.models.wan.attention.runtime import install_wan_attention_adapter


def _load_wan_modules():
    apply_runtime_patches()

    import wan.modules.attention as attention_module
    from wan.modules.model import WanT2VCrossAttention

    return attention_module, WanT2VCrossAttention


attention_module, WanT2VCrossAttention = _load_wan_modules()


@pytest.fixture(autouse=True)
def _wan_adapter(monkeypatch):
    monkeypatch.setenv("TELEBOOST_WAN_ATTN_BACKEND", "sdpa")
    handle = install_wan_attention_adapter(namespace="wan")
    yield
    handle.uninstall()


def test_sdpa_fallback_preserves_variable_length_mask():
    torch.manual_seed(7)
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 4, 2, 4)
    v = torch.randn(2, 4, 2, 4)
    q_lens = torch.tensor([2, 3])
    k_lens = torch.tensor([3, 1])

    actual = attention_module.attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dtype=torch.float32,
    )

    expected = torch.zeros_like(actual)
    for batch, (q_len, k_len) in enumerate(zip(q_lens, k_lens, strict=True)):
        q_len = int(q_len)
        k_len = int(k_len)
        expected[batch, :q_len] = F.scaled_dot_product_attention(
            q[batch, :q_len].transpose(0, 1),
            k[batch, :k_len].transpose(0, 1),
            v[batch, :k_len].transpose(0, 1),
        ).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_wan_cross_attention_uses_sdpa_without_flash_attn():
    layer = WanT2VCrossAttention(dim=8, num_heads=2).eval()
    x = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 8)

    output = layer(x, context, torch.tensor([4, 2]))

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_sdpa_fallback_uses_per_sample_bottom_right_causal_mask():
    torch.manual_seed(11)
    q = torch.randn(2, 3, 4, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 4)
    q_lens = torch.tensor([2, 3])
    k_lens = torch.tensor([5, 3])

    actual = attention_module.attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        causal=True,
        dtype=torch.float32,
    )

    expected = torch.zeros_like(actual)
    for batch, (q_len, k_len) in enumerate(zip(q_lens, k_lens, strict=True)):
        q_len = int(q_len)
        k_len = int(k_len)
        query_index = torch.arange(q_len)[:, None]
        key_index = torch.arange(k_len)[None, :]
        bottom_right_causal = key_index <= query_index + (k_len - q_len)
        expected[batch, :q_len] = F.scaled_dot_product_attention(
            q[batch, :q_len].transpose(0, 1),
            k[batch, :k_len].transpose(0, 1),
            v[batch, :k_len].transpose(0, 1),
            attn_mask=bottom_right_causal,
            enable_gqa=True,
        ).transpose(0, 1)

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[0, 2:]) == 0
