# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Wan attention operations.

This module owns the compute path used by the upstream Wan adapter.  Lifecycle
patching of the upstream namespace remains in ``teleboost.models.wan.attention.runtime``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from teleboost.models.wan.attention.backend import (
    resolve_wan_attention_backend,
    wan_flash_varlen_attention,
)


def _requested_backend(version: int | None) -> str | None:
    if version is None:
        return None
    if version == 2:
        return "flash_attn_2"
    if version == 3:
        return "flash_attn_3"
    raise ValueError(f"unknown FlashAttention version: {version!r}")


def _validate_lengths(
    lengths,
    *,
    batch: int,
    maximum: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(lengths, dtype=torch.int32, device=device)
    if result.ndim != 1 or result.numel() != batch:
        raise ValueError(f"{name} must contain one length per batch item")
    if bool(torch.any(result < 0)) or bool(torch.any(result > maximum)):
        raise ValueError(f"{name} entries must be within [0, {maximum}]")
    return result


def _pack_prefixes(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat([sample[:length] for sample, length in zip(tensor, lengths.tolist(), strict=True)])


def _packed_to_padded(
    packed: torch.Tensor,
    lengths: torch.Tensor,
    *,
    batch: int,
    maximum: int,
) -> torch.Tensor:
    """Restore a packed varlen result without assuming equal query lengths."""

    if int(lengths.sum().item()) != packed.shape[0]:
        raise RuntimeError("FlashAttention returned an unexpected packed token count")
    padded = packed.new_zeros((batch, maximum, *packed.shape[1:]))
    offset = 0
    for batch_index, length in enumerate(lengths.tolist()):
        padded[batch_index, :length] = packed[offset : offset + length]
        offset += length
    return padded


def _sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens: torch.Tensor,
    k_lens: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    q_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mask-correct PyTorch fallback for Wan's prefix-length contract."""

    batch, query_length = q.shape[:2]
    key_length = k.shape[1]
    output_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale

    query_index = torch.arange(query_length, device=q.device)[None, :, None]
    key_index = torch.arange(key_length, device=k.device)[None, None, :]
    query_valid = query_index[..., 0] < q_lens[:, None]
    key_valid = key_index[:, 0, :] < k_lens[:, None]

    attention_mask: torch.Tensor | None = key_valid[:, None, None, :]
    is_causal = False
    full_lengths = bool(torch.all(q_lens == query_length)) and bool(torch.all(k_lens == key_length))

    if not causal and full_lengths and window_size == (-1, -1):
        attention_mask = None
    elif causal and full_lengths and query_length == key_length and window_size == (-1, -1):
        attention_mask = None
        is_causal = True
    elif causal or window_size != (-1, -1):
        aligned_query = query_index + (k_lens - q_lens)[:, None, None]
        positional_mask = torch.ones(
            (batch, query_length, key_length),
            dtype=torch.bool,
            device=q.device,
        )
        if causal:
            positional_mask &= key_index <= aligned_query
        left, right = window_size
        if left >= 0:
            positional_mask &= key_index >= aligned_query - left
        if right >= 0:
            positional_mask &= key_index <= aligned_query + right
        attention_mask = positional_mask[:, None, :, :] & key_valid[:, None, None, :]

    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    working_dtype = dtype if q.device.type == "cuda" else q.dtype
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2).to(working_dtype),
        k.transpose(1, 2).to(working_dtype),
        v.transpose(1, 2).to(working_dtype),
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=softmax_scale,
        enable_gqa=q_heads != kv_heads,
    )
    output = output * query_valid[:, None, :, None]
    return output.transpose(1, 2).contiguous().to(output_dtype)


def _wan_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Wan attention expects q/k/v shaped [batch, sequence, heads, dim]")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("Wan attention requires matching q/k/v batch and k/v sequence dimensions")
    if q.shape[2] % k.shape[2] != 0 or k.shape[2] != v.shape[2]:
        raise ValueError("Wan attention requires q heads divisible by matching k/v heads")

    batch, query_length = q.shape[:2]
    key_length = k.shape[1]
    output_dtype = q.dtype
    q_lengths = _validate_lengths(
        [query_length] * batch if q_lens is None else q_lens,
        batch=batch,
        maximum=query_length,
        name="q_lens",
        device=q.device,
    )
    k_lengths = _validate_lengths(
        [key_length] * batch if k_lens is None else k_lens,
        batch=batch,
        maximum=key_length,
        name="k_lens",
        device=k.device,
    )

    requested = _requested_backend(version)
    resolved = resolve_wan_attention_backend(
        requested,
        tensor=q,
        dropout_p=dropout_p,
    )
    if resolved == "sdpa":
        return _sdpa_varlen_attention(
            q,
            k,
            v,
            q_lens=q_lengths,
            k_lens=k_lengths,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError("FlashAttention requires dtype torch.float16 or torch.bfloat16")
    packed_q = _pack_prefixes(q, q_lengths)
    packed_k = _pack_prefixes(k, k_lengths)
    packed_v = _pack_prefixes(v, k_lengths)
    packed_q = packed_q if packed_q.dtype in half_dtypes else packed_q.to(dtype)
    packed_k = packed_k if packed_k.dtype in half_dtypes else packed_k.to(dtype)
    packed_v = packed_v if packed_v.dtype in half_dtypes else packed_v.to(dtype)
    packed_q = packed_q.to(packed_v.dtype)
    packed_k = packed_k.to(packed_v.dtype)
    if q_scale is not None:
        packed_q = packed_q * q_scale

    cu_q = F.pad(torch.cumsum(q_lengths, dim=0, dtype=torch.int32), (1, 0))
    cu_k = F.pad(torch.cumsum(k_lengths, dim=0, dtype=torch.int32), (1, 0))
    packed_output, actual_backend = wan_flash_varlen_attention(
        packed_q,
        packed_k,
        packed_v,
        cu_q,
        cu_k,
        query_length,
        key_length,
        backend=resolved,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    if packed_output is None:
        raise RuntimeError(f"Wan FlashAttention unexpectedly resolved to {actual_backend}")
    return _packed_to_padded(
        packed_output,
        q_lengths,
        batch=batch,
        maximum=query_length,
    ).to(output_dtype)


def wan_flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Drop-in replacement for ``wan.modules.attention.flash_attention``."""

    return _wan_attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=version,
    )


def wan_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Drop-in replacement for ``wan.modules.attention.attention``."""

    return _wan_attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )


__all__ = ["_packed_to_padded", "wan_attention", "wan_flash_attention"]
