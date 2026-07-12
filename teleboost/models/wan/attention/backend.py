"""Unified FlashAttention selection for Wan and Wan-TeleTron paths.

Wan uses dense attention or padding represented by sequence lengths, not an
arbitrary additive/block mask.  The default ``auto`` policy is therefore:
FA3 on Hopper when the call is supported, then FA2, then mask-correct SDPA.
Explicit backend requests fail rather than silently changing kernels.
"""

from __future__ import annotations

import os
from types import ModuleType

import torch
import torch.nn.functional as F

_FA3_NAMES = {"3", "fa3", "flash3", "flash_attn_3"}
_FA2_NAMES = {"2", "fa2", "flash", "flash2", "flash_attn", "flash_attn_2"}
_SDPA_NAMES = {"none", "native", "sdpa", "torch"}


def _import_fa3() -> ModuleType | None:
    try:
        from flash_attn_3 import flash_attn_interface

        return flash_attn_interface
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn_3":
            raise RuntimeError("flash-attn-3 is installed but its extension cannot be imported") from exc
    try:
        import flash_attn_interface

        return flash_attn_interface
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn_interface":
            raise RuntimeError("flash-attn-3 compatibility module is broken") from exc
        return None


def _import_fa2() -> ModuleType | None:
    try:
        import flash_attn

        return flash_attn
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn":
            raise RuntimeError("flash-attn is installed but its extension cannot be imported") from exc
        return None


_UNSET = object()
_FA3: ModuleType | None | object = _UNSET
_FA2: ModuleType | None | object = _UNSET


def _get_fa3() -> ModuleType | None:
    global _FA3
    if _FA3 is _UNSET:
        _FA3 = _import_fa3()
    return _FA3  # type: ignore[return-value]


def _get_fa2() -> ModuleType | None:
    global _FA2
    if _FA2 is _UNSET:
        _FA2 = _import_fa2()
    return _FA2  # type: ignore[return-value]


def available_wan_attention_backends() -> tuple[str, ...]:
    available = []
    if _get_fa3() is not None:
        available.append("flash_attn_3")
    if _get_fa2() is not None:
        available.append("flash_attn_2")
    available.append("sdpa")
    return tuple(available)


def _normalize_backend(requested: str | int | None) -> str:
    value = str(requested if requested is not None else os.environ.get("TELEBOOST_WAN_ATTN_BACKEND", "auto")).strip().lower()
    if value == "auto":
        return value
    if value in _FA3_NAMES:
        return "flash_attn_3"
    if value in _FA2_NAMES:
        return "flash_attn_2"
    if value in _SDPA_NAMES:
        return "sdpa"
    raise ValueError(f"unknown Wan attention backend {value!r}; use auto|flash_attn_3|flash_attn_2|sdpa")


def _fa3_device_supported(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(tensor.device)
    # This integration is validated against the Hopper package on SM90. Do
    # not silently classify a newer architecture as Hopper merely by using a
    # numeric >= comparison; extend this allowlist after kernel validation.
    return (major, minor) == (9, 0)


def _fa2_device_supported(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda"


def resolve_wan_attention_backend(
    requested: str | int | None,
    *,
    tensor: torch.Tensor,
    dropout_p: float = 0.0,
    has_arbitrary_mask: bool = False,
) -> str:
    backend = _normalize_backend(requested)
    if backend == "sdpa":
        return backend
    if has_arbitrary_mask:
        if backend != "auto":
            raise ValueError(f"Wan backend {backend} does not accept an arbitrary attention mask")
        return "sdpa"

    if backend == "flash_attn_3":
        if not _fa3_device_supported(tensor):
            raise RuntimeError("Wan FA3 requires a validated Hopper SM90 CUDA tensor")
        if float(dropout_p) != 0.0:
            raise ValueError("Wan FA3 does not support attention dropout")
        fa3 = _get_fa3()
        if fa3 is None:
            raise RuntimeError("Wan explicitly requested FA3 but flash-attn-3 is not installed")
        return backend
    if backend == "flash_attn_2":
        if _get_fa2() is None:
            raise RuntimeError("Wan explicitly requested FA2 but flash-attn is not installed")
        if not _fa2_device_supported(tensor):
            raise RuntimeError("Wan FA2 requires a CUDA tensor")
        return backend

    # Auto probes an optional extension only on a device/call it can actually
    # serve. CPU and non-SM90 paths therefore never import the Hopper module.
    if _fa3_device_supported(tensor) and float(dropout_p) == 0.0 and _get_fa3() is not None:
        return "flash_attn_3"
    if _fa2_device_supported(tensor) and _get_fa2() is not None:
        return "flash_attn_2"
    return "sdpa"


def _output_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def wan_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str | int | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> torch.Tensor:
    """Run dense Wan attention on ``[batch, seq, heads, head_dim]`` tensors."""

    half_dtypes = (torch.float16, torch.bfloat16)
    normalized = _normalize_backend(backend)
    if any(tensor.dtype not in half_dtypes for tensor in (q, k, v)):
        if normalized != "auto" and normalized != "sdpa":
            raise ValueError(f"Wan {normalized} requires q/k/v in float16 or bfloat16; use backend=sdpa for full-precision attention")
        # Dense TeleTron callers do not expose the upstream Wan wrapper's
        # explicit dtype-cast argument. Preserve FP32 rather than silently
        # lowering precision merely because an optional CUDA kernel exists.
        backend = "sdpa"

    if q_scale is not None:
        q = q * q_scale
    resolved = resolve_wan_attention_backend(
        backend,
        tensor=q,
        dropout_p=dropout_p,
        has_arbitrary_mask=attn_mask is not None,
    )
    if resolved == "flash_attn_3":
        fa3 = _get_fa3()
        assert fa3 is not None
        return _output_tensor(
            fa3.flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        )
    if resolved == "flash_attn_2":
        fa2 = _get_fa2()
        assert fa2 is not None
        return _output_tensor(
            fa2.flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        )

    if window_size != (-1, -1):
        raise ValueError("Wan SDPA fallback does not implement sliding-window attention")
    if causal and attn_mask is not None:
        raise ValueError("Wan SDPA requires the caller to combine causal and arbitrary masks")
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
        enable_gqa=q_heads != kv_heads,
    )
    return out.transpose(1, 2).contiguous()


def wan_flash_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    backend: str | int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> tuple[torch.Tensor | None, str]:
    """Run packed varlen FA3/FA2, or return ``(None, 'sdpa')`` for fallback."""

    resolved = resolve_wan_attention_backend(
        backend,
        tensor=q,
        dropout_p=dropout_p,
    )
    if resolved == "flash_attn_3":
        fa3 = _get_fa3()
        assert fa3 is not None
        out = fa3.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            seqused_q=None,
            seqused_k=None,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        return _output_tensor(out), resolved
    if resolved == "flash_attn_2":
        fa2 = _get_fa2()
        assert fa2 is not None
        out = fa2.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        return _output_tensor(out), resolved
    return None, resolved
