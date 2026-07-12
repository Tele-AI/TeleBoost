#!/usr/bin/env python3
# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""One-H100 numerical/gradient proof for TeleBoost's Wan FA3 integration."""

from __future__ import annotations

import json
from importlib.metadata import version

import torch
import torch.nn.functional as F

from teleboost.models.wan.attention.backend import (
    resolve_wan_attention_backend,
    wan_dense_attention,
    wan_flash_varlen_attention,
)
from teleboost.models.wan.attention.runtime import wan_flash_attention


def _finite_gradients(*tensors: torch.Tensor) -> bool:
    return all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in tensors)


def _clone_inputs(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


def _bottom_right_mask(query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
    query = torch.arange(query_length, device=device)[:, None]
    key = torch.arange(key_length, device=device)[None, :]
    return key <= query + (key_length - query_length)


def _dense_case(dtype: torch.dtype) -> dict:
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for shape in ((2, 7, 4, 128), (2, 9, 2, 128), (2, 9, 2, 128)))
    q3, k3, v3 = _clone_inputs(q, k, v)
    out3 = wan_dense_attention(q3, k3, v3, backend="flash_attn_3", causal=True)
    out3.float().square().mean().backward()
    if not _finite_gradients(q3, k3, v3):
        raise RuntimeError(f"FA3 dense {dtype} produced non-finite gradients")

    q2, k2, v2 = _clone_inputs(q, k, v)
    out2 = wan_dense_attention(q2, k2, v2, backend="flash_attn_2", causal=True)
    out2.float().square().mean().backward()
    if not _finite_gradients(q2, k2, v2):
        raise RuntimeError(f"FA2 dense {dtype} produced non-finite gradients")

    mask = _bottom_right_mask(7, 9, q.device)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        enable_gqa=True,
    ).transpose(1, 2)
    tolerance = 4e-2 if dtype == torch.bfloat16 else 2e-2
    torch.testing.assert_close(out3, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(out3, out2, atol=tolerance, rtol=tolerance)

    local = wan_dense_attention(q, k, v, backend="flash_attn_3", window_size=(2, 1))
    query = torch.arange(7, device=q.device)[:, None] + 2
    key = torch.arange(9, device=q.device)[None, :]
    local_mask = (key >= query - 2) & (key <= query + 1)
    local_reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=local_mask,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(local, local_reference, atol=tolerance, rtol=tolerance)
    return {"dtype": str(dtype), "shape": list(out3.shape), "gradients": "finite"}


def _varlen_case() -> dict:
    q_lens = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    k_lens = torch.tensor([6, 4], device="cuda", dtype=torch.int32)
    cu_q = F.pad(q_lens.cumsum(0, dtype=torch.int32), (1, 0))
    cu_k = F.pad(k_lens.cumsum(0, dtype=torch.int32), (1, 0))
    q = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(10, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(10, 2, 128, device="cuda", dtype=torch.bfloat16)

    outputs = {}
    upstream_grad = torch.randn_like(q)
    for backend in ("flash_attn_3", "flash_attn_2"):
        qi, ki, vi = _clone_inputs(q, k, v)
        output, actual = wan_flash_varlen_attention(
            qi,
            ki,
            vi,
            cu_q,
            cu_k,
            5,
            6,
            backend=backend,
            causal=True,
            deterministic=True,
        )
        if output is None or actual != backend:
            raise RuntimeError(f"requested {backend}, resolved {actual}")
        output.backward(upstream_grad)
        if not _finite_gradients(qi, ki, vi):
            raise RuntimeError(f"{backend} varlen produced non-finite gradients")
        outputs[backend] = output.detach()

    torch.testing.assert_close(
        outputs["flash_attn_3"],
        outputs["flash_attn_2"],
        atol=4e-2,
        rtol=4e-2,
    )

    repeated = []
    for _ in range(2):
        qi, ki, vi = _clone_inputs(q, k, v)
        output, _ = wan_flash_varlen_attention(
            qi,
            ki,
            vi,
            cu_q,
            cu_k,
            5,
            6,
            backend="flash_attn_3",
            causal=True,
            deterministic=True,
        )
        assert output is not None
        output.backward(upstream_grad)
        repeated.append(tuple(tensor.grad.detach().clone() for tensor in (qi, ki, vi)))
    if not all(torch.equal(left, right) for left, right in zip(*repeated, strict=True)):
        raise RuntimeError("FA3 deterministic varlen backward was not bit-identical")

    qp = torch.randn(2, 5, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kp = torch.randn(2, 6, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    vp = torch.randn(2, 6, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    padded = wan_flash_attention(
        qp,
        kp,
        vp,
        q_lens=q_lens,
        k_lens=k_lens,
        causal=True,
        deterministic=True,
        version=3,
    )
    if torch.count_nonzero(padded[0, 3:]) != 0:
        raise RuntimeError("Wan padded adapter did not zero invalid query rows")
    padded.float().square().mean().backward()
    if not _finite_gradients(qp, kp, vp):
        raise RuntimeError("Wan padded FA3 adapter produced non-finite gradients")
    return {"packed_shape": list(q.shape), "padded_shape": list(padded.shape), "gradients": "finite"}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(0)
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise SystemExit(f"validated Wan FA3 profile requires SM90, found {capability}")
    installed = version("flash-attn-3")
    if not installed.startswith("3.0.0+teleboost.wan.sm90"):
        raise SystemExit(f"unexpected flash-attn-3 build: {installed}")

    probe = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    if resolve_wan_attention_backend(None, tensor=probe) != "flash_attn_3":
        raise RuntimeError("Wan auto policy did not select FA3 on SM90")
    if resolve_wan_attention_backend(None, tensor=probe, dropout_p=0.1) != "flash_attn_2":
        raise RuntimeError("Wan dropout policy did not select FA2")

    torch.manual_seed(20260711)
    result = {
        "flash-attn-3": installed,
        "flash-attn-2": version("flash-attn"),
        "device": torch.cuda.get_device_name(),
        "dense": [_dense_case(torch.bfloat16), _dense_case(torch.float16)],
        "varlen": _varlen_case(),
    }
    torch.cuda.synchronize()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
