from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import teleboost.models.wan.attention.backend as backend_module
from teleboost.models.wan.attention.ops import _packed_to_padded, wan_flash_attention


def _enable_fake_cuda(monkeypatch):
    monkeypatch.setattr(backend_module, "_fa3_device_supported", lambda _tensor: True)
    monkeypatch.setattr(backend_module, "_fa2_device_supported", lambda _tensor: True)


def test_wan_auto_prefers_fa3_and_filters_backend_kwargs(monkeypatch):
    calls = []

    def flash_attn_func(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q + 1, torch.zeros(())

    monkeypatch.setattr(backend_module, "_FA3", SimpleNamespace(flash_attn_func=flash_attn_func))
    monkeypatch.setattr(backend_module, "_FA2", SimpleNamespace())
    _enable_fake_cuda(monkeypatch)
    q = torch.zeros(1, 3, 4, 8, dtype=torch.bfloat16)

    output = backend_module.wan_dense_attention(
        q,
        q,
        q,
        window_size=(2, 3),
        deterministic=True,
    )

    torch.testing.assert_close(output, q + 1)
    kwargs = calls[0][3]
    assert kwargs["window_size"] == (2, 3)
    assert kwargs["deterministic"] is True
    assert "dropout_p" not in kwargs


def test_wan_auto_uses_fa2_when_dropout_is_nonzero(monkeypatch):
    fa3 = SimpleNamespace(flash_attn_func=lambda *_args, **_kwargs: pytest.fail("FA3 cannot implement dropout"))
    calls = []

    def flash_attn_func(q, _k, _v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(backend_module, "_FA3", fa3)
    monkeypatch.setattr(backend_module, "_FA2", SimpleNamespace(flash_attn_func=flash_attn_func))
    _enable_fake_cuda(monkeypatch)
    q = torch.zeros(1, 3, 2, 8, dtype=torch.bfloat16)

    backend_module.wan_dense_attention(q, q, q, dropout_p=0.25)

    assert calls[0]["dropout_p"] == 0.25
    with pytest.raises(ValueError, match="does not support attention dropout"):
        backend_module.wan_dense_attention(q, q, q, backend="fa3", dropout_p=0.25)


def test_wan_explicit_unavailable_backend_fails(monkeypatch):
    monkeypatch.setattr(backend_module, "_FA3", None)
    _enable_fake_cuda(monkeypatch)
    q = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="explicitly requested FA3"):
        backend_module.wan_dense_attention(q, q, q, backend="flash_attn_3")


def test_wan_cpu_auto_does_not_import_optional_cuda_extensions(monkeypatch):
    monkeypatch.setattr(backend_module, "_FA3", backend_module._UNSET)
    monkeypatch.setattr(backend_module, "_FA2", backend_module._UNSET)
    monkeypatch.setattr(
        backend_module,
        "_import_fa3",
        lambda: pytest.fail("CPU auto must not import the Hopper extension"),
    )
    monkeypatch.setattr(
        backend_module,
        "_import_fa2",
        lambda: pytest.fail("CPU auto must not import the CUDA extension"),
    )
    q = torch.randn(1, 2, 2, 4)

    assert backend_module.resolve_wan_attention_backend(None, tensor=q) == "sdpa"


def test_wan_sdpa_fallback_preserves_arbitrary_mask():
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    mask = torch.tensor([[[[True, True, False, False]]]])

    actual = backend_module.wan_dense_attention(q, k, v, backend="sdpa", attn_mask=mask)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected)


def test_wan_dense_auto_preserves_fp32_with_sdpa(monkeypatch):
    monkeypatch.setattr(
        backend_module,
        "_FA3",
        SimpleNamespace(flash_attn_func=lambda *_args, **_kwargs: pytest.fail("auto must not silently cast dense FP32 to FA3")),
    )
    monkeypatch.setattr(
        backend_module,
        "_FA2",
        SimpleNamespace(flash_attn_func=lambda *_args, **_kwargs: pytest.fail("auto must not silently cast dense FP32 to FA2")),
    )
    _enable_fake_cuda(monkeypatch)
    q = torch.randn(1, 3, 2, 4, dtype=torch.float32)

    output = backend_module.wan_dense_attention(q, q, q)

    assert output.dtype == torch.float32
    with pytest.raises(ValueError, match="requires q/k/v in float16 or bfloat16"):
        backend_module.wan_dense_attention(q, q, q, backend="flash_attn_3")


def test_wan_fa3_varlen_passes_lengths_window_and_determinism(monkeypatch):
    calls = []

    def flash_attn_varlen_func(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0]

    monkeypatch.setattr(
        backend_module,
        "_FA3",
        SimpleNamespace(flash_attn_varlen_func=flash_attn_varlen_func),
    )
    _enable_fake_cuda(monkeypatch)
    q = torch.randn(5, 4, 8)
    k = torch.randn(7, 2, 8)
    v = torch.randn(7, 2, 8)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)

    output, resolved = backend_module.wan_flash_varlen_attention(
        q,
        k,
        v,
        cu_q,
        cu_k,
        3,
        4,
        window_size=(1, 2),
        causal=True,
        deterministic=True,
    )

    assert output is q
    assert resolved == "flash_attn_3"
    args, kwargs = calls[0]
    assert args[3:7] == (cu_q, cu_k, 3, 4)
    assert kwargs["seqused_q"] is None and kwargs["seqused_k"] is None
    assert kwargs["window_size"] == (1, 2)
    assert kwargs["causal"] is True
    assert kwargs["deterministic"] is True
    assert "dropout_p" not in kwargs


def test_wan_varlen_repacking_supports_unequal_query_lengths():
    lengths = torch.tensor([2, 3], dtype=torch.int32)
    packed = torch.arange(5 * 2 * 3).reshape(5, 2, 3)

    padded = _packed_to_padded(packed, lengths, batch=2, maximum=4)

    assert padded.shape == (2, 4, 2, 3)
    torch.testing.assert_close(padded[0, :2], packed[:2])
    torch.testing.assert_close(padded[1, :3], packed[2:])
    assert torch.count_nonzero(padded[0, 2:]) == 0
    assert torch.count_nonzero(padded[1, 3:]) == 0


def test_upstream_version_two_contract_keeps_clip_on_fa2(monkeypatch):
    calls = []

    def flash_attn_varlen_func(q, *_args, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(
        backend_module,
        "_FA2",
        SimpleNamespace(flash_attn_varlen_func=flash_attn_varlen_func),
    )
    monkeypatch.setattr(
        backend_module,
        "_FA3",
        SimpleNamespace(flash_attn_varlen_func=lambda *_args, **_kwargs: pytest.fail("Wan CLIP explicitly requests FA2")),
    )
    _enable_fake_cuda(monkeypatch)
    q = torch.randn(2, 3, 2, 8)

    output = wan_flash_attention(q, q, q, version=2)

    assert output.shape == q.shape
    assert calls and calls[0]["dropout_p"] == 0.0
