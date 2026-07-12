"""Shared Ulysses helpers for flattened packed-token model stacks."""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "all_to_all_heads_to_tokens",
    "all_to_all_tokens_to_heads",
    "cp_shard_bounds",
    "detect_ulysses_cp_group",
    "gather_packed_dim0",
    "pad_packed_dim0",
    "padded_token_count",
    "remap_token_indexes",
    "scatter_shard_values",
    "shard_packed_dim0",
    "shard_token_slice",
    "sim_all_to_all_heads_to_tokens",
    "sim_all_to_all_tokens_to_heads",
]


def detect_ulysses_cp_group() -> Any:
    """Return the active verl Ulysses sequence-parallel group, or None."""

    try:
        from verl.utils.ulysses import get_ulysses_sequence_parallel_group
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None

    group = get_ulysses_sequence_parallel_group()
    if group is None:
        return None
    import torch.distributed as dist

    return group if dist.get_world_size(group) > 1 else None


def padded_token_count(total_tokens: int, cp_size: int) -> int:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    return ((int(total_tokens) + cp_size - 1) // cp_size) * cp_size


def shard_token_slice(padded_total: int, cp_size: int, rank: int) -> tuple[int, int]:
    if padded_total % cp_size != 0:
        raise ValueError(f"padded_total {padded_total} not divisible by cp_size {cp_size}")
    per_rank = padded_total // cp_size
    return int(rank) * per_rank, (int(rank) + 1) * per_rank


def cp_shard_bounds(total_tokens: int, group: Any) -> tuple[int, int]:
    import torch.distributed as dist

    cp = dist.get_world_size(group)
    return shard_token_slice(padded_token_count(total_tokens, cp), cp, dist.get_rank(group))


def pad_packed_dim0(packed: torch.Tensor, cp_size: int) -> torch.Tensor:
    total = packed.shape[0]
    padded = padded_token_count(total, cp_size)
    if padded == total:
        return packed
    pad_shape = (padded - total,) + tuple(packed.shape[1:])
    return torch.cat([packed, packed.new_zeros(pad_shape)], dim=0)


def shard_packed_dim0(packed: torch.Tensor, cp_size: int) -> list[torch.Tensor]:
    return list(pad_packed_dim0(packed, cp_size).chunk(cp_size, dim=0))


def remap_token_indexes(indexes: torch.Tensor, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    kept = ((indexes >= start) & (indexes < stop)).nonzero(as_tuple=True)[0]
    return indexes[kept] - start, kept


def _exchange(x: torch.Tensor, scatter_dim: int, gather_dim: int, group: Any) -> torch.Tensor:
    import torch.distributed as dist

    cp = dist.get_world_size(group)
    rank = dist.get_rank(group)
    chunks = [t.contiguous() for t in torch.tensor_split(x, cp, scatter_dim)]
    if dist.get_backend(group) == "gloo":
        stacked = torch.stack(chunks)
        buf = [torch.empty_like(stacked) for _ in range(cp)]
        dist.all_gather(buf, stacked, group=group)
        return torch.cat([buf[r][rank] for r in range(cp)], dim=gather_dim).contiguous()
    out = [torch.empty_like(chunks[0]) for _ in range(cp)]
    dist.all_to_all(out, chunks, group=group)
    return torch.cat(out, dim=gather_dim).contiguous()


class _PackedAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group: Any, x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _exchange(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = _exchange(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group)
        return None, grad, None, None


def all_to_all_heads_to_tokens(x: torch.Tensor, group: Any) -> torch.Tensor:
    return _PackedAllToAll.apply(group, x, 1, 0)


def all_to_all_tokens_to_heads(x: torch.Tensor, group: Any) -> torch.Tensor:
    return _PackedAllToAll.apply(group, x, 0, 1)


def sim_all_to_all_heads_to_tokens(xs: list[torch.Tensor]) -> list[torch.Tensor]:
    cp = len(xs)
    n = xs[0].shape[1]
    if n % cp:
        raise ValueError(f"num_heads {n} not divisible by cp_size {cp}")
    out = []
    for j in range(cp):
        pieces = [x.chunk(cp, dim=1)[j] for x in xs]
        out.append(torch.cat(pieces, dim=0))
    return out


def sim_all_to_all_tokens_to_heads(ys: list[torch.Tensor]) -> list[torch.Tensor]:
    cp = len(ys)
    t_global = ys[0].shape[0]
    if t_global % cp:
        raise ValueError(f"t_global {t_global} not divisible by cp_size {cp}")
    out = []
    for r in range(cp):
        pieces = [y.chunk(cp, dim=0)[r] for y in ys]
        out.append(torch.cat(pieces, dim=1))
    return out


class _GatherPackedDim0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group: Any, local: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        cp = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        ctx.t_local = local.shape[0]
        buf = [torch.empty_like(local) for _ in range(cp)]
        dist.all_gather(buf, local.contiguous(), group=group)
        return torch.cat(buf, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        start = ctx.rank * ctx.t_local
        return None, grad_output[start : start + ctx.t_local]


def gather_packed_dim0(local: torch.Tensor, group: Any) -> torch.Tensor:
    return _GatherPackedDim0.apply(group, local)


class _SumAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group: Any, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output


def scatter_shard_values(values: torch.Tensor, kept_positions: torch.Tensor, out_len: int, group: Any) -> torch.Tensor:
    out = values.new_zeros((int(out_len),) + tuple(values.shape[1:]))
    out = out.index_put((kept_positions,), values)
    return _SumAllReduce.apply(group, out)
