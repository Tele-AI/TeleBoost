"""Read OmegaConf, mapping, and attribute-backed configuration nodes."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised in the remote verl environment
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None

__all__ = ["as_bool", "as_float", "as_int", "select"]


def select(config: Any, path: str, default: Any = None) -> Any:
    """Look up a dotted path on an OmegaConf node, mapping, or object."""

    if OmegaConf is not None:
        try:
            return OmegaConf.select(config, path, default=default)
        except Exception:
            pass

    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
            continue
        getter = getattr(current, "get", None)
        if callable(getter):
            try:
                current = getter(part, default)
                continue
            except Exception:
                pass
        current = getattr(current, part, default)
    return current


def as_bool(config: Any, path: str, default: bool) -> bool:
    value = select(config, path, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_int(config: Any, path: str, default: int) -> int:
    return int(select(config, path, default))


def as_float(config: Any, path: str, default: float) -> float:
    return float(select(config, path, default))
