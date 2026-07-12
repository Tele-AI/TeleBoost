"""Shared execution helpers for verl-style sharding managers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterable

__all__ = [
    "call_manager_postprocess",
    "call_manager_preprocess",
    "optional_sharding_manager",
    "run_with_sharding_managers",
    "sharding_manager_context",
]


def optional_sharding_manager(manager: Any) -> Any:
    """Return ``manager`` or a no-op manager with the same hook surface."""

    if manager is not None:
        return manager
    try:
        from teleboost.engines.fsdp.sharding.identity import IdentityShardingManager

        return IdentityShardingManager()
    except Exception:  # pragma: no cover - only for stripped-down import envs

        class _Noop:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Noop()


def _call_data_hook(hook: Callable[..., Any], data: Any, *, keyword_first: bool) -> Any:
    if keyword_first:
        try:
            return hook(data=data)
        except TypeError:
            return hook(data)
    try:
        return hook(data)
    except TypeError:
        return hook(data=data)


def call_manager_preprocess(manager: Any, data: Any, *, keyword_first: bool = False) -> Any:
    hook = getattr(manager, "preprocess_data", None)
    if not callable(hook):
        return data
    return _call_data_hook(hook, data, keyword_first=keyword_first)


def call_manager_postprocess(manager: Any, data: Any, *, keyword_first: bool = False) -> Any:
    hook = getattr(manager, "postprocess_data", None)
    if not callable(hook):
        return data
    return _call_data_hook(hook, data, keyword_first=keyword_first)


@contextmanager
def sharding_manager_context(managers: Iterable[Any]):
    with ExitStack() as stack:
        for manager in managers:
            stack.enter_context(manager)
        yield


def run_with_sharding_managers(
    data: Any,
    *,
    context_managers: Iterable[Any],
    run: Callable[[Any], Any],
    preprocess_managers: Iterable[Any] = (),
    postprocess_managers: Iterable[Any] = (),
    preprocess_keyword_first: bool = False,
    postprocess_keyword_first: bool = False,
) -> Any:
    """Enter managers, preprocess data, run one phase, and postprocess output."""

    with sharding_manager_context(context_managers):
        for manager in preprocess_managers:
            data = call_manager_preprocess(manager, data, keyword_first=preprocess_keyword_first)
        output = run(data)
        for manager in postprocess_managers:
            output = call_manager_postprocess(manager, output, keyword_first=postprocess_keyword_first)
        return output
