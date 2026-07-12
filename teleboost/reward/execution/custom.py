# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications Copyright 2025-2026 TeleAI and the TeleBoost contributors
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
"""Config-loaded custom batch reward functions.

The public config surface mirrors upstream verl:

``reward.custom_reward_function.path``
    ``pkg://package.module`` or a Python file path.
``reward.custom_reward_function.name``
    Callable name inside that module. Defaults to ``compute_score``.
``reward.custom_reward_function.reward_kwargs``
    Optional keyword arguments merged into every call.

Unlike upstream's token-level ``NaiveRewardManager`` call site, this
TeleBoost path is diffusion batch-level: the callable receives the generated
``DataProto`` and returns either a reward ``DataProto``, a ``{"rewards":
tensor}`` mapping, or a reward tensor shaped ``(B,)``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any

import torch

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    _OMEGACONF_TYPES = (DictConfig, ListConfig)
except Exception:  # pragma: no cover - lightweight reward envs may omit omegaconf
    OmegaConf = None
    _OMEGACONF_TYPES = ()

PKG_PATH_PREFIX = "pkg://"
FILE_PATH_PREFIX = "file://"


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if _OMEGACONF_TYPES and isinstance(value, _OMEGACONF_TYPES):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"reward_kwargs must be a mapping, got {type(value).__name__}")
    return dict(value)


def _load_module(module_path: str) -> Any:
    """Load a Python module, matching verl's ``pkg://`` / file-path forms."""
    if module_path.startswith(PKG_PATH_PREFIX):
        module_name = module_path[len(PKG_PATH_PREFIX) :].replace("/", ".")
        return importlib.import_module(module_name)

    if module_path.startswith(FILE_PATH_PREFIX):
        module_path = module_path[len(FILE_PATH_PREFIX) :]

    if not module_path.endswith(".py") and not os.path.exists(module_path):
        return importlib.import_module(module_path)

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Custom reward function module not found: {module_path}")

    spec_name = f"teleboost_custom_reward_{hash(os.path.abspath(module_path))}"
    spec = importlib.util.spec_from_file_location(spec_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reward function module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Error loading reward function module from {module_path}") from exc
    return module


def _load_object(module_path: str, object_name: str) -> Callable:
    module = _load_module(module_path)
    if not hasattr(module, object_name):
        raise AttributeError(f"Reward function '{object_name}' not found in {module_path}")
    obj = getattr(module, object_name)
    if not callable(obj):
        raise TypeError(f"Configured reward function '{object_name}' in {module_path} is not callable")
    return obj


def load_custom_reward_fn(config: Any) -> Callable:
    """Load the batch reward function configured for ``trainer.use_rm=false``."""
    reward_cfg = _get(_get(config, "reward"), "custom_reward_function") or {}
    module_path = _get(reward_cfg, "path")
    if not module_path:
        raise ValueError("trainer.use_rm=false requires reward.custom_reward_function.path (for example: pkg://teleboost.reward.providers.debug.red_intensity)")

    fn_name = _get(reward_cfg, "name") or "compute_score"
    raw_fn = _load_object(str(module_path), str(fn_name))
    if inspect.iscoroutinefunction(raw_fn):
        raise TypeError("TeleBoost batch reward functions must be synchronous; async compute_score is for RewardLoopWorker.")

    reward_kwargs = _as_plain_mapping(_get(reward_cfg, "reward_kwargs", {}))

    def _wrapped(data):
        return raw_fn(data, **reward_kwargs)

    return _wrapped


def normalize_custom_reward_output(output: Any, template) -> Any:
    """Normalize common reward-fn return types to a ``DataProto``.

    ``template`` is the generated batch; it supplies batch size and
    non-tensor metadata when the reward function returns a plain tensor.
    """
    from tensordict import TensorDict
    from verl import DataProto

    if isinstance(output, DataProto):
        return output

    if isinstance(output, Mapping):
        if "rewards" not in output:
            raise KeyError("Reward function returned a mapping without a 'rewards' key")
        rewards = output["rewards"]
    else:
        rewards = output

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
    rewards = rewards.detach().to(torch.device("cpu"))
    if rewards.ndim == 0:
        rewards = rewards.expand(len(template))
    if rewards.shape[0] != len(template):
        raise ValueError(f"Reward function returned {rewards.shape[0]} rewards for batch size {len(template)}")

    batch = TensorDict({"rewards": rewards}, batch_size=len(template))
    return DataProto(batch=batch, non_tensor_batch=template.non_tensor_batch)


__all__ = ["load_custom_reward_fn", "normalize_custom_reward_output"]
