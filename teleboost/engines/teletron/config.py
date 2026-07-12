# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
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
import copy
import os
import pprint

from teleboost.config.io import import_function, load_file, save_file


def set_config():
    import importlib
    import traceback

    from teleboost.engines.teletron import get_args

    args = get_args()
    try:
        *module_parts, var_name = args.config_path.split(".")
        module_path = ".".join(module_parts)

        # Dynamically import the module
        module = importlib.import_module(module_path)
        config = getattr(module, var_name)
    except Exception:
        raise ValueError(f"Failed on load config: {traceback.format_exc()}")

    config = load_config(config)
    return config


def load_config(config_or_path, *, allow_unsafe_pickle=False):
    if isinstance(config_or_path, str | os.PathLike):
        config_or_path = os.fspath(config_or_path)
        if os.path.isdir(config_or_path):
            config_path = os.path.join(config_or_path, "config.json")
        else:
            config_path = config_or_path
        config = Config.load(
            config_path,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
    elif isinstance(config_or_path, Config):
        config = config_or_path
    elif isinstance(config_or_path, dict):
        config = Config(config_or_path)
    else:
        raise AssertionError()
    return config


class Config(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def _process_value(self, value):
        assert value is None or isinstance(value, int | float | bool | str | list | tuple | dict | self.__class__)
        if isinstance(value, list | tuple):
            if len(value) > 0 and value[0] == "__tuple__":
                value = tuple(value[1:])
            is_tuple = isinstance(value, tuple)
            value = [self._process_value(x) if isinstance(x, list | tuple | dict) else x for x in value]
            if is_tuple:
                value = tuple(value)
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        return value

    def __setattr__(self, name, value):
        value = self._process_value(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __str__(self):
        return "Config:\n{}".format(self.pretty_text)

    @property
    def pretty_text(self):
        return pprint.pformat(self)

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k, v in d.items():
            if hasattr(self, k):
                force = v.pop("__force__", False) if isinstance(v, dict) else False
                if isinstance(v, dict) and isinstance(self[k], dict) and not force:
                    self[k].update(v)
                else:
                    setattr(self, k, v)
            else:
                setattr(self, k, v)
        return self

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super().pop(k, d)

    def setdefault(self, k, d=None):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            setattr(self, k, d)
            return d

    def copy(self):
        return copy.deepcopy(self)

    def _to_dict(self, data, tuple_as_list=False):
        if isinstance(data, dict | self.__class__):
            return {k: self._to_dict(d, tuple_as_list) for k, d in data.items()}
        elif isinstance(data, list | tuple):
            new_data = [self._to_dict(d, tuple_as_list) for d in data]
            if isinstance(data, tuple):
                if tuple_as_list:
                    new_data = ["__tuple__"] + new_data
                else:
                    new_data = tuple(new_data)
            return new_data
        else:
            return data

    def to_dict(self, tuple_as_list=False):
        return self._to_dict(self, tuple_as_list=tuple_as_list)

    def save(self, filename):
        filename = os.fspath(filename)
        tuple_as_list = True if filename.endswith(".json") else False
        data = self.to_dict(tuple_as_list=tuple_as_list)
        save_file(filename, data)

    @classmethod
    def load(cls, filename, *, allow_unsafe_pickle=False):
        filename = os.fspath(filename)
        if filename.endswith(".config"):
            config = import_function(filename)
        elif filename.endswith(".py"):
            config = import_function(filename[:-3] + "/config", "/")
        else:
            config = load_file(
                filename,
                allow_unsafe_pickle=allow_unsafe_pickle,
            )
        return cls(config)
