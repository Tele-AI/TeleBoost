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
# TeleTron lineage (https://github.com/Tele-AI/TeleTron).

import inspect
from collections.abc import Callable
from typing import Any, Optional, TypeVar

T = TypeVar("T")


class RegistryError(Exception):
    """Base exception for registry-related errors."""

    pass


class ModuleAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a module that already exists."""

    pass


class ModuleNotFoundError(RegistryError):
    """Raised when attempting to build a module that doesn't exist in registry."""

    pass


class ModelRegistry:
    """
    A registry for managing and instantiating modules dynamically.

    This registry allows you to register classes/functions and later instantiate
    them by name with configuration parameters.

    Example:
        >>> registry = ModelRegistry("processors")
        >>>
        >>> @registry.register
        >>> class TextProcessor:
        >>>     def __init__(self, config: str):
        >>>         self.config = config
        >>>
        >>> # or register with custom name
        >>> registry.register_module(TextProcessor, "custom_processor")
        >>>
        >>> # Build instances
        >>> processor = registry.build("TextProcessor", config="my_config")
        >>> custom = registry.build({"type": "custom_processor", "config": "data"})
    """

    def __init__(self, name: str = "Registry"):
        """
        Initialize the registry.

        Args:
            name: A descriptive name for this registry (used in error messages).
        """
        self.name = name
        self._modules: dict[str, type] = {}

    def register_module(self, module_class: type[T], module_name: Optional[str] = None) -> type[T]:
        """
        Register a module class with the registry.

        Args:
            module_class: The class to register.
            module_name: Optional custom name. If None, uses class.__name__.

        Returns:
            The registered module class (for decorator chaining).

        Raises:
            ModuleAlreadyRegisteredError: If module_name already exists.
        """
        if module_name is None:
            module_name = module_class.__name__

        if module_name in self._modules:
            raise ModuleAlreadyRegisteredError(f"Module '{module_name}' is already registered in {self.name}. Existing: {self._modules[module_name]}, New: {module_class}")

        self._modules[module_name] = module_class
        return module_class

    def register(self, name_or_class: str | type[T]) -> Callable[[type[T]], type[T]] | type[T]:
        """
        Register a module class, supporting both direct registration and decorator usage.

        Usage:
            # Direct registration
            registry.register(MyClass)

            # Decorator without custom name
            @registry.register
            class MyClass: ...

            # Decorator with custom name
            @registry.register("custom_name")
            class MyClass: ...

        Args:
            name_or_class: Either a string name or the class to register.

        Returns:
            Either the registered class or a decorator function.
        """
        if isinstance(name_or_class, str):
            # Decorator with custom name
            def decorator(module_class: type[T]) -> type[T]:
                return self.register_module(module_class, name_or_class)

            return decorator
        else:
            # Direct registration or decorator without custom name
            return self.register_module(name_or_class)

    def get_module(self, module_name: str) -> type:
        """
        Get a registered module class by name.

        Args:
            module_name: Name of the module to retrieve.

        Returns:
            The registered module class.

        Raises:
            ModuleNotFoundError: If module doesn't exist.
        """
        if module_name not in self._modules:
            available = list(self._modules.keys())
            raise ModuleNotFoundError(f"Module '{module_name}' not found in {self.name}. Available modules: {available}")
        return self._modules[module_name]

    def build(self, name: str, config=None, *args, **kwargs) -> Any:
        """
        Build and instantiate a registered module by name.

        Args:
            name: The name of the registered module to build.
            *args: Positional arguments to pass to the module constructor.
            **kwargs: Keyword arguments to pass to the module constructor.

        Returns:
            An instance of the requested module.

        Raises:
            ModuleNotFoundError: If the specified module doesn't exist.
            TypeError: If the module constructor fails.
        """
        module_class = self.get_module(name)

        try:
            return module_class(config=config, *args, **kwargs)
        except TypeError as e:
            # Enhance error message with signature info
            sig = inspect.signature(module_class.__init__)
            raise TypeError(f"Failed to instantiate {name}: {e}. Expected signature: {sig}") from e

    def __contains__(self, module_name: str) -> bool:
        """Check if a module is registered."""
        return module_name in self._modules

    def __len__(self) -> int:
        """Get the number of registered modules."""
        return len(self._modules)

    def __repr__(self) -> str:
        """String representation of the registry."""
        modules = list(self._modules.keys())
        return f"{self.name}({len(modules)} modules: {modules})"
