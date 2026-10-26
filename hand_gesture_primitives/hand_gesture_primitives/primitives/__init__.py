"""Discover and register built-in hand gesture primitives.

Adding a primitive module no longer requires editing this package file. Every
concrete :class:`HandGesturePrimitive` subclass defined directly in a module
at any public package depth below this package is discovered at import time.
Classes with a default constructor use their runtime ``name`` property;
parameter-only primitives use their snake_case class name as the command key.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterable, Sequence
from types import ModuleType

from ..primitive_base import HandGesturePrimitive

_COMMAND_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _snake_case(class_name: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _primitive_name(primitive_class: type[HandGesturePrimitive]) -> str:
    """Resolve a stable command name without requiring constructor metadata."""

    try:
        inspect.signature(primitive_class).bind()
    except TypeError:
        # Named/parameterized primitives cannot be instantiated solely for
        # discovery. Their class name is the stable registration contract.
        name = _snake_case(primitive_class.__name__)
    else:
        name = str(primitive_class().name).strip()
    if not _COMMAND_NAME.fullmatch(name):
        raise RuntimeError(
            f"Primitive {primitive_class.__module__}.{primitive_class.__name__} "
            f"has invalid command name {name!r}"
        )
    return name


def _module_primitive_classes(
    module: ModuleType,
) -> Iterable[type[HandGesturePrimitive]]:
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ != module.__name__:
            continue
        if candidate is HandGesturePrimitive or not issubclass(
            candidate, HandGesturePrimitive
        ):
            continue
        if inspect.isabstract(candidate):
            continue
        yield candidate


def _import_primitive_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(
            f"Failed to import primitive module {module_name!r}"
        ) from error


def _iter_primitive_modules(
    package_name: str,
    package_paths: Sequence[str],
) -> Iterable[ModuleType]:
    """Recursively import public modules in deterministic lexical order."""

    modules = sorted(
        pkgutil.iter_modules(package_paths, prefix=f"{package_name}."),
        key=lambda item: item.name,
    )
    for module_info in modules:
        relative_parts = module_info.name.removeprefix(
            f"{package_name}."
        ).split(".")
        if any(part.startswith("_") for part in relative_parts):
            continue
        module = _import_primitive_module(module_info.name)
        yield module
        if not module_info.ispkg:
            continue
        child_paths = tuple(str(path) for path in getattr(module, "__path__", ()))
        if child_paths:
            yield from _iter_primitive_modules(module_info.name, child_paths)


def _discover_primitives(
    package_name: str,
    package_paths: Sequence[str],
) -> dict[str, type[HandGesturePrimitive]]:
    """Discover one package tree and return a collision-checked registry."""

    registry: dict[str, type[HandGesturePrimitive]] = {}
    for module in _iter_primitive_modules(package_name, package_paths):
        for primitive_class in _module_primitive_classes(module):
            command_name = _primitive_name(primitive_class)
            existing = registry.get(command_name)
            if existing is not None:
                raise RuntimeError(
                    f"Duplicate primitive command {command_name!r}: "
                    f"{existing.__module__}.{existing.__name__} and "
                    f"{primitive_class.__module__}.{primitive_class.__name__}"
                )
            registry[command_name] = primitive_class
    if not registry:
        raise RuntimeError(f"No hand gesture primitives discovered under {package_name}")
    return registry


def discover_primitives() -> dict[str, type[HandGesturePrimitive]]:
    """Discover all built-in primitive modules recursively."""

    return _discover_primitives(__name__, tuple(str(path) for path in __path__))


PRIMITIVE_REGISTRY = discover_primitives()

__all__ = ["PRIMITIVE_REGISTRY", "discover_primitives"]
