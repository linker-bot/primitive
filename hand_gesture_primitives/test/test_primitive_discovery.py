from __future__ import annotations

import importlib
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
from hand_gesture_primitives.primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
)
from hand_gesture_primitives.primitives import (
    PRIMITIVE_REGISTRY,
    _discover_primitives,
    _module_primitive_classes,
    _primitive_name,
)


def _fixture_package(
    root: Path,
    package_name: str,
    modules: dict[str, str],
) -> Path:
    package = root / package_name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for relative, source in modules.items():
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for parent in destination.parents:
            if parent == package or package not in parent.parents:
                break
            (parent / "__init__.py").touch()
        destination.write_text(textwrap.dedent(source), encoding="utf-8")
    importlib.invalidate_caches()
    return package


_PRIMITIVE_SOURCE = """
    from hand_gesture_primitives.primitive_base import (
        HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    )

    class NestedPrimitive(HandGesturePrimitive):
        @property
        def name(self):
            return {name!r}

        def compute(self, current_angles, elapsed, ctx):
            return self._hold()
"""


def test_discovery_preserves_all_existing_commands() -> None:
    assert {
        "open",
        "init",
        "named_pose",
        "small_warp_by_vision",
        "large_wrap_by_vision",
        "hammer_grip",
    }.issubset(PRIMITIVE_REGISTRY)
    assert len(PRIMITIVE_REGISTRY) == 40


def test_default_constructible_primitive_uses_runtime_name() -> None:
    class OpenAlias(HandGesturePrimitive):
        @property
        def name(self) -> str:
            return "runtime_name"

        def compute(
            self,
            current_angles: list[float],
            elapsed: float,
            ctx: PrimitiveContext,
        ) -> PrimitiveResult:
            del current_angles, elapsed, ctx
            return self._hold()

    assert _primitive_name(OpenAlias) == "runtime_name"


def test_required_constructor_primitive_uses_snake_case_class_name() -> None:
    class ParameterizedPrimitive(HandGesturePrimitive):
        def __init__(self, required: str) -> None:
            self.required = required

        @property
        def name(self) -> str:
            return f"parameterized:{self.required}"

        def compute(
            self,
            current_angles: list[float],
            elapsed: float,
            ctx: PrimitiveContext,
        ) -> PrimitiveResult:
            del current_angles, elapsed, ctx
            return self._hold()

    assert _primitive_name(ParameterizedPrimitive) == "parameterized_primitive"


def test_module_scan_ignores_imported_and_abstract_classes() -> None:
    module = ModuleType("example_primitives")

    class LocalPrimitive(HandGesturePrimitive):
        @property
        def name(self) -> str:
            return "local"

        def compute(
            self,
            current_angles: list[float],
            elapsed: float,
            ctx: PrimitiveContext,
        ) -> PrimitiveResult:
            del current_angles, elapsed, ctx
            return self._hold()

    LocalPrimitive.__module__ = module.__name__
    module.LocalPrimitive = LocalPrimitive
    module.ImportedPrimitive = PRIMITIVE_REGISTRY["open"]

    assert list(_module_primitive_classes(module)) == [LocalPrimitive]


def test_invalid_runtime_name_fails_startup_validation() -> None:
    class InvalidName(HandGesturePrimitive):
        @property
        def name(self) -> str:
            return "invalid:name"

        def compute(
            self,
            current_angles: list[float],
            elapsed: float,
            ctx: PrimitiveContext,
        ) -> PrimitiveResult:
            del current_angles, elapsed, ctx
            return self._hold()

    with pytest.raises(RuntimeError, match="invalid command name"):
        _primitive_name(InvalidName)


def test_discovery_recurses_through_multiple_package_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fixture_package(
        tmp_path,
        "nested_fixture_primitives",
        {
            "grasp/top/enveloping.py": _PRIMITIVE_SOURCE.format(name="nested_top_enveloping"),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = _discover_primitives(
        "nested_fixture_primitives",
        [str(package)],
    )

    assert registry["nested_top_enveloping"].__module__ == (
        "nested_fixture_primitives.grasp.top.enveloping"
    )


def test_discovery_skips_private_packages_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fixture_package(
        tmp_path,
        "private_fixture_primitives",
        {
            "public.py": _PRIMITIVE_SOURCE.format(name="public_primitive"),
            "_private/should_not_import.py": "raise AssertionError('imported private module')",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = _discover_primitives(
        "private_fixture_primitives",
        [str(package)],
    )

    assert set(registry) == {"public_primitive"}


def test_nested_duplicate_names_report_both_module_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fixture_package(
        tmp_path,
        "duplicate_fixture_primitives",
        {
            "first/item.py": _PRIMITIVE_SOURCE.format(name="duplicate_name"),
            "second/item.py": _PRIMITIVE_SOURCE.format(name="duplicate_name"),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(RuntimeError, match="Duplicate primitive command") as error:
        _discover_primitives(
            "duplicate_fixture_primitives",
            [str(package)],
        )

    assert "duplicate_fixture_primitives.first.item.NestedPrimitive" in str(error.value)
    assert "duplicate_fixture_primitives.second.item.NestedPrimitive" in str(error.value)
