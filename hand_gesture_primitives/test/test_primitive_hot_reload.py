"""验证真实源码加载、失败回滚及 debug/空闲边界，不连接 ROS 或硬件。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from hand_gesture_primitives.primitive_reloader import PrimitiveHotReloader

from hand_gesture_primitives import primitive_reloader, primitives


def _status(path: Path, mode: str, state: str = "READY") -> None:
    path.write_text(json.dumps({"skill_source_mode": mode, "state": state}))


def _primitive(class_name: str, name: str, value: int = 1) -> str:
    return f"""from hand_gesture_primitives.primitive_base import HandGesturePrimitive
VALUE = {value}
class {class_name}(HandGesturePrimitive):
    @property
    def name(self): return {name!r}
    def compute(self, *args): return VALUE
"""


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    prefix = primitives.__name__ + "."
    old_modules = {n: m for n, m in sys.modules.copy().items() if n.startswith(prefix)}
    old_parent = dict(primitives.__dict__)
    old_registry = dict(primitives.PRIMITIVE_REGISTRY)
    source = tmp_path / "primitives"
    source.mkdir()
    (source / "required.py").write_text(
        _primitive("Init", "init") + _primitive("NamedPose", "named_pose")
    )
    monkeypatch.setattr(primitives, "__path__", [str(source)])
    try:
        yield source
    finally:
        for name in tuple(sys.modules):
            if name.startswith(prefix):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        primitives.__dict__.clear()
        primitives.__dict__.update(old_parent)
        primitives.PRIMITIVE_REGISTRY.clear()
        primitives.PRIMITIVE_REGISTRY.update(old_registry)


def test_changed_source_waits_for_idle_and_hub_debug(source_tree, tmp_path):
    item = source_tree / "demo.py"
    item.write_text(_primitive("Demo", "demo"))
    status = tmp_path / "status.json"
    _status(status, "production")
    reloader = PrimitiveHotReloader(status)
    item.write_text(_primitive("Demo", "demo", 2))
    assert reloader.reload_if_changed(idle=True, force=True) == "disabled"
    _status(status, "debug", "STARTING")
    assert reloader.reload_if_changed(idle=True) == "disabled"
    _status(status, "debug")
    assert reloader.reload_if_changed(idle=False, force=True) == "deferred"
    assert reloader.reload_if_changed(idle=True) == "reloaded"
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 2
    assert reloader.reload_if_changed(idle=True) == "unchanged"


@pytest.mark.parametrize("text", ["", "{", "[]", "{}", '{"skill_source_mode":"debug"}'])
def test_invalid_status_disables_reload(tmp_path, text):
    status = tmp_path / "status.json"
    status.write_text(text)
    assert not PrimitiveHotReloader(status, package_paths=[str(tmp_path)]).enabled
    status.unlink()
    assert not PrimitiveHotReloader(status, package_paths=[str(tmp_path)]).enabled


def test_same_timestamp_change_ignores_old_bytecode(source_tree):
    import py_compile

    item = source_tree / "demo.py"
    item.write_text(_primitive("Demo", "demo", 1))
    py_compile.compile(str(item))
    primitive_reloader.reload_primitive_registry()
    old = primitives.PRIMITIVE_REGISTRY["demo"]()
    before = primitive_reloader.primitive_source_fingerprint([str(source_tree)])
    stat = item.stat()
    item.write_text(_primitive("Demo", "demo", 2))
    os.utime(item, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert primitive_reloader.primitive_source_fingerprint([str(source_tree)]) != before
    registry_id = id(primitives.PRIMITIVE_REGISTRY)
    primitive_reloader.reload_primitive_registry()
    assert id(primitives.PRIMITIVE_REGISTRY) == registry_id
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 2
    assert old.compute() == 1


def test_deleted_class_and_module_are_not_retained(source_tree):
    item = source_tree / "demo.py"
    item.write_text(_primitive("Old", "old"))
    primitive_reloader.reload_primitive_registry()
    item.write_text(_primitive("New", "new"))
    primitive_reloader.reload_primitive_registry()
    assert "old" not in primitives.PRIMITIVE_REGISTRY
    assert "new" in primitives.PRIMITIVE_REGISTRY
    item.unlink()
    primitive_reloader.reload_primitive_registry()
    assert "new" not in primitives.PRIMITIVE_REGISTRY
    assert primitives.__name__ + ".demo" not in sys.modules
    assert not hasattr(primitives, "demo")


def test_cross_module_import_and_nested_package_use_new_sources(source_tree):
    nested = source_tree / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("from .z import Helper\n")
    helper = nested / "z.py"
    helper.write_text(_primitive("Helper", "helper", 1))
    (source_tree / "a.py").write_text("""from .nested import Helper
class Demo(Helper):
    @property
    def name(self): return "demo"
""")
    primitive_reloader.reload_primitive_registry()
    old = primitives.PRIMITIVE_REGISTRY["demo"]()
    helper.write_text(_primitive("Helper", "helper", 2))
    primitive_reloader.reload_primitive_registry()
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 2
    assert old.compute() == 1


@pytest.mark.parametrize(
    "bad_source",
    [
        "def broken(:\n",
        "raise RuntimeError('import failed')\n",
        _primitive("Conflict", "init"),
        "from .missing import Value\n",
    ],
)
def test_failed_reload_restores_modules_and_retries_after_edit(source_tree, tmp_path, bad_source):
    item = source_tree / "demo.py"
    item.write_text(_primitive("Demo", "demo"))
    primitive_reloader.reload_primitive_registry()
    old_registry = dict(primitives.PRIMITIVE_REGISTRY)
    old_module = sys.modules[primitives.__name__ + ".demo"]
    status = tmp_path / "status.json"
    _status(status, "debug")
    reloader = PrimitiveHotReloader(status)
    item.write_text(bad_source)
    with pytest.raises(Exception):
        reloader.reload_if_changed(idle=True)
    assert primitives.PRIMITIVE_REGISTRY == old_registry
    assert sys.modules[old_module.__name__] is old_module
    assert primitives.demo is old_module
    assert old_registry["demo"]().compute() == 1
    assert reloader.reload_if_changed(idle=True) == "unchanged"
    item.write_text(_primitive("Demo", "demo", 3))
    assert reloader.reload_if_changed(idle=True) == "reloaded"
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 3


def test_cannot_remove_required_fallback_primitives(source_tree):
    primitive_reloader.reload_primitive_registry()
    before = dict(primitives.PRIMITIVE_REGISTRY)
    (source_tree / "required.py").write_text(_primitive("Demo", "demo"))
    with pytest.raises(RuntimeError, match="Required primitive missing"):
        primitive_reloader.reload_primitive_registry()
    assert primitives.PRIMITIVE_REGISTRY == before


def test_root_registry_implementation_is_outside_reload_scope(source_tree):
    before = primitive_reloader.primitive_source_fingerprint([str(source_tree)])
    (source_tree / "__init__.py").write_text("raise RuntimeError('restart required')\n")
    assert primitive_reloader.primitive_source_fingerprint([str(source_tree)]) == before


def test_profile_capability_cache_tracks_registry_changes(monkeypatch):
    from hand_gesture_primitives.hand_profile import HandProfile

    profile = HandProfile.__new__(HandProfile)
    profile._supported_set = None
    profile._mode = "explicit"
    profile._supported_explicit = set()
    profile._unsupported = set()
    before = profile._compute_supported()
    monkeypatch.setitem(primitives.PRIMITIVE_REGISTRY, "new_primitive", object)
    assert "new_primitive" not in before
    assert "new_primitive" in profile._compute_supported()
    monkeypatch.delitem(primitives.PRIMITIVE_REGISTRY, "new_primitive")
    assert "new_primitive" not in profile._compute_supported()


def test_actual_builtin_tree_reloads_init_and_named_pose(source_tree, monkeypatch):
    monkeypatch.setattr(primitives, "__path__", [str(Path(primitives.__file__).parent)])
    old_registry = dict(primitives.PRIMITIVE_REGISTRY)
    candidate = primitive_reloader.reload_primitive_registry()
    assert candidate.keys() == old_registry.keys()
    for name in ("init", "named_pose", "open"):
        assert candidate[name] is not old_registry[name]
    assert candidate["named_pose"]("grip").pose_name == "grip"


def _lazy_primitive(source_tree):
    (source_tree / "demo.py").write_text(
        _primitive("Demo", "demo").replace(
            "def compute(self, *args): return VALUE",
            "def compute(self, *args):\n        from ._helper import VALUE\n        return VALUE",
        )
    )
    helper = source_tree / "_helper.py"
    helper.write_text("VALUE = 1\n")
    return helper


@pytest.mark.parametrize("hand_type", ["o20", "o6", "l6", "l25"])
@pytest.mark.parametrize("name", ["init", "open"])
def test_reload_without_source_edits_preserves_motion_results(
    source_tree, monkeypatch, hand_type, name
):
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM

    monkeypatch.setattr(primitives, "__path__", [str(Path(primitives.__file__).parent)])
    old = primitives.PRIMITIVE_REGISTRY[name]()
    primitive_reloader.reload_primitive_registry()
    new = primitives.PRIMITIVE_REGISTRY[name]()
    angles = [0.0] * SEMANTIC_DIM
    old.on_enter(angles)
    new.on_enter(angles)
    for elapsed in (0.0, 0.1, 0.5, 2.0):
        expected = old.compute(angles, elapsed, PrimitiveContext(hand_type=hand_type))
        actual = new.compute(angles, elapsed, PrimitiveContext(hand_type=hand_type))
        assert actual == expected
        assert new.done == old.done


def test_lazy_import_uses_new_source_despite_old_bytecode(source_tree):
    import py_compile

    helper = _lazy_primitive(source_tree)
    py_compile.compile(str(helper), doraise=True)
    stat = helper.stat()
    primitive_reloader.reload_primitive_registry()
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 1
    helper.write_text("VALUE = 2\n")
    os.utime(helper, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    primitive_reloader.reload_primitive_registry()
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 2
    finders = [f for f in sys.meta_path if isinstance(f, primitive_reloader._SourceFinder)]
    assert len(finders) == 1
    assert finders[0].find_spec("unrelated_package") is None


@pytest.mark.parametrize("change", ["edit", "delete", "failed_reload"])
def test_lazy_import_keeps_last_validated_generation(source_tree, change):
    helper = _lazy_primitive(source_tree)
    primitive_reloader.reload_primitive_registry()
    old_registry = dict(primitives.PRIMITIVE_REGISTRY)
    assert primitives.__name__ + "._helper" not in sys.modules
    if change == "delete":
        helper.unlink()
    else:
        helper.write_text("VALUE = 2\n")
    if change == "failed_reload":
        (source_tree / "broken.py").write_text("raise RuntimeError('import failed')\n")
        with pytest.raises(RuntimeError):
            primitive_reloader.reload_primitive_registry()
    assert primitives.PRIMITIVE_REGISTRY == old_registry
    assert primitives.PRIMITIVE_REGISTRY["demo"]().compute() == 1


@pytest.mark.parametrize(
    "mode,idle,expected",
    [
        ("production", True, "disabled"),
        ("debug", False, "deferred"),
        ("debug", True, "unchanged"),
    ],
)
def test_no_reload_preserves_existing_classes_modules_and_imports(
    source_tree, tmp_path, mode, idle, expected
):
    helper = _lazy_primitive(source_tree)
    status = tmp_path / "status.json"
    _status(status, mode)
    reloader = PrimitiveHotReloader(status)
    old_registry = dict(primitives.PRIMITIVE_REGISTRY)
    old_modules = dict(sys.modules)
    old_finders = list(sys.meta_path)
    if expected != "unchanged":
        helper.write_text("VALUE = 2\n")
    assert reloader.reload_if_changed(idle=idle) == expected
    assert primitives.PRIMITIVE_REGISTRY == old_registry
    assert sys.meta_path == old_finders
    assert {n: m for n, m in sys.modules.items() if n.startswith(primitives.__name__)} == {
        n: m for n, m in old_modules.items() if n.startswith(primitives.__name__)
    }
