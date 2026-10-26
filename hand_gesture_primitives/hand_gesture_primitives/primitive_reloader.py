"""Transactional debug hot reload for hand gesture primitives.

The authoritative debug/production choice is resolved by Framework from the
Hub robot configuration and written to ``runtime-status.json``.  Gesture nodes
read that durable result; they do not introduce a second environment switch.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

from . import primitives


def _python_sources(package_paths: Sequence[str]) -> tuple[Path, ...]:
    sources: list[Path] = []
    for raw_root in package_paths:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        sources.extend(
            path
            for path in root.rglob("*.py")
            if path != root / "__init__.py"
            and "__pycache__" not in path.parts
            and not any(part.startswith("_") for part in path.relative_to(root).parts[:-1])
        )
    return tuple(sorted(set(sources), key=lambda path: str(path)))


def primitive_source_fingerprint(package_paths: Sequence[str]) -> str:
    """Hash source paths and bytes so same-timestamp edits are detected."""

    digest = hashlib.sha256()
    for path in _python_sources(package_paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class _SourceLoader(importlib.abc.Loader):
    """执行已编译的源码快照，避免读取同时间戳的旧 pyc。"""

    def __init__(self, code):
        self.code = code

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(self.code, module.__dict__)


class _SourceFinder(importlib.abc.MetaPathFinder):
    """仅接管 primitives 子模块；源文件消失时禁止回退到旧缓存。"""

    def __init__(self, package_name, package_paths):
        self.prefix = package_name + "."
        self.sources = {}
        for raw_root in package_paths:
            root = Path(raw_root)
            for path in _python_sources([raw_root]):
                parts = path.relative_to(root).with_suffix("").parts
                is_package = parts[-1] == "__init__"
                if is_package:
                    parts = parts[:-1]
                name = self.prefix + ".".join(parts)
                self.sources[name] = (
                    path,
                    is_package,
                    compile(path.read_bytes(), str(path), "exec"),
                )

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix):
            return None
        if fullname not in self.sources:
            raise ModuleNotFoundError(f"Primitive source not found: {fullname}")
        source, is_package, code = self.sources[fullname]
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=_SourceLoader(code),
            submodule_search_locations=[str(source.parent)] if is_package else None,
        )


def reload_primitive_registry() -> dict[str, type]:
    """在新模块对象中构建注册表，校验成功后才切换；失败恢复旧模块。"""

    package_name = primitives.__name__
    prefix = package_name + "."
    package_paths = tuple(str(path) for path in primitives.__path__)
    # 先编译全部候选源码，语法错误时不触碰运行中的模块。
    finder = _SourceFinder(package_name, package_paths)
    old_modules = {
        name: module for name, module in sys.modules.copy().items() if name.startswith(prefix)
    }
    parent_snapshot = dict(primitives.__dict__)
    registry_snapshot = dict(primitives.PRIMITIVE_REGISTRY)
    for name in old_modules:
        sys.modules.pop(name, None)
    for name, value in tuple(primitives.__dict__.items()):
        if isinstance(value, ModuleType) and value.__name__.startswith(prefix):
            delattr(primitives, name)
    importlib.invalidate_caches()
    sys.meta_path.insert(0, finder)
    try:
        candidate = primitives.discover_primitives()
        for required in ("init", "named_pose"):
            if required not in candidate:
                raise RuntimeError(f"Required primitive missing: {required}")
    except Exception:
        for name in tuple(sys.modules):
            if name.startswith(prefix):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        primitives.__dict__.clear()
        primitives.__dict__.update(parent_snapshot)
        primitives.PRIMITIVE_REGISTRY.clear()
        primitives.PRIMITIVE_REGISTRY.update(registry_snapshot)
        raise
    finally:
        sys.meta_path.remove(finder)
    primitives.PRIMITIVE_REGISTRY.clear()
    primitives.PRIMITIVE_REGISTRY.update(candidate)
    # compute()/on_enter() 内的延迟导入也必须使用已校验的快照，避免读取旧 pyc，
    # 或在原语执行期间读入尚未校验的编辑。仅成功时替换上一代加载器；失败保留旧代。
    for previous in tuple(sys.meta_path):
        if isinstance(previous, _SourceFinder) and previous.prefix == prefix:
            sys.meta_path.remove(previous)
    sys.meta_path.insert(0, finder)
    return dict(candidate)


class PrimitiveHotReloader:
    """Track Hub-resolved source mode and reload changed idle primitives."""

    def __init__(
        self,
        runtime_status_path: str | Path,
        *,
        package_paths: Sequence[str] | None = None,
    ) -> None:
        self.runtime_status_path = Path(runtime_status_path)
        self.package_paths = tuple(package_paths or (str(path) for path in primitives.__path__))
        self._applied_fingerprint = primitive_source_fingerprint(self.package_paths)
        self._failed_fingerprint = ""

    def source_mode(self) -> str:
        try:
            payload = json.loads(self.runtime_status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return ""
        if not isinstance(payload, Mapping) or payload.get("state") != "READY":
            return ""
        mode = str(payload.get("skill_source_mode") or "").strip().lower()
        return mode if mode in {"debug", "production"} else ""

    @property
    def enabled(self) -> bool:
        return self.source_mode() == "debug"

    def reload_if_changed(self, *, idle: bool, force: bool = False) -> str:
        """Return ``unchanged``, ``deferred``, ``disabled`` or ``reloaded``.

        Reload exceptions are intentionally propagated so the node can publish
        a visible diagnostic. A failed fingerprint is remembered to avoid log
        spam until the developer edits the source again; ``force`` retries it.
        """

        if not self.enabled:
            return "disabled"
        fingerprint = primitive_source_fingerprint(self.package_paths)
        changed = fingerprint != self._applied_fingerprint
        if not force and not changed:
            return "unchanged"
        if not force and fingerprint == self._failed_fingerprint:
            return "unchanged"
        if not idle:
            return "deferred"
        try:
            reload_primitive_registry()
        except Exception:
            self._failed_fingerprint = fingerprint
            raise
        self._applied_fingerprint = fingerprint
        self._failed_fingerprint = ""
        return "reloaded"
