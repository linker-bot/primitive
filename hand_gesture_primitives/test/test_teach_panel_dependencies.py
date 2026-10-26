"""Wrong Web dependencies fail before the panel starts any runtime."""

from types import SimpleNamespace

import pytest
from hand_gesture_primitives.teach_panel import dependencies


@pytest.mark.parametrize("version", ["1.10.14", "3.0.0"])
def test_wrong_pydantic_fails_before_importing_the_web_runtime(monkeypatch, version):
    imported = []

    def load(name):
        imported.append(name)
        assert name == "pydantic"
        return SimpleNamespace(__version__=version)

    monkeypatch.setattr(dependencies.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match=f"found {version}") as failure:
        dependencies.check_web_dependencies()
    assert "system-Python venv" in str(failure.value)
    assert dependencies.sys.executable in str(failure.value)
    assert imported == ["pydantic"]


def test_missing_web_dependency_points_to_the_same_installation(monkeypatch):
    def load(name):
        if name == "pydantic":
            return SimpleNamespace(__version__="2.13.5")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(dependencies.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="requires fastapi") as failure:
        dependencies.check_web_dependencies()
    assert "docs/workbench.md" in str(failure.value)


def test_installed_web_stack_passes():
    dependencies.check_web_dependencies()


def test_mock_entrypoint_checks_dependencies_before_creating_host(monkeypatch, tmp_path):
    from hand_gesture_primitives.teach_panel.main import main

    def reject():
        raise RuntimeError("unsupported Pydantic")

    monkeypatch.setattr(dependencies, "check_web_dependencies", reject)
    with pytest.raises(SystemExit, match="unsupported Pydantic"):
        main(["--mock", "--pose-dir", str(tmp_path / "poses")])
    assert not (tmp_path / "poses").exists()
