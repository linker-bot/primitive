"""Real wheel/sdist delivery of the repository-owned O6 right FK model."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from hand_gesture_primitives import urdf_utils

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('o6/right/linkerhand_o6_right.urdf')


def run(command, cwd, environment=None):
    result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_installed_search_preserves_explicit_model_overrides(tmp_path, monkeypatch):
    share = tmp_path / 'venv/share/hand_gesture_primitives/urdf'
    external = tmp_path / 'external'
    for folder in (share, external):
        path = folder / MODEL
        path.parent.mkdir(parents=True)
        path.write_text('<robot name="test"/>')
    monkeypatch.setattr(urdf_utils, '_repo_robot_model_dirs', lambda: [])
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'venv'))
    # Exercise ordinary pip fallback even on a ROS-enabled test machine.
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', None)
    monkeypatch.setenv('HOME', str(tmp_path / 'no-home'))
    monkeypatch.delenv('LINKERHAND_URDF_DIR', raising=False)
    monkeypatch.setattr(urdf_utils, '_URDF_SEARCH_DIRS', urdf_utils._default_urdf_search_dirs())
    assert urdf_utils.resolve_urdf_path('o6', 'right') == str(share / MODEL)
    monkeypatch.setenv('LINKERHAND_URDF_DIR', str(external))
    urdf_utils.refresh_search_dirs()
    assert urdf_utils.resolve_urdf_path('o6', 'right') == str(external / MODEL)
    assert urdf_utils.resolve_urdf_path('o6', 'right', str(share / MODEL)) == str(share / MODEL)


def test_colcon_model_share_is_discovered_outside_python_prefix(tmp_path, monkeypatch):
    share = tmp_path / 'colcon/install/hand_gesture_primitives/share/hand_gesture_primitives'
    model = share / 'urdf' / MODEL
    model.parent.mkdir(parents=True)
    model.write_text('<robot name="test"/>')
    packages = ModuleType('ament_index_python.packages')
    packages.get_package_share_directory = lambda package: str(share)
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', packages)
    monkeypatch.setattr(urdf_utils, '_repo_robot_model_dirs', lambda: [])
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'separate-venv'))
    monkeypatch.delenv('LINKERHAND_URDF_DIR', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path / 'no-home'))
    monkeypatch.setattr(urdf_utils, '_URDF_SEARCH_DIRS', urdf_utils._default_urdf_search_dirs())
    assert urdf_utils.resolve_urdf_path('o6', 'right') == str(model)


def test_isolated_snapshot_includes_models_and_hashes(tmp_path):
    script = ROOT / 'hand_gesture_primitives/scripts/validate_ros_install.py'
    spec = importlib.util.spec_from_file_location('gesture_model_snapshot_probe', script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source = tmp_path / 'checkout'
    for filename in ['hand_gesture_primitives/setup.py', 'messages/example/package.xml',
                     'data/urdf/o6/right/linkerhand_o6_right.urdf', 'data/urdf/o6/right/meshes/finger.STL']:
        path = source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    snapshot = tmp_path / 'snapshot'
    hashes = helper.snapshot_sources(source, snapshot)
    assert 'data/urdf/o6/right/linkerhand_o6_right.urdf' in hashes
    assert 'data/urdf/o6/right/meshes/finger.STL' in hashes
    assert all((snapshot / name).read_bytes() == (source / name).read_bytes() for name in hashes)


@pytest.mark.parametrize('from_sdist', [False, True], ids=['checkout-wheel', 'sdist-wheel'])
def test_wheel_resolves_bundled_model_and_all_meshes_without_checkout(tmp_path, from_sdist):
    checkout = tmp_path / 'checkout'
    package = checkout / 'hand_gesture_primitives'
    ignore = shutil.ignore_patterns('__pycache__', '*.egg-info', '.pytest_cache', '.ruff_cache',
                                   'build', 'install', 'log', 'dist')
    shutil.copytree(ROOT / 'hand_gesture_primitives', package, ignore=ignore)
    shutil.copytree(ROOT / 'data/urdf', checkout / 'data/urdf')
    dist = tmp_path / 'dist'
    if from_sdist:
        run([sys.executable, 'setup.py', 'sdist', '--dist-dir', str(dist)], package)
        archive = next(dist.glob('*.tar.gz'))
        with tarfile.open(archive) as source:
            assert all(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                       for member in source.getmembers())
            assert any(member.name.endswith('data/urdf/' + MODEL.as_posix()) for member in source.getmembers())
            source.extractall(tmp_path / 'unpacked', filter='data')
        package = next((tmp_path / 'unpacked').iterdir())
        manifest = next(package.glob('*.egg-info/SOURCES.txt')).read_text().splitlines()
        assert 'data/urdf/' + MODEL.as_posix() in manifest
        assert all('..' not in Path(name).parts for name in manifest)
        # Ensure wheel construction can only use the archive, not its original checkout.
        shutil.rmtree(checkout)
    run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', str(dist)], package)
    wheel = next(dist.glob('*.whl'))
    with zipfile.ZipFile(wheel) as built:
        prefix = next(name[:-len(MODEL.as_posix())] for name in built.namelist()
                      if name.endswith('share/hand_gesture_primitives/urdf/' + MODEL.as_posix()))
        assert all(name.startswith(prefix) for name in built.namelist() if name.lower().endswith(('.urdf', '.stl')))
        assert sum(name.endswith(MODEL.as_posix()) for name in built.namelist()) == 1
        for resource in (ROOT / 'data/urdf').rglob('*'):
            if resource.is_file():
                assert built.read(prefix + resource.relative_to(ROOT / 'data/urdf').as_posix()) == resource.read_bytes()
    venv = tmp_path / 'installed'
    run([sys.executable, '-m', 'venv', str(venv)], tmp_path)
    python = venv / 'bin/python'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PYTHON', 'PIP_', 'AMENT_', 'COLCON_', 'CMAKE_', 'LINKERHAND_'))}
    environment.update(HOME=str(tmp_path / 'empty-home'), PYTHONDONTWRITEBYTECODE='1')
    run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)], tmp_path, environment)
    # The installed resolver has neither source checkout nor an external ROS/model tree.
    shutil.rmtree(package.parent if not from_sdist else package)
    probe = '''import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
from hand_gesture_primitives.urdf_utils import resolve_urdf_path
model=Path(resolve_urdf_path("o6","right"))
assert model.is_relative_to(Path(sys.prefix)/"share/hand_gesture_primitives/urdf")
meshes=[model.parent/item.attrib["filename"] for item in ET.parse(model).iter("mesh")]
assert meshes and all(path.is_file() for path in meshes)
print(json.dumps({"model":str(model),"mesh_count":len(meshes)}))
'''
    result = json.loads(run([str(python), '-I', '-c', probe], tmp_path, environment))
    assert result['mesh_count'] > 0
