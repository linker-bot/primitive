"""Ruff 门禁 — 取代原 ament_flake8 / ament_pep257 / ament_copyright。

风格权威为仓库根 ``pyproject.toml`` 的 ``[tool.ruff]``（99 列、select E/F/I），
ament 默认规则（79 列等）与仓库配置不同，统一使用 Ruff 检查。

**扫描范围限定为 ``hand_gesture_primitives/``**（本包）。仓库内 ``perception/``
等既有目录尚有历史遗留问题（F821/F841/F401/E402 等），不纳入本门禁，
以免其历史债长期阻塞手势原语包的检查。仓库整体状态可用
``ruff check --statistics .`` 单独评估。

ruff 未安装时跳过，不制造无法修复的红灯。
"""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Find the Ruff configuration, skipping the package build-system pyproject."""
    for parent in Path(__file__).resolve().parents:
        config = parent / "pyproject.toml"
        if config.is_file() and "ruff" in tomllib.loads(config.read_text()).get("tool", {}):
            return parent
    pytest.skip("未找到仓库根 pyproject.toml（可能运行于已安装包）")


def _find_ruff() -> str | None:
    """定位 ruff：优先 PATH，其次当前解释器同目录（conda/venv 常见未激活 PATH）。"""
    found = shutil.which("ruff")
    if found:
        return found
    sibling = Path(sys.executable).parent / "ruff"
    return str(sibling) if sibling.is_file() else None


def test_ruff() -> None:
    ruff = _find_ruff()
    if ruff is None:
        pytest.skip("ruff 未安装：pip install ruff")

    root = _repo_root()
    result = subprocess.run(
        [ruff, "check", "--output-format=concise", "hand_gesture_primitives"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff 发现问题:\n{result.stdout}\n{result.stderr}"
