"""README 手势原语清单防漂移校验。

README §3.1 的计数表与 §3.2~3.4 的原语表格是人工维护的，容易随代码演进失准
（本测试引入前，计数表 4 个手型全部偏低，5 个原语与 O30/G20 两个手型完全缺失）。
此处把 README 与 ``hand_profile`` / ``PRIMITIVE_REGISTRY`` 绑定，使漂移在 CI 暴露。

校验两件事：
  1. 计数表每行与 ``hand_profile.supported_primitives()`` 的长度一致
  2. 每个已注册原语都在 README 表格「指令名称」列出现过

README 定位不到时跳过（例如仅安装了 Python 包、未带仓库源码树）。
"""

import re
from pathlib import Path

import pytest
from hand_gesture_primitives.hand_profile import load_hand_profile
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY

# README 手型标签 → hand_profile 档案名。L20 与 L25 同协议，共用 l25 档案。
HAND_LABEL_TO_PROFILE = {
    "O6": "o6",
    "O20": "o20",
    "L25": "l25",
    "L20": "l25",
    "L6": "l6",
    "O30": "o30",
    "G20": "g20",
}


def _readme_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "README.md"
        # 仓库根 README（而非包内同名文件）：以项目标题为准
        if candidate.is_file() and "手势原语清单" in candidate.read_text(encoding="utf-8"):
            return candidate
    pytest.skip("未找到仓库根 README.md（可能运行于已安装包）")


def _readme_text() -> str:
    return _readme_path().read_text(encoding="utf-8")


def _parse_count_table(text: str) -> dict[str, int]:
    """解析 §3.1 的「手型 / 支持原语数」两列表格。"""
    counts: dict[str, int] = {}
    for label in HAND_LABEL_TO_PROFILE:
        match = re.search(rf"^\|{label}\|(\d+)\|$", text, re.MULTILINE)
        if match:
            counts[label] = int(match.group(1))
    return counts


def _readme_primitive_names(text: str) -> set[str]:
    """取 §3.2~3.4 表格第二列的指令名，还原 Markdown 转义的下划线。"""
    names: set[str] = set()
    for row in re.finditer(r"^\|\d+\|([^|]+)\|", text, re.MULTILINE):
        names.add(row.group(1).replace("\\_", "_").strip())
    return names


def test_count_table_covers_all_hands() -> None:
    counts = _parse_count_table(_readme_text())
    missing = sorted(set(HAND_LABEL_TO_PROFILE) - set(counts))
    assert not missing, f"README 计数表缺少手型: {missing}"


def test_counts_match_hand_profiles() -> None:
    counts = _parse_count_table(_readme_text())
    mismatched = []
    for label, documented in sorted(counts.items()):
        actual = len(load_hand_profile(HAND_LABEL_TO_PROFILE[label]).supported_primitives())
        if documented != actual:
            mismatched.append(f"{label}: README={documented} 实际={actual}")
    assert not mismatched, "README 计数表与 hand_profile 不一致:\n  " + "\n  ".join(mismatched)


def test_every_registered_primitive_is_documented() -> None:
    documented = _readme_primitive_names(_readme_text())
    missing = sorted(set(PRIMITIVE_REGISTRY) - documented)
    assert not missing, f"以下已注册原语未出现在 README 表格中: {missing}"


def test_no_unknown_primitive_documented() -> None:
    """README 不应列出已删除/改名的原语。"""
    documented = _readme_primitive_names(_readme_text())
    unknown = sorted(documented - set(PRIMITIVE_REGISTRY))
    assert not unknown, f"README 列出的原语未注册（可能已改名或删除）: {unknown}"
