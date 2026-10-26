"""Validate the Web stack before starting ROS or a workbench control thread."""

from __future__ import annotations

import importlib
import sys


def check_web_dependencies() -> None:
    """Require Pydantic 2 in the interpreter that will run the panel."""
    guidance = (
        f"teach_panel interpreter: {sys.executable}. "
        "Install hand_gesture_primitives[teach] in this environment. "
        "For ROS 2 Jazzy, follow docs/workbench.md and use the system-Python venv "
        "for both colcon and runtime; Ubuntu 24.04 apt python3-pydantic is v1."
    )
    try:
        pydantic = importlib.import_module("pydantic")
        version = str(pydantic.__version__)
        major = int(version.split(".", 1)[0])
    except (ImportError, AttributeError, ValueError) as exc:
        raise RuntimeError(f"teach_panel requires Pydantic >=2,<3. {guidance}") from exc
    if major != 2:
        raise RuntimeError(f"teach_panel requires Pydantic >=2,<3; found {version}. {guidance}")
    for name in ("fastapi", "uvicorn", "websockets"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(f"teach_panel requires {name}. {guidance}") from exc
