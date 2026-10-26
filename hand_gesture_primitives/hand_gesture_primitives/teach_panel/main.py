"""Standalone teach_panel entrypoint; Mock never imports ROS or LeRobot."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(args=None):
    arguments = list(sys.argv[1:] if args is None else args)
    if "--mock" not in arguments:
        from .ros_main import main as ros_main
        return ros_main(args=arguments)
    parser = argparse.ArgumentParser(description="Hand gesture workbench with local simulated feedback")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--pose-dir", default=os.environ.get("HAND_GESTURE_POSE_DIR", ""))
    parser.add_argument("--hand", action="append", default=[], metavar="SIDE:MODEL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--control-rate", type=float, default=30.0)
    parser.add_argument("--observe-only", action="store_true")
    options = parser.parse_args(arguments)
    if not options.pose_dir:
        parser.error("--pose-dir or HAND_GESTURE_POSE_DIR is required")
    os.environ["HAND_GESTURE_POSE_DIR"] = str(Path(options.pose_dir).expanduser().absolute())
    models = {}
    for value in options.hand or ["right:o6"]:
        parts = value.split(":")
        if len(parts) != 2 or parts[0] not in {"left", "right"} or parts[0] in models:
            parser.error("--hand must be a unique left:MODEL or right:MODEL")
        models[parts[0]] = parts[1].lower()
    from .dependencies import check_web_dependencies

    try:
        check_web_dependencies()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    import uvicorn

    from ..pose_library import pose_dir
    from .server import create_workbench_app
    from .workbench import WorkbenchHost

    pose_dir()  # Fail before any runtime starts for invalid or symlinked storage.
    host = WorkbenchHost.mock(models, observe_only=options.observe_only,
                              speed=options.speed, control_rate=options.control_rate)
    try:
        uvicorn.run(create_workbench_app(host), host=options.host, port=options.port)
    finally:
        host.close()


def __getattr__(name):
    if name == "TeachPanelNode":
        from .ros_main import TeachPanelNode
        return TeachPanelNode
    raise AttributeError(name)


if __name__ == "__main__":
    main()
