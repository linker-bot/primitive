"""teach_panel 入口：ROS spin 线程 + uvicorn Web 服务。"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import rclpy
from rclpy.node import Node

from ..hand_config import HandConfig
from ..joint_coupling import tip_mid_couple_map
from ..topic_defaults import side_hand_topics
from .configuration import resolve_hands
from .controller import PanelController
from .layout_convert import panel_layout_hand_key
from .ros_bridge import RosBridge

# fastapi/uvicorn 为可选 extra；勿在模块顶层 import .server。

if TYPE_CHECKING:
    from .server import TeachPanelApp

# Web 广播限频：joint_state 100 Hz 会压死 asyncio 事件循环，限制到 20 Hz。
_WEB_SNAPSHOT_HZ = 20.0
_WEB_SNAPSHOT_INTERVAL = 1.0 / _WEB_SNAPSHOT_HZ


class TeachPanelNode(Node):
    def __init__(self) -> None:
        super().__init__("teach_panel")
        self.declare_parameter("hand_side", "left")
        self.declare_parameter("hand_type", "l25")
        self.declare_parameter("hand_joint", "")
        self.declare_parameter("cmd_topic", "")
        self.declare_parameter("status_topic", "")
        self.declare_parameter("result_topic", "")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("cmd_backend", "")
        self.declare_parameter("hand_cmd_topic", "")
        # 与机载 gesture / FCC 默认一致；遗留 SDK 直连可显式设 hardware
        self.declare_parameter("ros2_control_command_units", "radians")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("g20_joint_order", "")
        self.declare_parameter("publish_rate", 15.0)
        self.declare_parameter("status_timeout_sec", 3.0)
        self.declare_parameter("http_host", "127.0.0.1")
        self.declare_parameter("http_port", 8001)
        self.declare_parameter("primitive_speed", 1.0)
        self.declare_parameter("control_rate", 30.0)
        self.declare_parameter("observe_only", False)
        self.declare_parameter("workbench_hands", [""])

        side = self.get_parameter("hand_side").value.strip().lower()
        hand_type = self.get_parameter("hand_type").value.strip()
        hand_joint = self.get_parameter("hand_joint").value.strip()
        if hand_joint:
            hand_type = hand_joint
        self.side, models, primary_topics = resolve_hands(
            side, hand_type, self.get_parameter("workbench_hands").value,
            {"gesture_cmd": self.get_parameter("cmd_topic").value,
             "gesture_status": self.get_parameter("status_topic").value,
             "gesture_result": self.get_parameter("result_topic").value},
        )
        self.product_hand_key = models[self.side]
        robot_description = str(self.get_parameter("robot_description").value or "")
        g20_joint_order = str(self.get_parameter("g20_joint_order").value or "")
        # 布局仅影响滑条/耦合；姿态库与 gesture 执行仍用 product
        self.layout_hand_key = panel_layout_hand_key(
            self.product_hand_key,
            g20_joint_order=g20_joint_order,
            robot_description=robot_description,
            hand_side=self.side,
        )
        self.layout_hand = HandConfig(self.layout_hand_key)

        cmd_topic = primary_topics["gesture_cmd"]
        status_topic = primary_topics["gesture_status"]
        result_topic = primary_topics["gesture_result"]
        state_topic = self.get_parameter("state_topic").value.strip() or "/joint_states"
        motion_progress_topic = primary_topics["motion_progress"]

        self._lock = threading.Lock()
        self._last_state_broadcast = 0.0
        self._last_status_text = ""
        couple_map = tip_mid_couple_map(self.layout_hand_key, list(self.layout_hand.joint_names))
        self.controller = PanelController(
            self.layout_hand.num_joints,
            publish_rate=float(self.get_parameter("publish_rate").value),
            status_timeout_sec=float(self.get_parameter("status_timeout_sec").value),
            couple_map=couple_map,
        )
        from ..workbench_client import GestureNodeClient
        from ..workbench_core import JointAdapter
        from .workbench import HandRuntime, WorkbenchHost

        self.bridges, hands = {}, {}
        self._feedback_adapters = {}
        observe_only = bool(self.get_parameter("observe_only").value)
        speed = float(self.get_parameter("primitive_speed").value)
        rate = float(self.get_parameter("control_rate").value)
        try:
            for current_side, model in models.items():
                layout_key = panel_layout_hand_key(model, g20_joint_order=g20_joint_order,
                    robot_description=robot_description, hand_side=current_side)
                defaults = side_hand_topics(current_side)
                # The existing single-hand overrides apply to the primary hand.
                primary = current_side == self.side
                topic_map = {
                    "gesture_cmd": cmd_topic if primary else defaults["gesture_cmd"],
                    "gesture_status": status_topic if primary else defaults["gesture_status"],
                    "gesture_result": result_topic if primary else defaults["gesture_result"],
                    "motion_progress": motion_progress_topic if primary else defaults["motion_progress"],
                }
                bridge = RosBridge(
                    self, hand_side=current_side, product_hand_key=model, layout_hand_key=layout_key,
                    layout_hand=HandConfig(layout_key), cmd_topic=topic_map["gesture_cmd"],
                    status_topic=topic_map["gesture_status"], result_topic=topic_map["gesture_result"],
                    state_topic=state_topic,
                    cmd_backend=str(self.get_parameter("cmd_backend").value or ""),
                    hand_cmd_topic=str(self.get_parameter("hand_cmd_topic").value or "") if primary else "",
                    ros2_control_command_units=str(self.get_parameter("ros2_control_command_units").value or "radians"),
                    robot_description=robot_description, g20_joint_order=g20_joint_order,
                    on_status=lambda _text: None, on_result=lambda _text: None,
                    on_state=lambda positions, selected=current_side: self._workbench_feedback(selected, positions),
                    motion_progress_topic=topic_map["motion_progress"], feedback_only=True,
                    on_controller_state=lambda positions, selected=current_side: self._workbench_controller_feedback(selected, positions),
                )
                converter = bridge.radian_converter
                if converter is None:
                    if not robot_description:
                        raise ValueError("The workbench requires robot_description for controller-radian conversion")
                    converter = JointAdapter.from_urdf(model, current_side, robot_description).converter
                names = [name.split("_hand_", 1)[-1] for name in converter.joint_names]
                lower = [value[0] for value in converter.limits]
                upper = [value[1] for value in converter.limits]
                adapter = JointAdapter(model, current_side, names, lower, upper, converter)
                self._feedback_adapters[current_side] = adapter, HandConfig(layout_key)
                client = GestureNodeClient(current_side, speed_scale=speed,
                                           topics=topic_map, configure=not observe_only)
                hands[current_side] = HandRuntime(adapter, client=client, layout_hand_type=layout_key)
                self.bridges[current_side] = bridge
            self.workbench = WorkbenchHost(hands, backend="ros", observe_only=observe_only,
                                          speed=speed, control_rate=rate, default_side=self.side)
        except Exception:
            for runtime in hands.values():
                runtime.client.close()
            raise
        self.bridge = self.bridges[self.side]

        from .server import TeachPanelApp

        self.web = TeachPanelApp(
            self.controller, hand_type=self.product_hand_key, layout_hand_type=self.layout_hand_key,
            hand_side=self.side, joint_names=list(self.layout_hand.joint_names), lock=self._lock,
            radian_converter=hands[self.side].adapter.converter, workbench=self.workbench,
        )
        self.create_timer(0.05, self._on_timer)
        host = str(self.get_parameter("http_host").value)
        port = int(self.get_parameter("http_port").value)
        layout_note = (
            f" layout={self.layout_hand_key}"
            if self.layout_hand_key != self.product_hand_key
            else ""
        )
        self.get_logger().info(
            f"Teach Web: http://{host}:{port}/  "
            f"hand={self.product_hand_key}{layout_note} side={self.side} "
            f"joints={self.layout_hand.num_joints} "
            f"cmd={cmd_topic} status={status_topic}"
        )

    def _workbench_controller_feedback(self, side, positions):
        self.workbench.update_feedback(side, positions)

    def _workbench_feedback(self, side, positions):
        adapter, layout = self._feedback_adapters[side]
        try:
            q = adapter.converter.semantic_to_radians(layout.from_hardware(positions))
            self.workbench.update_feedback(side, q)
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Rejected {side} hand feedback: {exc}")

    def _on_status(self, text: str) -> None:
        with self._lock:
            self.controller.on_status(text)
        if text != self._last_status_text:
            self._last_status_text = text
            self.web.notify_snapshot()

    def _on_state(self, positions) -> None:
        with self._lock:
            self.controller.update_feedback(positions)
        now = time.monotonic()
        if now - self._last_state_broadcast >= _WEB_SNAPSHOT_INTERVAL:
            self._last_state_broadcast = now
            self.web.notify_snapshot()

    def _on_result(self, text: str) -> None:
        with self._lock:
            self.controller.on_result(text)
        self.web.notify_snapshot()

    def _on_timer(self) -> None:
        with self._lock:
            self.controller.tick()


def _run_uvicorn(app: TeachPanelApp, host: str, port: int) -> None:
    import asyncio

    import uvicorn

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app.set_event_loop(loop)

    config = uvicorn.Config(app=app.app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    logging.getLogger(__name__).info("uvicorn starting on %s:%d ...", host, port)
    loop.run_until_complete(server.serve())


def main(args=None) -> None:
    from .dependencies import check_web_dependencies

    try:
        check_web_dependencies()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    rclpy.init(args=args)
    node = TeachPanelNode()
    host = str(node.get_parameter("http_host").value)
    port = int(node.get_parameter("http_port").value)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        _run_uvicorn(node.web, host, port)
    except KeyboardInterrupt:
        pass
    finally:
        node.workbench.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
