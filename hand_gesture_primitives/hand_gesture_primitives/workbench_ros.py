# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Optional ROS transport for the standalone confirmed gesture client.

Imports remain local so the shared workbench core and Mock host need no ROS install.
Each context owns its ROS context, node and executor; an embedding host can inject
its own context factory into GestureNodeClient instead.
"""

from __future__ import annotations

import threading
import uuid


class RosContext:
    def __init__(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor

        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(args=[], context=self._context)
        self.node = rclpy.create_node(
            "gesture_workbench_" + uuid.uuid4().hex[:8], context=self._context
        )
        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._spin, name="gesture-workbench-ros", daemon=True
        )
        self._thread.start()

    @classmethod
    def acquire(cls):
        return cls()

    def _spin(self):
        while not self._stop.is_set() and self._context.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                if not self._stop.is_set():
                    self.node.get_logger().error("Workbench ROS executor failed")
                return
            self._stop.wait(0.0005)

    @staticmethod
    def sensor_qos():
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

    def publisher(self, kind, topic):
        return self.node.create_publisher(kind, topic, 10)

    def release(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        self._context.try_shutdown()
