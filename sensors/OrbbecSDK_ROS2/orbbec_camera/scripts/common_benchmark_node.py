#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support: ROS2
name: common_benchmark_node.py
function: A ROS2 node to monitor and log the performance of an Orbbec camera node:
          frame rates, delays, CPU and RAM usage, packet/frame loss statistics.
usage:
    ros2 run orbbec_camera common_benchmark_node.py --run_time 20 --csv_file /tmp/cam_log.csv
                    You can also pass an ideal frame rate for drop detection: --ideal_fps 30
                    Monitor multiple cameras: --camera_names camera,camera01
"""

import argparse
import rclpy
from rclpy.node import Node
import psutil
import time
import csv
import os
from collections import defaultdict
from orbbec_camera_msgs.msg import DeviceStatus
from sensor_msgs.msg import Image
import sys

from tabulate import tabulate

CAMERA_NODE_NAMES = ["component_container", "orbbec_camera_node", "nodelet"]

# ----------------tool functions----------------
def parse_duration(s):
    # Parse duration strings like "10s", "5m", "1h", "2d" into seconds.
    if isinstance(s, (int, float)):
        return float(s)

    s = str(s).strip().lower()
    if s.endswith("s"):
        return float(s[:-1])
    elif s.endswith("m"):
        return float(s[:-1]) * 60
    elif s.endswith("h"):
        return float(s[:-1]) * 3600
    elif s.endswith("d"):
        return float(s[:-1]) * 86400
    else:
        return float(s)

def format_duration(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)

def parse_camera_names(camera_names):
    if isinstance(camera_names, str):
        names = camera_names.replace(";", ",").split(",")
    else:
        names = camera_names or []

    parsed_names = []
    for name in names:
        normalized = str(name).strip().strip("/")
        if normalized and normalized not in parsed_names:
            parsed_names.append(normalized)
    return parsed_names or ["camera"]

def make_stat():
    return {"cur": 0.0, "avg": 0.0, "min": float("inf"), "max": float("-inf"), "count": 0, "sum": 0.0}
# ----------------------------------------------

class TopicTracker:
    def __init__(self, logger=None):
        self.received = 0

        self.last_time = None
        self.drop_frames = 0

        self.logger = logger

    def on_msg(self, header, avg_fps):
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self.received += 1

        if self.last_time is not None and avg_fps > 0:
            dt = stamp - self.last_time
            expected_interval = 1.0 / avg_fps
            if expected_interval > 0 and dt > 1.5 * expected_interval:
                self.drop_frames += 1

        self.last_time = stamp

    def frames_loss_rate(self):
        total = self.received + self.drop_frames
        if total <= 0:
            return 0.0
        return float(self.drop_frames) / total

    def reset(self):
        self.__init__(logger=self.logger)


class CameraMonitorNode(Node):
    def __init__(self, run_time, csv_file="camera_monitor_log.csv", ideal_fps: float = 0.0, camera_names=None):
        super().__init__("camera_monitor_node")

        self.run_time = run_time
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.first_data_collected = False
        self.camera_names = parse_camera_names(camera_names)
        self.node_names = {camera_name: "Not Found" for camera_name in self.camera_names}
        self.total_node_name = "Not Found"
        # If > 0, use this ideal fps value for drop-frame detection instead of the reported average
        self.ideal_fps = float(ideal_fps) if ideal_fps is not None else 0.0
        self.finished = False

        self.cameras = {}
        for camera_name in self.camera_names:
            self.cameras[camera_name] = {
                "connection_type": None,
                "disconnect_count": 0,
                "prev_online": True,
                "data_collected": False,
                "stats": defaultdict(make_stat),
                "cpu_stats": make_stat(),
                "ram_stats": make_stat(),
                "trackers": {
                    "color": TopicTracker(logger=self.get_logger()),
                    "depth": TopicTracker(logger=self.get_logger())
                }
            }

        self.total_cpu_stats = make_stat()
        self.total_ram_stats = make_stat()

        # CSV
        self.csv_file = csv_file
        self.csv_fh = open(self.csv_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_fh)
        self.csv_writer.writerow(self.build_csv_header())

        # subscriptions
        for camera_name in self.camera_names:
            ns = self.camera_namespace(camera_name)
            self.create_subscription(
                DeviceStatus,
                f"{ns}/device_status",
                lambda msg, name=camera_name: self.status_callback(msg, name),
                5
            )
            self.create_subscription(
                Image,
                f"{ns}/color/image_raw",
                lambda msg, name=camera_name: self.image_callback(msg, name, "color"),
                5
            )
            self.create_subscription(
                Image,
                f"{ns}/depth/image_raw",
                lambda msg, name=camera_name: self.image_callback(msg, name, "depth"),
                5
            )

        # timer runs every 1s to update system stats, log csv and print status
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.run_time:
            self.finish()
            rclpy.shutdown()
            return

        camera_sys_stats, total_cpu, total_ram, self.total_node_name = self.get_camera_stats()
        for camera_name in self.camera_names:
            camera = self.cameras[camera_name]
            cpu, ram, node_name = camera_sys_stats.get(camera_name, (0.0, 0.0, "Not Found"))
            self.node_names[camera_name] = node_name
            self.update_sys_stat(camera["cpu_stats"], cpu, camera["prev_online"])
            self.update_sys_stat(camera["ram_stats"], ram, camera["prev_online"])

        self.update_sys_stat(self.total_cpu_stats, total_cpu, True)
        self.update_sys_stat(self.total_ram_stats, total_ram, True)

        if self.first_data_collected:
            self.log_to_csv(elapsed)
            self.print_status()

    def finish(self):
        if self.finished:
            return
        self.finished = True

        elapsed = time.time() - self.start_time
        try:
            self.csv_fh.close()
        except Exception:
            pass
        print(f"Monitoring finished, it takes time: {format_duration(elapsed)}")
        print(f"CSV data is saved to: {self.csv_file}")

    def camera_namespace(self, camera_name):
        return "/" + camera_name.strip("/")

    def cmdline_has_camera_namespace(self, cmdline_args, camera_name):
        ns = self.camera_namespace(camera_name)
        candidates = [
            f"__ns:={ns}",
            f"__ns:={camera_name}",
            f"namespace:={ns}",
            f"namespace:={camera_name}",
        ]
        return any(arg in candidates for arg in cmdline_args)

    def find_camera_nodes(self):
        found = {camera_name: [] for camera_name in self.camera_names}
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline_args = proc.info.get('cmdline') or []
                cmdline = " ".join(cmdline_args)
                if not any(name.lower() in cmdline.lower() for name in CAMERA_NODE_NAMES):
                    continue
                for camera_name in self.camera_names:
                    if self.cmdline_has_camera_namespace(cmdline_args, camera_name):
                        found[camera_name].append(proc)
            except Exception:
                continue
        return found

    def get_camera_stats(self):
        found = self.find_camera_nodes()
        camera_stats = {}
        total_cpu = 0.0
        total_ram = 0.0
        total_proc_count = 0

        for camera_name, root_procs in found.items():
            if not root_procs:
                camera_stats[camera_name] = (0.0, 0.0, "Not Found")
                continue

            seen_pids = set()
            procs = []
            for proc in root_procs:
                try:
                    proc_group = [proc] + proc.children(recursive=True)
                    for p in proc_group:
                        if p.pid not in seen_pids:
                            seen_pids.add(p.pid)
                            procs.append(p)
                except Exception:
                    continue

            try:
                cpu = sum((p.cpu_percent(interval=None) for p in procs)) / max(1, psutil.cpu_count())
                mem_bytes = sum((p.memory_info().rss for p in procs))
                mem_mb = mem_bytes / (1024 * 1024)
                root_names = ", ".join(f"{p.name()}[{p.pid}]" for p in root_procs)
                if len(procs) > len(root_procs):
                    root_names = f"{root_names} + {len(procs) - len(root_procs)} child"
                camera_stats[camera_name] = (cpu, mem_mb, root_names)
                total_cpu += cpu
                total_ram += mem_mb
                total_proc_count += len(procs)
            except Exception:
                camera_stats[camera_name] = (0.0, 0.0, "Error")

        total_node_name = f"{total_proc_count} matched process(es)" if total_proc_count > 0 else "Not Found"
        return camera_stats, total_cpu, total_ram, total_node_name

    def status_callback(self, msg: DeviceStatus, camera_name: str):
        if not self.first_data_collected:
            self.first_data_collected = True

        camera = self.cameras[camera_name]
        camera["data_collected"] = True
        camera["connection_type"] = msg.connection_type
        if camera["prev_online"] and not msg.device_online:
            camera["disconnect_count"] += 1
            camera["prev_online"] = msg.device_online
            return

        camera["prev_online"] = msg.device_online

        # update stats from DeviceStatus message fields
        self.update_stats(camera["stats"], "color_fps", msg.color_frame_rate_cur, msg.color_frame_rate_min, msg.color_frame_rate_max, msg.color_frame_rate_avg)
        self.update_stats(camera["stats"], "color_delay", msg.color_delay_ms_cur, msg.color_delay_ms_min, msg.color_delay_ms_max, msg.color_delay_ms_avg)
        self.update_stats(camera["stats"], "depth_fps", msg.depth_frame_rate_cur, msg.depth_frame_rate_min, msg.depth_frame_rate_max, msg.depth_frame_rate_avg)
        self.update_stats(camera["stats"], "depth_delay", msg.depth_delay_ms_cur, msg.depth_delay_ms_min, msg.depth_delay_ms_max, msg.depth_delay_ms_avg)

    def image_callback(self, msg: Image, camera_name: str, stream: str):
        if stream not in ("color", "depth"):
            return
        header = msg.header
        camera = self.cameras[camera_name]
        tracker = camera["trackers"][stream]
        # Prefer a user-specified ideal fps for drop detection when provided.
        fps_to_use = self.ideal_fps if (self.ideal_fps and self.ideal_fps > 0.0) else camera["stats"][f"{stream}_fps"]["avg"]
        tracker.on_msg(header, fps_to_use)

    def update_stats(self, stats, key, cur, min_val, max_val, avg_val):
        if min_val <= 1e-3 or avg_val < 0:  # ignore invalid data
            return
        s = stats[key]
        s["cur"] = (cur)
        s["count"] += 1
        s["sum"] += avg_val
        s["avg"] = s["sum"] / s["count"] if s["count"] > 0 else 0.0
        s["min"] = min(s["min"], min_val)
        s["max"] = max(s["max"], max_val)

    def update_sys_stat(self, stat_dict, value, online=True):
        stat_dict["cur"] = value
        if value is None or value <= 0.0 or not online:
            return

        stat_dict["count"] += 1
        stat_dict["sum"] += value
        stat_dict["avg"] = stat_dict["sum"] / stat_dict["count"] if stat_dict["count"] > 0 else 0.0
        stat_dict["min"] = min(stat_dict["min"], value)
        stat_dict["max"] = max(stat_dict["max"], value)

    def log_to_csv(self, elapsed):
        row = [round(elapsed, 2)]
        for camera_name in self.camera_names:
            camera = self.cameras[camera_name]
            row.extend(self.build_camera_csv_values(camera))

        row.extend([
            round(self.total_cpu_stats["cur"], 2), round(self.total_cpu_stats["avg"], 2),
            self.format_csv_number(self.total_cpu_stats["min"]), self.format_csv_number(self.total_cpu_stats["max"]),
            round(self.total_ram_stats["cur"], 2), round(self.total_ram_stats["avg"], 2),
            self.format_csv_number(self.total_ram_stats["min"]), self.format_csv_number(self.total_ram_stats["max"]),
        ])
        self.csv_writer.writerow(row)

    def build_csv_header(self):
        header = ["time(s)"]
        camera_fields = [
            "connection_type", "status_online", "disconnects",
            "color_fps_cur", "color_fps_avg", "color_fps_min", "color_fps_max",
            "color_delay_cur", "color_delay_avg", "color_delay_min", "color_delay_max",
            "depth_fps_cur", "depth_fps_avg", "depth_fps_min", "depth_fps_max",
            "depth_delay_cur", "depth_delay_avg", "depth_delay_min", "depth_delay_max",
            "cpu_cur", "cpu_avg", "cpu_min", "cpu_max",
            "ram_cur", "ram_avg", "ram_min", "ram_max",
            "color_frames_loss", "color_frames_loss_rate(%)",
            "depth_frames_loss", "depth_frames_loss_rate(%)"
        ]
        for camera_name in self.camera_names:
            header.extend([f"{camera_name}_{field}" for field in camera_fields])

        header.extend([
            "total_cpu_cur", "total_cpu_avg", "total_cpu_min", "total_cpu_max",
            "total_ram_cur", "total_ram_avg", "total_ram_min", "total_ram_max",
        ])
        return header

    def build_camera_csv_values(self, camera):
        color_tracker = camera["trackers"]["color"]
        depth_tracker = camera["trackers"]["depth"]

        def safe(k):
            v = camera["stats"].get(k, {})
            return (
                round(v.get("cur", 0.0), 2),
                round(v.get("avg", 0.0), 2),
                self.format_csv_number(v.get("min", 0.0)),
                self.format_csv_number(v.get("max", 0.0)),
            )

        if not camera["prev_online"]:
            return [
                camera["connection_type"], camera["prev_online"], camera["disconnect_count"],
                *["N/A"] * 16,
                round(camera["cpu_stats"]["cur"], 2), "N/A", "N/A", "N/A",
                round(camera["ram_stats"]["cur"], 2), "N/A", "N/A", "N/A",
                color_tracker.drop_frames, round(color_tracker.frames_loss_rate() * 100.0, 3),
                depth_tracker.drop_frames, round(depth_tracker.frames_loss_rate() * 100.0, 3)
            ]

        return [
            camera["connection_type"], camera["prev_online"], camera["disconnect_count"],
            *safe("color_fps"),
            *safe("color_delay"),
            *safe("depth_fps"),
            *safe("depth_delay"),
            round(camera["cpu_stats"]["cur"], 2), round(camera["cpu_stats"]["avg"], 2),
            self.format_csv_number(camera["cpu_stats"]["min"]), self.format_csv_number(camera["cpu_stats"]["max"]),
            round(camera["ram_stats"]["cur"], 2), round(camera["ram_stats"]["avg"], 2),
            self.format_csv_number(camera["ram_stats"]["min"]), self.format_csv_number(camera["ram_stats"]["max"]),
            color_tracker.drop_frames, round(color_tracker.frames_loss_rate() * 100.0, 3),
            depth_tracker.drop_frames, round(depth_tracker.frames_loss_rate() * 100.0, 3)
        ]

    def format_csv_number(self, value):
        if value == float("inf") or value == float("-inf"):
            return 0.0
        return round(value, 2)

    def print_status(self):
        def format_stats(s):
            if s["count"] <= 0:
                return "0.00", "0.00", "0.00", "0.00"
            return f"{s['cur']:.2f}", f"{s['avg']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}"

        rows = []
        for camera_name in self.camera_names:
            camera = self.cameras[camera_name]
            for stream in ["color", "depth"]:
                fps_key = f"{stream}_fps"
                delay_key = f"{stream}_delay"
                topic_name = f"/{camera_name}/{stream}/image_raw"
                if not camera["prev_online"]:
                    rows.append([camera_name, topic_name, *["N/A"] * 10])
                else:
                    fps_vals = format_stats(camera["stats"][fps_key])
                    delay_vals = format_stats(camera["stats"][delay_key])
                    tracker = camera["trackers"][stream]

                    frames_loss = tracker.drop_frames
                    frames_loss_rate = round(tracker.frames_loss_rate() * 100.0, 3)
                    rows.append([camera_name, topic_name, *fps_vals, *delay_vals, frames_loss, frames_loss_rate])

        header_bottom = ["Camera", "Topic", "fps_cur", "fps_avg", "fps_min", "fps_max", "delay_cur(ms)", "delay_avg(ms)", "delay_min(ms)", "delay_max(ms)", "Pub_lost_count", "Pub_lost_rate(%)"]

        os.system("clear")
        print("Orbbec Camera Benchmark\n")
        print(tabulate([header_bottom] + rows, tablefmt="fancy_grid"))

        sys_rows = []
        for camera_name in self.camera_names:
            camera = self.cameras[camera_name]
            if not camera["prev_online"]:
                cpu_vals = (f"{camera['cpu_stats']['cur']:.2f}", "N/A", "N/A", "N/A")
                ram_vals = (f"{camera['ram_stats']['cur']:.2f}", "N/A", "N/A", "N/A")
            else:
                cpu_vals = format_stats(camera["cpu_stats"])
                ram_vals = format_stats(camera["ram_stats"])

            sys_rows.append([camera_name, "CPU Usage (%)", *cpu_vals, self.node_names[camera_name]])
            sys_rows.append([camera_name, "RAM Usage (MB)", *ram_vals, self.node_names[camera_name]])

        sys_rows.append(["TOTAL", "CPU Usage (%)", *format_stats(self.total_cpu_stats), self.total_node_name])
        sys_rows.append(["TOTAL", "RAM Usage (MB)", *format_stats(self.total_ram_stats), self.total_node_name])

        print("\n\n(CPU & RAM)\n")
        print(tabulate(sys_rows, headers=["Camera", "Option", "cur", "avg", "min", "max", "Camera Node"], tablefmt="fancy_grid"))

        status_rows = []
        for camera_name in self.camera_names:
            camera = self.cameras[camera_name]
            status_rows.append([camera_name, camera["connection_type"], camera["prev_online"], camera["disconnect_count"]])
        print("\n")
        print(tabulate(status_rows, headers=["Camera", "connection_type", "status_online", "disconnect_count"], tablefmt="fancy_grid"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_time", type=str, default="10s", help="Total run time for monitoring, e.g., 10s, 5m, 1h.")
    parser.add_argument("--csv_file", type=str, default="camera_monitor_log.csv")
    parser.add_argument("--ideal_fps", type=float, default=0.0, help="Optional ideal frame rate to use for drop detection (overrides reported avg).")
    parser.add_argument("--camera_names", type=str, default="camera", help="Comma-separated camera namespaces, e.g., camera,camera01,camera02.")
    cli_args, _ = parser.parse_known_args(argv)

    rclpy.init(args=argv)
    run_time = parse_duration(cli_args.run_time)
    node = CameraMonitorNode(run_time, cli_args.csv_file, ideal_fps=cli_args.ideal_fps, camera_names=cli_args.camera_names)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.finish()
    finally:
        try:
            node.csv_fh.close()
        except Exception:
            pass
        node.destroy_node()

if __name__ == "__main__":
    main()
