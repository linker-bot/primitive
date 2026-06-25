#!/usr/bin/env python3
"""
Detection bbox node: Grounding DINO + SAM2 + hardware depth (optional FoundationStereo).

Publishes tag-labeled 2D and 3D axis-aligned bounding boxes.

Usage:
    ros2 launch robot_perception detection_bbox.launch.py \
        tags:="['2_4_blue_lego']"
"""
import json
import os
import time

import yaml
import cv2
import numpy as np
import torch
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.parameter import ParameterType
from ros_gz_interfaces.msg import StringVec
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
import tf2_ros

from robot_perception.utils.bbox3d_from_depth import (
    aabb_in_frame,
    backproject_mask,
    classify_grasp_type,
    compute_result_geometry_stats,
    compute_top_normal,
    draw_aabb_projection,
    format_geometry_stats_log,
    format_publish_draw_log,
    aabb_projected_bbox_xyxy,
    bbox_area_xyxy,
    mask_to_bbox_xyxy,
    pca_orientation,
    transform_points,
)
from robot_perception.ros_defaults import DEFAULT_CAMERA_NS, DEFAULT_DETECTION_PROMPTS_TOPIC
from robot_perception.utils.gdino_sam import (
    gdino_detect,
    load_gdino_sam_models,
    load_sam2_predictor,
    sam2_segment_boxes,
)
from robot_perception.utils.vlm_detector import VLMDetector, vlm_detect_as_gdino
from robot_perception.utils.scene_prompt_config import load_scene_prompt_config
from robot_perception.utils.rviz_markers import (
    build_camera_marker_array,
    build_hybrid_surface_marker_array,
    build_workbench_marker_array,
    build_workbench_plane_marker_array,
)
from robot_perception.utils.stereo_depth import (
    load_stereo_engine,
    run_stereo_trt,
    stereo_disp_to_aligned_depth,
)
from robot_perception.utils.bbox_track_manager import BboxTrackManager, VIZ_COLOR_NAMES
from robot_perception.utils.calib_loader import default_calib_file, load_T_world_cam
from robot_perception.utils.paths import (
    default_vlm_serve_model_id,
    SAM2_CHECKPOINT,
    GDINO_CONFIG,
    GDINO_CHECKPOINT,
    STEREO_PLAN_PATH,
    PIXEL3D_MODEL_PATH,
    MINICPM_WEIGHT_DIR,
)
from robot_perception.utils.static_tf import publish_static_tf_once
from robot_perception.utils.world_roi import (
    apply_world_roi_to_aabbs,
    get_effective_workbench_z,
    roi_mode_label,
)
from robot_perception.utils.workbench_plane import WorkbenchPlaneEstimator
from robot_perception.utils.surface_mesh import attach_hybrid_surface
from robot_perception.constants import (
    DEFAULT_INDUSTRY_SCENE_PROMPTS,
    DEFAULT_LEGO_SCENE_PROMPTS,
    DEFAULT_OPEN_SCENE_PROMPTS,
)
from robot_perception.utils.tag_mapping import (
    build_combined_caption,
    build_prompt_to_label,
    match_phrase_to_label,
    normalize_prompt_key,
    resolve_detection_targets,
    targets_from_scene_prompts,
    targets_from_text_prompts,
)

try:
    from robot_perception_msgs.msg import (
        LabeledBBox2D,
        LabeledBBox2DArray,
        LabeledBBox3D,
        LabeledBBox3DArray,
        SurfaceMesh,
        SurfaceMeshArray,
        WorkbenchPlane,
    )
except ImportError as exc:
    raise ImportError(
        'robot_perception_msgs bbox messages not found — build robot_perception_msgs first'
    ) from exc

try:
    from visualization_msgs.msg import MarkerArray
except ImportError as exc:
    raise ImportError(
        'visualization_msgs not found — install ros-${ROS_DISTRO}-visualization-msgs'
    ) from exc


def _get_string_list_param(node, name):
    """Read a string-list parameter; tolerate launch YAML string overrides."""
    param = node.get_parameter(name)
    pvalue = param.get_parameter_value()
    if pvalue.type == ParameterType.PARAMETER_STRING_ARRAY:
        return [s for s in pvalue.string_array_value if s]
    if pvalue.type == ParameterType.PARAMETER_STRING:
        raw = pvalue.string_value.strip()
        if not raw:
            return []
        parsed = yaml.safe_load(raw)
        if parsed is None:
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x]
        return [str(parsed)]
    return []


OBJECT_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def _fill_bbox3d_msg(msg, label, prompt, score, instance_id, frame_id, aabb,
                     orientation=None, top_normal=None, grasp_type='',
                     track_mode=''):
    msg.label = label
    msg.prompt = prompt
    msg.score = float(score)
    msg.instance_id = int(instance_id)
    msg.frame_id = frame_id
    msg.center = [float(v) for v in aabb['center']]
    msg.size = [float(v) for v in aabb['size']]
    msg.min = [float(v) for v in aabb['min']]
    msg.max = [float(v) for v in aabb['max']]
    if orientation is not None:
        msg.orientation = [float(v) for v in orientation]
    else:
        msg.orientation = [0.0, 0.0, 0.0, 1.0]
    if top_normal is not None:
        msg.top_normal = [float(v) for v in top_normal]
    else:
        msg.top_normal = [0.0, 0.0, 1.0]
    msg.grasp_type = grasp_type
    msg.track_mode = track_mode
    return msg


class CameraBBoxHandler:
    """Per-camera GDINO+SAM+Stereo bbox detection."""

    def __init__(self, cam_ns, node, args, grounding_model, sam2_predictor,
                 stereo_engine, T_world_cam, device, vlm_detector=None):
        self.cam_ns = cam_ns
        self.node = node
        self.args = args
        self.device = device
        self.bridge = CvBridge()
        self.T_world_cam = T_world_cam

        self.grounding_model = grounding_model
        self.sam2_predictor = sam2_predictor
        self.stereo_engine = stereo_engine
        self.vlm_detector = vlm_detector

        self.active_targets = []
        self.text_prompts = []
        self.prompt_to_label = {}

        ns_clean = cam_ns.rstrip('/')
        self.bbox2d_pub = node.create_publisher(
            LabeledBBox2DArray, f'{ns_clean}/detection_bbox/bboxes_2d', 10)
        self.bbox3d_pub = node.create_publisher(
            LabeledBBox3DArray, f'{ns_clean}/detection_bbox/bboxes_3d', 10)
        # Default reliable QoS — compatible with RViz Image display
        self.annotated_pub = node.create_publisher(
            Image, f'{ns_clean}/detection_bbox/annotated', 10)
        self.markers_camera_pub = node.create_publisher(
            MarkerArray, f'{ns_clean}/detection_bbox/markers_camera', 10)
        self.markers_workbench_pub = node.create_publisher(
            MarkerArray, f'{ns_clean}/detection_bbox/markers_workbench', 10)
        self.markers_workbench_plane_pub = node.create_publisher(
            MarkerArray, f'{ns_clean}/detection_bbox/markers_workbench_plane', 10)
        self.markers_surface_pub = node.create_publisher(
            MarkerArray, f'{ns_clean}/detection_bbox/markers_surface', 10)
        self.track_debug_pub = None
        if getattr(args, 'publish_track_debug', False):
            self.track_debug_pub = node.create_publisher(
                String, f'{ns_clean}/detection_bbox/track_debug', 10)
        self.pointcloud_pub = None
        if getattr(args, 'publish_object_pointcloud', False):
            self.pointcloud_pub = node.create_publisher(
                PointCloud2, f'{ns_clean}/detection_bbox/object_points', 10)
        self.mask_pub = None
        if getattr(args, 'publish_mask_image', False):
            self.mask_pub = node.create_publisher(
                Image, f'{ns_clean}/detection_bbox/mask', 10)

        self.surface_mesh_pub = None
        if getattr(args, 'publish_surface_mesh_topic', True):
            self.surface_mesh_pub = node.create_publisher(
                SurfaceMeshArray, f'{ns_clean}/detection_bbox/surface_meshes', 10)
        self.workbench_plane_pub = None
        if getattr(args, 'enable_ransac_workbench_plane', False):
            self.workbench_plane_pub = node.create_publisher(
                WorkbenchPlane, f'{ns_clean}/detection_bbox/workbench_plane', 10)

        self.frame_interval = 1.0 / max(args.fps, 0.1)
        self._last_frame_timing: dict = {}
        self._last_publish_timing: dict = {}
        self.last_process_time = 0.0
        self.K = None
        self.frame_count = 0

        self.use_stereo_depth = args.use_stereo_depth and stereo_engine is not None
        self.stereo_baseline = args.stereo_baseline
        self.stereo_depth = None
        self.K_ir = None
        self.T_ir_to_color = None
        self._color_shape = None
        self._color_frame_id = ''

        self._latest_depth_msg = None
        self._latest_info_msg = None
        self.plane_state = None
        self._prev_exclude_mask = None
        self._stereo_depth_stamp = None
        self._static_tf_broadcaster = None
        self._static_tf_published = set()
        self._stereo_depth_logged = False
        self._hardware_depth_logged = False

        self.plane_estimator = None
        if getattr(args, 'enable_ransac_workbench_plane', False) and T_world_cam is not None:
            self.plane_estimator = WorkbenchPlaneEstimator(
                args, T_world_cam, logger=node.get_logger())

        self.track_manager = None
        if args.use_temporal_tracking:
            self.track_manager = BboxTrackManager(
                args=args,
                T_world_cam=T_world_cam,
                object_colors=OBJECT_COLORS,
                logger=node.get_logger(),
                vlm_detector=vlm_detector,
                device=device,
            )
            det_backend = 'VLM' if getattr(args, 'use_vlm_detect', False) else 'GDINO'
            node.get_logger().info(
                f'[{cam_ns}] Temporal tracking enabled (Cutie + {det_backend} fallback)')

        self.scene_manager = None
        if getattr(args, 'use_scene_understand', False) and vlm_detector is not None:
            from robot_perception.utils.scene_understand import SceneUnderstandManager
            self.scene_manager = SceneUnderstandManager(
                vlm_detector=vlm_detector, args=args, logger=node.get_logger())
            node.get_logger().info(f'[{cam_ns}] Scene understanding enabled (VLM-driven)')

        self.pixel3d_manager = None
        if getattr(args, 'use_pixel3d', False):
            from robot_perception.utils.pixel3d_mesh import Pixel3DManager
            self.pixel3d_manager = Pixel3DManager(args=args, logger=node.get_logger())
            node.get_logger().info(f'[{cam_ns}] Pixel3D mesh completion enabled')

        self._user_prompts = []

        ns_clean = cam_ns.rstrip('/')
        color_sub = Subscriber(node, Image, f'{ns_clean}/color/image_raw')
        depth_sub = Subscriber(node, Image, f'{ns_clean}/depth/image_raw')
        info_sub = Subscriber(node, CameraInfo, f'{ns_clean}/color/camera_info')
        self.rgbd_sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub],
            queue_size=5,
            slop=args.slop,
        )
        self.rgbd_sync.registerCallback(self._on_rgbd_frame)

        if self.use_stereo_depth:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

            ir_left_sub = Subscriber(node, Image, f'{ns_clean}/left_ir/image_raw')
            ir_right_sub = Subscriber(node, Image, f'{ns_clean}/right_ir/image_raw')
            ir_info_sub = Subscriber(node, CameraInfo, f'{ns_clean}/left_ir/camera_info')

            self.stereo_sync = ApproximateTimeSynchronizer(
                [ir_left_sub, ir_right_sub, ir_info_sub],
                queue_size=5,
                slop=args.slop,
            )
            self.stereo_sync.registerCallback(self._on_stereo_frame)
            node.get_logger().info(f'[{cam_ns}] Stereo depth enabled for bbox node')

        node.get_logger().info(f'[{cam_ns}] detection_bbox handler ready')

    def set_active_targets(self, targets):
        """Update active detection targets (tag-based or freeform prompts)."""
        self.active_targets = list(targets)
        self.text_prompts = [t['prompt'] for t in self.active_targets]
        self._user_prompts = list(self.text_prompts)
        self.prompt_to_label = build_prompt_to_label(self.active_targets)
        labels = [t['label'] for t in self.active_targets]
        self.node.get_logger().info(
            f'[{self.cam_ns}] Active targets ({len(self.active_targets)}): {labels}')
        if self.track_manager is not None:
            self.track_manager.update_targets(
                self.active_targets, self.text_prompts, self.prompt_to_label)

    def _stable_track_masks(self) -> list:
        """Masks from stable tracks only — drift/lost masks don't block scene change detect."""
        if self.track_manager is None:
            return []
        return [
            t.last_mask for t in self.track_manager.tracks
            if t.last_mask is not None
            and t.lost_count == 0
            and t.mode not in ('drift',)
        ]

    def _scene_prompt_keys(self, merged_prompts: list[str]) -> set[str]:
        """Normalized keys for scene-only prompts (excludes user prompts)."""
        user_keys = {normalize_prompt_key(p) for p in self._user_prompts if p}
        return {
            normalize_prompt_key(p) for p in merged_prompts
            if p and normalize_prompt_key(p) not in user_keys
        }

    def _active_scene_prompt_keys(self) -> set[str]:
        user_keys = {normalize_prompt_key(p) for p in self._user_prompts if p}
        keys = set()
        for t in self.active_targets:
            if t.get('is_tag'):
                continue
            label_key = normalize_prompt_key(t.get('label', ''))
            if label_key and label_key not in user_keys:
                keys.add(label_key)
        return keys

    def _format_prompt_debug(self) -> str:
        """Compact prompt/caption dump for frame logs."""
        labels = [t.get('label', '?') for t in self.active_targets]
        gdino = [t.get('prompt', '?') for t in self.active_targets]
        cap = build_combined_caption(self.text_prompts)
        parts = [f'labels={labels}', f'gdino={gdino}', f'caption="{cap}"']
        if self.scene_manager is not None:
            dbg = self.scene_manager.get_debug_info()
            parts.append(f'scene_objs={dbg.get("cumulative_objects", [])}')
            vlm_meta = dbg.get('last_vlm_meta', {})
            if vlm_meta.get('prompt_snip'):
                parts.append(f'vlm_prompt="{vlm_meta["prompt_snip"]}"')
            if dbg.get('last_vlm_raw'):
                parts.append(f'vlm_raw={dbg.get("last_vlm_raw")}')
            if vlm_meta.get('prior_objects'):
                parts.append(f'vlm_prior={vlm_meta["prior_objects"]}')
            if dbg.get('last_vlm_mode'):
                parts.append(f'vlm_mode={dbg.get("last_vlm_mode")}')
            if self.scene_manager is not None and self.vlm_detector is not None:
                cfg = getattr(self.vlm_detector, 'scene_prompt_config', None)
                if cfg is not None:
                    parts.append(f'vlm_cfg={cfg.source}')
            te = dbg.get('last_trigger_eval', {})
            parts.append(f'scene_vlm={"trigger" if te.get("trigger") else "skip"}:{te.get("reason", "?")}')
            cs = dbg.get('last_change_stats', {})
            if cs:
                parts.append(
                    f'change={cs.get("change_ratio", 0):.3f}'
                    f'/{cs.get("threshold", 0):.2f}'
                    f'({cs.get("changed_pixels", 0)}/{cs.get("untracked_pixels", 0)}px)')
        return ', prompts={' + ', '.join(parts) + '}'

    def _sync_scene_prompts(self, changes: dict | None = None) -> bool:
        """Rebuild GDINO caption from scene objects; add/remove vs user prompts."""
        if self.scene_manager is None:
            return False

        stale_removed = self.scene_manager.prune_stale_objects()
        merged = self.scene_manager.get_merged_prompts(self._user_prompts)
        if not stale_removed and self._scene_prompt_keys(merged) == self._active_scene_prompt_keys():
            return False

        user_targets = targets_from_text_prompts(self._user_prompts)
        scene_only = [p for p in merged if p not in self._user_prompts]
        preserve = bool(getattr(self.args, 'scene_preserve_gdino_prompts', True))
        scene_targets = targets_from_scene_prompts(
            scene_only, preserve_full_prompt=preserve)
        targets = user_targets + scene_targets
        self.active_targets = targets
        self.text_prompts = [t['prompt'] for t in targets]
        self.prompt_to_label = build_prompt_to_label(targets)

        scene_dbg = self.scene_manager.get_debug_info()
        parts = []
        if changes:
            if changes.get('added'):
                parts.append(f"added={changes['added']}")
            if changes.get('removed'):
                parts.append(f"removed={changes['removed']}")
        if stale_removed:
            parts.append(f'stale_pruned={stale_removed}')
        change_snip = f" ({', '.join(parts)})" if parts else ''
        cap = build_combined_caption(self.text_prompts)
        self.node.get_logger().info(
            f'[{self.cam_ns}] Scene prompts synced{change_snip}: '
            f'{len(self._user_prompts)} user + {len(scene_targets)} scene → '
            f'{len(self.text_prompts)} prompts, cumulative='
            f'{scene_dbg.get("cumulative_objects", [])}, '
            f'labels={[t["label"] for t in targets]}, '
            f'gdino={[t["prompt"] for t in targets]}, caption="{cap}"')
        if self.track_manager is not None:
            self.track_manager.sync_targets_preserve_tracks(
                self.active_targets, self.text_prompts, self.prompt_to_label)
        return True

    def _merge_scene_prompts(self, changes: dict | None):
        """Backward-compatible wrapper."""
        self._sync_scene_prompts(changes)

    def _maybe_publish_static_tf(self, cam_frame, stamp):
        if not getattr(self.args, 'publish_static_tf', True):
            return
        if self.T_world_cam is None or not cam_frame:
            return
        if cam_frame in self._static_tf_published:
            return
        if self._static_tf_broadcaster is None:
            self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        world_frame = self.args.workbench_frame_id
        if publish_static_tf_once(
                self._static_tf_broadcaster, self.T_world_cam,
                world_frame, cam_frame, stamp):
            self._static_tf_published.add(cam_frame)
            self.node.get_logger().info(
                f'[{self.cam_ns}] Published static TF: {world_frame} -> {cam_frame}')

    def _log_depth_source_once(self, used_stereo):
        if used_stereo:
            if self._stereo_depth_logged:
                return
            self._stereo_depth_logged = True
            self.node.get_logger().info(
                f'[{self.cam_ns}] Depth source: FoundationStereo (aligned to color)')
            return
        if self._hardware_depth_logged:
            return
        self._hardware_depth_logged = True
        if self.use_stereo_depth:
            self.node.get_logger().warn(
                f'[{self.cam_ns}] Depth source: hardware depth '
                f'(stereo configured but IR topics/TF not ready — check left/right IR + TF)')
        else:
            self.node.get_logger().info(
                f'[{self.cam_ns}] Depth source: hardware depth (use_stereo_depth=false)')

    def _log_depth_quality(self, depth):
        if not hasattr(self, '_depth_zero_streak'):
            self._depth_zero_streak = 0
            self._depth_ok_logged = False
        valid = (depth > self.args.depth_min_m) & (depth < self.args.depth_max_m) & np.isfinite(depth)
        n_valid = int(valid.sum())
        total = depth.shape[0] * depth.shape[1]
        if n_valid < self.args.min_depth_points:
            self._depth_zero_streak += 1
            if self._depth_zero_streak in (1, 5, 20, 50):
                self.node.get_logger().warn(
                    f'[{self.cam_ns}] Depth empty: {n_valid}/{total} valid pixels '
                    f'(range {self.args.depth_min_m}-{self.args.depth_max_m}m), '
                    f'streak={self._depth_zero_streak} frames — '
                    f'3D bbox disabled until depth recovers')
        else:
            if self._depth_zero_streak > 0 and not self._depth_ok_logged:
                self.node.get_logger().info(
                    f'[{self.cam_ns}] Depth recovered: {n_valid}/{total} valid pixels '
                    f'(was empty for {self._depth_zero_streak} frames)')
                self._depth_ok_logged = True
            self._depth_zero_streak = 0

    def _lookup_stereo_tf(self):
        if self.T_ir_to_color is not None:
            return True

        ns = self.cam_ns.strip('/')
        ir_frame = self.args.ir_optical_frame or f'{ns}_left_ir_optical_frame'
        color_frame = self.args.color_optical_frame or f'{ns}_color_optical_frame'
        right_ir_frame = self.args.right_ir_optical_frame or f'{ns}_right_ir_optical_frame'

        # First attempt uses longer timeout to accommodate camera connection_delay
        if not hasattr(self, '_tf_lookup_attempts'):
            self._tf_lookup_attempts = 0
        self._tf_lookup_attempts += 1
        timeout_sec = 10.0 if self._tf_lookup_attempts <= 3 else 2.0

        try:
            tf_ic = self.tf_buffer.lookup_transform(
                color_frame, ir_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec))
            t = tf_ic.transform.translation
            q = tf_ic.transform.rotation
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = [t.x, t.y, t.z]
            self.T_ir_to_color = T

            tf_lr = self.tf_buffer.lookup_transform(
                ir_frame, right_ir_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec))
            t_lr = tf_lr.transform.translation
            self.stereo_baseline = float(np.sqrt(t_lr.x ** 2 + t_lr.y ** 2 + t_lr.z ** 2))
            self.node.get_logger().info(
                f'[{self.cam_ns}] Stereo TF acquired: baseline={self.stereo_baseline:.4f}m '
                f'(frames: {ir_frame} → {color_frame})')
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.node.get_logger().warn(
                f'[{self.cam_ns}] TF lookup failed (attempt {self._tf_lookup_attempts}, '
                f'timeout={timeout_sec}s): {e}',
                throttle_duration_sec=5.0)
            return False

    def _on_stereo_frame(self, left_ir_msg, right_ir_msg, ir_info_msg):
        now = time.time()
        if not hasattr(self, '_last_stereo_time'):
            self._last_stereo_time = 0.0
        if (now - self._last_stereo_time) < self.frame_interval:
            return
        self._last_stereo_time = now

        if self.K_ir is None:
            self.K_ir = np.array(ir_info_msg.k, dtype=np.float64).reshape(3, 3)
        if not self._lookup_stereo_tf() or self.K is None:
            return

        try:
            left_ir = self.bridge.imgmsg_to_cv2(left_ir_msg, desired_encoding='passthrough')
            right_ir = self.bridge.imgmsg_to_cv2(right_ir_msg, desired_encoding='passthrough')
        except Exception as e:
            self.node.get_logger().warn(f'[{self.cam_ns}] Stereo cv_bridge error: {e}')
            return

        if left_ir.ndim == 2:
            left_ir = cv2.cvtColor(left_ir, cv2.COLOR_GRAY2RGB)
        if right_ir.ndim == 2:
            right_ir = cv2.cvtColor(right_ir, cv2.COLOR_GRAY2RGB)

        disp, orig_shape = run_stereo_trt(self.stereo_engine, left_ir, right_ir)
        if self._color_shape is None:
            return

        aligned = stereo_disp_to_aligned_depth(
            disp=disp,
            K_ir_orig=self.K_ir,
            baseline=self.stereo_baseline,
            orig_ir_shape=orig_shape,
            K_color=self.K,
            T_ir_to_color=self.T_ir_to_color,
            color_shape=self._color_shape,
        )
        self.stereo_depth = aligned
        self._stereo_depth_stamp = left_ir_msg.header.stamp

    def _stereo_depth_fresh(self, color_stamp):
        if self.stereo_depth is None or self._stereo_depth_stamp is None:
            return False
        max_age = float(getattr(self.args, 'stereo_max_age_sec', 0.25))
        dt = abs((color_stamp.sec + color_stamp.nanosec * 1e-9)
                 - (self._stereo_depth_stamp.sec + self._stereo_depth_stamp.nanosec * 1e-9))
        return dt <= max_age

    def _prepare_depth(self, depth_raw, h_color, w_color, color_stamp=None):
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) / 1000.0
        else:
            depth = depth_raw.astype(np.float32)

        h_depth, w_depth = depth.shape[:2]
        if (h_depth, w_depth) != (h_color, w_color):
            depth = cv2.resize(depth, (w_color, h_color), interpolation=cv2.INTER_NEAREST)

        used_stereo = False
        if (self.use_stereo_depth and self.stereo_depth is not None
                and (color_stamp is None or self._stereo_depth_fresh(color_stamp))):
            sd = self.stereo_depth
            if sd.shape[:2] == (h_color, w_color):
                depth = sd
            else:
                depth = cv2.resize(sd, (w_color, h_color), interpolation=cv2.INTER_NEAREST)
            used_stereo = True
        return depth, used_stereo

    def _build_exclude_mask(self, masks, shape_hw):
        if not masks or not getattr(self.args, 'workbench_plane_exclude_objects', True):
            return None
        h, w = shape_hw
        excl = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            if mask is None:
                continue
            m = mask.astype(np.uint8)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            excl = np.maximum(excl, (m > 0).astype(np.uint8))
        if not np.any(excl):
            return None
        dilate_px = int(getattr(self.args, 'workbench_plane_exclude_dilate_px', 8))
        if dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            excl = cv2.dilate(excl, k)
        return excl

    def _update_exclude_mask_from_results(self, results, shape_hw):
        masks = [r.get('mask') for r in results if r.get('mask') is not None]
        self._prev_exclude_mask = self._build_exclude_mask(masks, shape_hw)

    def _sync_plane_state_to_args(self):
        self.args.workbench_plane_state = self.plane_state

    def _update_workbench_plane(self, depth, K, stamp, work_frame):
        """Run RANSAC plane fit and publish RViz plane marker + structured topic."""
        if self.plane_estimator is None:
            self.plane_state = None
            return
        exclude = self._prev_exclude_mask
        self.plane_state = self.plane_estimator.update(
            depth, K, self.frame_count, exclude_mask=exclude)
        self._sync_plane_state_to_args()

        if self.workbench_plane_pub is not None:
            plane_msg = WorkbenchPlane()
            plane_msg.header.stamp = stamp
            plane_msg.header.frame_id = work_frame
            if self.plane_state is not None and self.plane_state.valid:
                plane_msg.valid = True
                plane_msg.normal = [float(v) for v in self.plane_state.normal]
                plane_msg.d = float(self.plane_state.d)
                plane_msg.centroid = [float(v) for v in self.plane_state.centroid]
                plane_msg.estimated_z = float(self.plane_state.estimated_z)
                plane_msg.tilt_deg = float(self.plane_state.tilt_deg)
                plane_msg.inlier_count = int(self.plane_state.inlier_count)
            else:
                plane_msg.valid = False
            self.workbench_plane_pub.publish(plane_msg)

        if not self.args.publish_rviz_markers:
            return
        plane_markers = build_workbench_plane_marker_array(
            stamp, work_frame, self.plane_state, self.args)
        self.markers_workbench_plane_pub.publish(plane_markers)

    def _on_rgbd_frame(self, color_msg, depth_msg, info_msg):
        if not self.active_targets and self.scene_manager is None:
            self.node.get_logger().warn(
                f'[{self.cam_ns}] No detection targets active — publish text prompts to '
                f'{DEFAULT_DETECTION_PROMPTS_TOPIC} or set text_prompts at launch',
                throttle_duration_sec=10.0)
            return
        self._on_frame(color_msg, depth_msg, info_msg)

    def _on_frame(self, color_msg, depth_msg, info_msg):
        now = time.time()
        if (now - self.last_process_time) < self.frame_interval:
            return
        self.last_process_time = now
        self.frame_count += 1
        log_timing = bool(getattr(self.args, 'log_frame_timing', True))
        t_frame = time.perf_counter()
        node_timings: dict = {}

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.node.get_logger().warn(f'[{self.cam_ns}] cv_bridge error: {e}')
            return

        target_w = self.args.resize_width
        target_h = self.args.resize_height
        if target_w > 0 and target_h > 0:
            orig_h, orig_w = color.shape[:2]
            if (orig_w, orig_h) != (target_w, target_h):
                color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                depth_raw = cv2.resize(depth_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                sx, sy = target_w / orig_w, target_h / orig_h
            else:
                sx = sy = 1.0
        else:
            sx = sy = 1.0

        h_color, w_color = color.shape[:2]
        depth, used_stereo = self._prepare_depth(
            depth_raw, h_color, w_color, color_stamp=color_msg.header.stamp)
        self._log_depth_source_once(used_stereo)
        self._log_depth_quality(depth)
        self._color_shape = (h_color, w_color)
        self._color_frame_id = color_msg.header.frame_id

        K = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)
        if sx != 1.0 or sy != 1.0:
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        self.K = K

        if self.scene_manager is not None:
            t0 = time.perf_counter()
            async_result = self.scene_manager.check_async_result()
            if async_result:
                self._sync_scene_prompts(async_result)
            stable_masks = self._stable_track_masks()
            if self.scene_manager.should_trigger(color, stable_masks):
                changes = self.scene_manager.run(color)
                if changes.get('added') or changes.get('removed'):
                    self._sync_scene_prompts(changes)
            node_timings['scene_ms'] = (time.perf_counter() - t0) * 1000.0

        caption = build_combined_caption(self.text_prompts)
        if not caption:
            if self.scene_manager is not None and not self.scene_manager.has_run:
                self.node.get_logger().warn(
                    f'[{self.cam_ns}] Waiting for scene understanding (no caption yet)',
                    throttle_duration_sec=5.0)
            elif self.scene_manager is not None:
                self.node.get_logger().warn(
                    f'[{self.cam_ns}] No detection caption — scene returned no objects '
                    f'and no user text_prompts; skipping frame',
                    throttle_duration_sec=5.0)
            else:
                self.node.get_logger().warn(
                    f'[{self.cam_ns}] No detection caption (empty text_prompts)',
                    throttle_duration_sec=5.0)
            return

        stamp = color_msg.header.stamp
        cam_frame = color_msg.header.frame_id
        work_frame = self.args.workbench_frame_id
        self._maybe_publish_static_tf(cam_frame, stamp)
        t0 = time.perf_counter()
        self._update_workbench_plane(depth, K, stamp, work_frame)
        node_timings['plane_ms'] = (time.perf_counter() - t0) * 1000.0

        if self.track_manager is not None:
            self._sync_plane_state_to_args()
            prev_track_ids = {t.track_id for t in self.track_manager.tracks}
            t0 = time.perf_counter()
            results, raw_det_count = self.track_manager.process_frame(
                color, depth, K,
                self.grounding_model, self.sam2_predictor, self.device,
            )
            node_timings['track_ms'] = (time.perf_counter() - t0) * 1000.0
            if self.pixel3d_manager is not None:
                cur_track_ids = {t.track_id for t in self.track_manager.tracks}
                for tid in prev_track_ids - cur_track_ids:
                    self.pixel3d_manager.evict_track(tid)
            if not results:
                self.node.get_logger().info(
                    f'[{self.cam_ns}] No tracked detections (raw_dets={raw_det_count})',
                    throttle_duration_sec=2.0)
            t0 = time.perf_counter()
            self._publish_outputs(
                results, stamp, cam_frame, work_frame, color_msg, color, K, raw_det_count)
            node_timings['publish_ms'] = (time.perf_counter() - t0) * 1000.0
            if self.track_debug_pub is not None:
                dbg_info = self.track_manager.get_debug_info()
                if self.scene_manager is not None:
                    dbg_info['scene'] = self.scene_manager.get_debug_info()
                dbg_info['publish_timing'] = dict(self._last_publish_timing)
                dbg_info['pipeline_timing'] = {
                    k: round(v, 1) for k, v in node_timings.items()
                }
                dbg = String()
                dbg.data = json.dumps(dbg_info)
                self.track_debug_pub.publish(dbg)
            node_timings['total_ms'] = (time.perf_counter() - t_frame) * 1000.0
            self._last_frame_timing = {
                k: round(v, 1) for k, v in node_timings.items()
            }
            if log_timing:
                track_timing = (
                    self.track_manager.get_debug_info()
                    .get('layer_stats', {})
                    .get('timing_ms', {}))
                pub_detail = self._format_publish_timing(self._last_publish_timing)
                pub_suffix = f' [{pub_detail}]' if pub_detail else ''
                self.node.get_logger().info(
                    f'[{self.cam_ns}] Frame {self.frame_count} pipeline timing: '
                    f'scene={node_timings.get("scene_ms", 0):.0f}ms '
                    f'plane={node_timings.get("plane_ms", 0):.0f}ms '
                    f'track={node_timings.get("track_ms", 0):.0f}ms '
                    f'publish={node_timings.get("publish_ms", 0):.0f}ms'
                    f'{pub_suffix} '
                    f'total={node_timings.get("total_ms", 0):.0f}ms '
                    f'(cutie={track_timing.get("cutie_ms", 0):.0f}ms '
                    f'discovery={track_timing.get("discovery_ms", 0):.0f}ms '
                    f'reinit={track_timing.get("reinit_ms", 0):.0f}ms)',
                    throttle_duration_sec=0.5)
            if self.scene_manager is not None:
                n_tracks = len(self.track_manager.tracks) if self.track_manager else 0
                self.scene_manager.record_frame_outcome(
                    [r['tag'] for r in results], n_tracks, raw_det_count)
                self._sync_scene_prompts()
            self._update_exclude_mask_from_results(results, (h_color, w_color))
            return

        if self.vlm_detector is not None and getattr(self.args, 'use_vlm_detect', False):
            detections = vlm_detect_as_gdino(self.vlm_detector, color, caption)
        else:
            detections = gdino_detect(
                self.grounding_model, color, caption,
                self.args.box_threshold, self.args.text_threshold, self.device,
            )
            detections = [
                d for d in detections if d['score'] >= self.args.min_detection_score]

        stamp = color_msg.header.stamp
        cam_frame = color_msg.header.frame_id
        work_frame = self.args.workbench_frame_id

        if not detections:
            det_name = 'VLM' if getattr(self.args, 'use_vlm_detect', False) else 'GDINO'
            self.node.get_logger().info(
                f'[{self.cam_ns}] No {det_name} detections for caption="{caption}"',
                throttle_duration_sec=2.0)
            self._publish_outputs([], stamp, cam_frame, work_frame, color_msg, color, K, 0)
            self._update_exclude_mask_from_results([], (h_color, w_color))
            return

        boxes_xyxy = np.stack([d['box_xyxy'] for d in detections], axis=0)
        masks, sam_scores = sam2_segment_boxes(
            self.sam2_predictor, color, boxes_xyxy, return_scores=True)

        sam_min = getattr(self.args, 'sam2_score_min', 0.7)
        results = []
        instance_counters = {}
        for det, mask, sam_sc in zip(detections, masks, sam_scores):
            if sam_sc < sam_min:
                continue
            label = match_phrase_to_label(
                det['phrase'], self.prompt_to_label, self.active_targets,
                accept_unmatched=self.args.accept_unmatched_detections)
            if label is None:
                self.node.get_logger().warn(
                    f'[{self.cam_ns}] Unmatched phrase "{det["phrase"]}", skipping',
                    throttle_duration_sec=2.0)
                continue

            if int(mask.sum()) < self.args.min_mask_pixels:
                continue

            bbox2d = mask_to_bbox_xyxy(mask.astype(np.uint8))
            if bbox2d is None:
                continue

            pts_cam = backproject_mask(
                depth, K, mask.astype(np.uint8),
                depth_min=self.args.depth_min_m,
                depth_max=self.args.depth_max_m,
            )
            aabb_cam = None
            aabb_work = None
            aabb_work_mesh = None
            n_pts = len(pts_cam)
            if n_pts >= self.args.min_depth_points:
                aabb_cam = aabb_in_frame(pts_cam, T_frame_cam=None)
                if self.T_world_cam is not None:
                    aabb_work_mesh = aabb_in_frame(pts_cam, T_frame_cam=self.T_world_cam)
                    aabb_cam, aabb_work = apply_world_roi_to_aabbs(
                        aabb_cam, aabb_work_mesh, self.T_world_cam, self.args,
                        plane_state=self.plane_state)
                    if aabb_work is None and aabb_work_mesh is not None:
                        z_hint = get_effective_workbench_z(self.args, self.plane_state)
                        plane_tag = (
                            'RANSAC plane'
                            if self.plane_state is not None and self.plane_state.valid
                            else f'workbench z={z_hint:.3f}')
                        self.node.get_logger().info(
                            f'[{self.cam_ns}] [{label}] 3D bbox dropped — outside ROI '
                            f'({roi_mode_label(self.args)}, {plane_tag}); surface kept',
                            throttle_duration_sec=2.0)
            else:
                self.node.get_logger().info(
                    f'[{self.cam_ns}] [{label}] 2D only — too few depth points '
                    f'({n_pts}<{self.args.min_depth_points})',
                    throttle_duration_sec=2.0)

            instance_counters[label] = instance_counters.get(label, 0)
            instance_id = instance_counters[label]
            instance_counters[label] += 1

            orientation = pca_orientation(pts_cam)
            top_normal = compute_top_normal(pts_cam)
            orientation_world = None
            top_normal_world = None
            if self.T_world_cam is not None and n_pts >= 10:
                pts_world = transform_points(self.T_world_cam, pts_cam)
                orientation_world = pca_orientation(pts_world)
                top_normal_world = compute_top_normal(pts_world)
            grasp_type = classify_grasp_type(
                aabb_work['size'] if aabb_work is not None
                else (aabb_cam['size'] if aabb_cam is not None else None))

            idx = int(instance_id) % len(OBJECT_COLORS)
            results.append({
                'tag': label,
                'prompt': det['phrase'],
                'score': det['score'],
                'instance_id': instance_id,
                'track_mode': 'gdino',
                'bbox2d': bbox2d,
                'mask': mask,
                'aabb_cam': aabb_cam,
                'aabb_work': aabb_work,
                'aabb_work_mesh': aabb_work_mesh,
                'pts_cam': pts_cam,
                'orientation': orientation,
                'orientation_world': orientation_world,
                'top_normal': top_normal,
                'top_normal_world': top_normal_world,
                'grasp_type': grasp_type,
                'color': OBJECT_COLORS[idx],
                'color_name': VIZ_COLOR_NAMES[idx % len(VIZ_COLOR_NAMES)],
            })

        self._publish_outputs(
            results, stamp, cam_frame, work_frame, color_msg, color, K, len(detections))
        self._update_exclude_mask_from_results(results, (h_color, w_color))

    def _pixel3d_process(self, r, color, K):
        """Poll/submit Pixel3D mesh for a detection result."""
        if self.pixel3d_manager is None:
            return
        track_id = r.get('instance_id')
        label = r.get('tag', '')
        aabb_work = r.get('aabb_work')
        if track_id is None or aabb_work is None:
            return

        cached = self.pixel3d_manager.try_cached_mesh(track_id, label, aabb_work)
        if cached is not None:
            r['surface_mesh'] = cached
            return

        if self.pixel3d_manager.is_pending(track_id):
            return
        if self.pixel3d_manager.is_completed(track_id):
            return

        trigger_age = getattr(self.args, 'pixel3d_trigger_age', 5)
        track = None
        if self.track_manager:
            for t in self.track_manager.tracks:
                if t.track_id == track_id:
                    track = t
                    break
        if track is None or track.age < trigger_age:
            return
        if track.lost_count > 0:
            return

        from robot_perception.utils.pixel3d_mesh import prepare_crop
        crop_rgba = prepare_crop(color, r['mask'], r['bbox2d'])
        h, w = color.shape[:2]
        self.pixel3d_manager.submit(
            track_id, label, crop_rgba, K, aabb_work, self.T_world_cam, w)

    @staticmethod
    def _mesh_stats(results) -> dict:
        """Aggregate hybrid mesh size for publish timing logs."""
        verts = 0
        faces = 0
        n_mesh = 0
        for r in results:
            mesh = r.get('surface_mesh')
            if mesh is None:
                continue
            n_mesh += 1
            v = mesh.get('vertices')
            f = mesh.get('faces')
            if v is not None:
                verts += len(v)
            if f is not None:
                faces += len(f)
        return {'mesh_count': n_mesh, 'mesh_verts': verts, 'mesh_faces': faces}

    def _format_publish_timing(self, pt: dict) -> str:
        if not pt:
            return ''
        parts = []
        for key, label in (
            ('hybrid_ms', 'hybrid'),
            ('pixel3d_ms', 'pixel3d'),
            ('bbox_build_ms', 'bbox_build'),
            ('markers_build_ms', 'mk_build'),
            ('markers_pub_ms', 'mk_pub'),
            ('bbox_pub_ms', 'bbox_pub'),
            ('surface_mesh_ms', 'surf_mesh'),
            ('annotated_ms', 'annotated'),
            ('pointcloud_ms', 'cloud'),
            ('mask_ms', 'mask'),
            ('frame_log_ms', 'log'),
        ):
            ms = pt.get(key, 0.0)
            if ms > 0.5:
                parts.append(f'{label}={ms:.0f}')
        if pt.get('mesh_count', 0) > 0:
            parts.append(
                f"meshes={pt['mesh_count']}:v{pt.get('mesh_verts', 0)}"
                f":f{pt.get('mesh_faces', 0)}")
        if pt.get('marker_tris', 0) > 0:
            parts.append(f'rviz_tris={pt["marker_tris"]}')
        return ','.join(parts)

    def _publish_outputs(self, results, stamp, cam_frame, work_frame,
                         color_msg, color, K, detections_count):
        log_timing = bool(getattr(self.args, 'log_frame_timing', True))
        pub_ms: dict = {}
        t_pub = time.perf_counter()

        bbox2d_array = LabeledBBox2DArray()
        bbox2d_array.header.stamp = stamp
        bbox2d_array.header.frame_id = cam_frame

        bbox3d_array = LabeledBBox3DArray()
        bbox3d_array.header.stamp = stamp
        bbox3d_array.header.frame_id = (
            work_frame if self.args.publish_workbench_bbox else cam_frame)

        hybrid_ms = 0.0
        pixel3d_ms = 0.0
        bbox_build_ms = 0.0
        for r in results:
            th = time.perf_counter()
            attach_hybrid_surface(r, self.T_world_cam, self.args, plane_state=self.plane_state)
            hybrid_ms += (time.perf_counter() - th) * 1000.0
            tp = time.perf_counter()
            self._pixel3d_process(r, color, K)
            pixel3d_ms += (time.perf_counter() - tp) * 1000.0
            tb = time.perf_counter()
            b2 = LabeledBBox2D()
            b2.label = r['tag']
            b2.prompt = r['prompt']
            b2.score = float(r['score'])
            b2.instance_id = int(r['instance_id'])
            b2.bbox_xyxy = [float(v) for v in r['bbox2d']]
            bbox2d_array.boxes.append(b2)

            orientation = r.get('orientation')
            top_normal = r.get('top_normal')
            grasp_type = r.get('grasp_type', '')
            track_mode = r.get('track_mode', '')

            if self.args.publish_camera_bbox and r.get('aabb_cam') is not None:
                b3 = LabeledBBox3D()
                _fill_bbox3d_msg(
                    b3, r['tag'], r['prompt'], r['score'], r['instance_id'],
                    cam_frame, r['aabb_cam'],
                    orientation=orientation, top_normal=top_normal,
                    grasp_type=grasp_type, track_mode=track_mode)
                bbox3d_array.boxes.append(b3)

            if self.args.publish_workbench_bbox and r.get('aabb_work') is not None:
                b3w = LabeledBBox3D()
                ow = r.get('orientation_world')
                tn = r.get('top_normal_world')
                _fill_bbox3d_msg(
                    b3w, r['tag'], r['prompt'], r['score'], r['instance_id'],
                    work_frame, r['aabb_work'],
                    orientation=ow if ow is not None else orientation,
                    top_normal=tn if tn is not None else top_normal,
                    grasp_type=grasp_type, track_mode=track_mode)
                bbox3d_array.boxes.append(b3w)
            bbox_build_ms += (time.perf_counter() - tb) * 1000.0
        pub_ms['hybrid_ms'] = hybrid_ms
        pub_ms['pixel3d_ms'] = pixel3d_ms
        pub_ms['bbox_build_ms'] = bbox_build_ms
        pub_ms.update(self._mesh_stats(results))

        if self.args.publish_rviz_markers:
            t0 = time.perf_counter()
            cam_markers = build_camera_marker_array(results, stamp, cam_frame, self.args)
            work_markers = build_workbench_marker_array(
                results, stamp, work_frame, self.args)
            surface_markers = build_hybrid_surface_marker_array(
                results, stamp, work_frame, self.args)
            pub_ms['markers_build_ms'] = (time.perf_counter() - t0) * 1000.0
            pub_ms['marker_tris'] = sum(
                len(m.points) // 3
                for m in surface_markers.markers
                if m.ns.startswith('hybrid_surface/'))
            t0 = time.perf_counter()
            self.markers_camera_pub.publish(cam_markers)
            self.markers_workbench_pub.publish(work_markers)
            self.markers_surface_pub.publish(surface_markers)
            pub_ms['markers_pub_ms'] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self.bbox2d_pub.publish(bbox2d_array)
        self.bbox3d_pub.publish(bbox3d_array)
        pub_ms['bbox_pub_ms'] = (time.perf_counter() - t0) * 1000.0

        if self.surface_mesh_pub is not None:
            t0 = time.perf_counter()
            self._publish_surface_meshes(results, stamp, work_frame)
            pub_ms['surface_mesh_ms'] = (time.perf_counter() - t0) * 1000.0

        n3d = len(bbox3d_array.boxes)

        publish_ann = results and (
            self.args.always_publish_annotated
            or self.annotated_pub.get_subscription_count() > 0)
        draw_3d_proj = getattr(self.args, 'annotate_draw_3d_projection', False)
        if publish_ann:
            t0 = time.perf_counter()
            annotated = color.copy()
            for r in results:
                x1, y1, x2, y2 = [int(v) for v in r['bbox2d']]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), r['color'], 2)
                cv2.putText(
                    annotated,
                    f"{r.get('color_name', '?')}:{r['tag']}#{r['instance_id']}:{r['score']:.2f}",
                    (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r['color'], 1)
                overlay = annotated.copy()
                overlay[r['mask']] = r['color']
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
                if (draw_3d_proj and r.get('aabb_work') is not None
                        and r.get('aabb_cam') is not None):
                    draw_aabb_projection(annotated, K, r['aabb_cam'], color=r['color'])

            ann_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            h_ann, w_ann = ann_bgr.shape[:2]
            max_w = 640
            if w_ann > max_w:
                scale = max_w / w_ann
                ann_bgr = cv2.resize(
                    ann_bgr, (max_w, int(h_ann * scale)), interpolation=cv2.INTER_AREA)
            ann_msg = self.bridge.cv2_to_imgmsg(ann_bgr, encoding='bgr8')
            ann_msg.header = color_msg.header
            self.annotated_pub.publish(ann_msg)
            pub_ms['annotated_ms'] = (time.perf_counter() - t0) * 1000.0

        if self.pointcloud_pub is not None and results:
            t0 = time.perf_counter()
            self._publish_object_pointclouds(results, stamp, cam_frame)
            pub_ms['pointcloud_ms'] = (time.perf_counter() - t0) * 1000.0

        if self.mask_pub is not None and results:
            t0 = time.perf_counter()
            self._publish_mask_image(results, stamp, cam_frame, color_msg)
            pub_ms['mask_ms'] = (time.perf_counter() - t0) * 1000.0

        track_info = ''
        warn_mask = ''
        geom_log = ''
        draw_log = ''
        t0 = time.perf_counter()
        if getattr(self.args, 'log_bbox_geometry', True) and results and color is not None:
            img_h, img_w = color.shape[:2]
            draw_log = format_publish_draw_log(results, img_h, img_w)
            if draw_log:
                draw_log = f', annotated_draw=[{draw_log}]'
            gstats = compute_result_geometry_stats(results, img_h, img_w)
            if gstats:
                geom_log = f', 2dbox=[{format_geometry_stats_log(gstats)}]'
                max_mask = max(s['mask_ratio'] for s in gstats) * 100.0
                max_box = max(s['bbox_ratio'] for s in gstats) * 100.0
                min_fill = min(
                    (s['fill'] for s in gstats if s['fill'] > 0), default=100.0)
                box_warn = float(getattr(self.args, 'track_log_bbox_ratio_warn', 12.0))
                if max_box >= box_warn:
                    warn_mask += f' LARGE_2DBOX(max={max_box:.1f}%)'
                if max_box >= 8.0 and max_mask < 6.0:
                    warn_mask += f' SPARSE_BBOX(box={max_box:.1f}% mask={max_mask:.1f}%)'
                if min_fill < float(getattr(self.args, 'track_bbox_min_fill_ratio', 0.10)):
                    warn_mask += f' LOW_FILL(min={min_fill * 100:.0f}%)'
                n_proj3d = sum(
                    1 for r in results
                    if r.get('aabb_work') is not None and r.get('aabb_cam') is not None)
                geom_log += f', 3d_valid={n_proj3d}/{len(results)}'
                proj_parts = []
                for r in results:
                    if r.get('aabb_cam') is None:
                        continue
                    pb = aabb_projected_bbox_xyxy(r['aabb_cam'], K)
                    if pb is None:
                        continue
                    pct = bbox_area_xyxy(pb) / max(1, img_h * img_w) * 100.0
                    roi_ok = r.get('aabb_work') is not None
                    proj_parts.append(
                        f"{r['tag']}#{r['instance_id']}:"
                        f"proj2d={pct:.1f}%:roi={'ok' if roi_ok else 'fail'}")
                if proj_parts:
                    geom_log += f', 3d_bbox=[{", ".join(proj_parts)}]'
                if any(
                        bbox_area_xyxy(pb) / max(1, img_h * img_w) > 0.15
                        for r in results
                        if r.get('aabb_cam') is not None
                        for pb in [aabb_projected_bbox_xyxy(r['aabb_cam'], K)]
                        if pb is not None
                ):
                    warn_mask += ' LARGE_3D_PROJ'
        if self.track_manager is not None:
            labels = [t.label for t in self.track_manager.tracks]
            track_info = f', {len(labels)} tracks: {labels}'
            layer = self.track_manager.get_debug_info().get('layer_stats', {})
            de = layer.get('discovery_eval', {})
            if de:
                track_info += (
                    f", disc={de.get('mode', '?')}/{de.get('interval', 0)}f"
                    f":{'run' if de.get('need') else de.get('reason', 'skip')}")
            elif layer.get('discovery_skipped_stable'):
                track_info += ', discovery_skipped'
            if any(t.mode == 'drift' for t in self.track_manager.tracks):
                warn_mask += ' DRIFT'
            tstats = self.track_manager.get_track_publish_stats()
            if tstats:
                internal_parts = []
                for s in tstats:
                    internal_parts.append(
                        f"#{s['id']}:{s.get('color_name', '?')}:{s['label']}:"
                        f"mask={s['mask_ratio'] * 100:.1f}%:"
                        f"box={s['bbox_ratio'] * 100:.1f}%:"
                        f"mode={s['mode']}")
                track_info += f', track_internal=[{", ".join(internal_parts)}]'
                t_mask = max(s['mask_ratio'] for s in tstats) * 100.0
                t_box = max(s['bbox_ratio'] for s in tstats) * 100.0
                if results and color is not None and getattr(
                        self.args, 'log_bbox_geometry', True):
                    img_area = max(1, int(color.shape[0]) * int(color.shape[1]))
                    pub_box = max(
                        bbox_area_xyxy(r['bbox2d']) / img_area
                        for r in results if r.get('bbox2d') is not None)
                    if t_box > pub_box * 2.0 and t_box >= 8.0:
                        warn_mask += (
                            f' TRACK_PUB_MISMATCH(internal_box={t_box:.1f}%'
                            f' pub_box={pub_box * 100:.1f}%)')
        caption_snip = ''
        if self.text_prompts:
            cap = build_combined_caption(self.text_prompts)
            if len(cap) > 48:
                cap = cap[:45] + '...'
            caption_snip = f', caption="{cap}"'
        prompt_debug_snip = self._format_prompt_debug() if self.scene_manager is not None else ''
        result_labels = [r['tag'] for r in results]
        pub_ms['frame_log_ms'] = (time.perf_counter() - t0) * 1000.0
        pub_ms['total_ms'] = (time.perf_counter() - t_pub) * 1000.0
        self._last_publish_timing = {k: round(v, 1) for k, v in pub_ms.items()}

        timing_snip = ''
        if log_timing and self._last_publish_timing:
            pt = self._last_publish_timing
            detail = self._format_publish_timing(pt)
            timing_snip = (
                f', pub={pt.get("total_ms", 0):.0f}ms'
                f'[{detail}]' if detail else f', pub={pt.get("total_ms", 0):.0f}ms'
            )
        self.node.get_logger().info(
            f'[{self.cam_ns}] Frame {self.frame_count}: '
            f'{detections_count} det, {len(results)} result(s) {result_labels}, '
            f'2d={len(bbox2d_array.boxes)}, 3d={n3d} '
            f'(frame={work_frame if self.args.publish_workbench_bbox else cam_frame})'
            + caption_snip
            + prompt_debug_snip
            + track_info
            + draw_log
            + geom_log
            + timing_snip
            + warn_mask)
        if log_timing and self._last_publish_timing:
            self.node.get_logger().info(
                f'[{self.cam_ns}] Frame {self.frame_count} publish timing: '
                f'{self._format_publish_timing(self._last_publish_timing)} '
                f'total={self._last_publish_timing.get("total_ms", 0):.0f}ms',
                throttle_duration_sec=0.5)

    def _publish_surface_meshes(self, results, stamp, frame_id):
        """Publish surface mesh data as SurfaceMeshArray topic."""
        msg = SurfaceMeshArray()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for r in results:
            mesh = r.get('surface_mesh')
            if mesh is None:
                continue
            sm = SurfaceMesh()
            sm.label = r['tag']
            sm.instance_id = int(r['instance_id'])
            sm.frame_id = frame_id
            verts = np.asarray(mesh['vertices'], dtype=np.float32).flatten()
            faces = np.asarray(mesh['faces'], dtype=np.uint32).flatten()
            sm.num_vertices = len(mesh['vertices'])
            sm.num_faces = len(mesh['faces'])
            sm.vertices = verts.tolist()
            sm.faces = faces.tolist()
            msg.meshes.append(sm)
        self.surface_mesh_pub.publish(msg)

    def _publish_object_pointclouds(self, results, stamp, frame_id):
        """Publish per-object segmented point clouds as a single PointCloud2 with instance field."""
        all_pts = []
        all_ids = []
        for r in results:
            pts = r.get('pts_cam')
            if pts is None or len(pts) == 0:
                continue
            all_pts.append(pts)
            all_ids.append(np.full(len(pts), r['instance_id'], dtype=np.uint32))
        if not all_pts:
            return

        pts_arr = np.vstack(all_pts).astype(np.float32)
        ids_arr = np.concatenate(all_ids).astype(np.uint32)
        n = len(pts_arr)

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = n
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='instance_id', offset=12, datatype=PointField.UINT32, count=1),
        ]
        buf = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('id', '<u4')])
        buf['x'] = pts_arr[:, 0]
        buf['y'] = pts_arr[:, 1]
        buf['z'] = pts_arr[:, 2]
        buf['id'] = ids_arr
        msg.data = buf.tobytes()
        self.pointcloud_pub.publish(msg)

    def _publish_mask_image(self, results, stamp, frame_id, color_msg):
        """Publish instance-indexed mask image (uint16: 0=bg, N=instance_id+1)."""
        h, w = self._color_shape or (480, 640)
        mask_img = np.zeros((h, w), dtype=np.uint16)
        for r in results:
            m = r.get('mask')
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            mask_img[m > 0] = int(r['instance_id']) + 1
        mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding='mono16')
        mask_msg.header.stamp = stamp
        mask_msg.header.frame_id = frame_id
        self.mask_pub.publish(mask_msg)


class DetectionBBoxNode(Node):
    def __init__(self):
        super().__init__('detection_bbox')

        self.declare_parameter('cam_ns', [DEFAULT_CAMERA_NS])
        self.declare_parameter('tags', [])
        self.declare_parameter('text_prompts', [])
        self.declare_parameter('auto_open_scene_prompts', False)
        self.declare_parameter('auto_industry_scene_prompts', True)
        self.declare_parameter('auto_lego_scene_prompts', False)
        self.declare_parameter('allow_freeform_prompts', True)
        self.declare_parameter('accept_unmatched_detections', True)
        self.declare_parameter('always_publish_annotated', True)
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('min_detection_score', 0.45)
        self.declare_parameter('use_temporal_tracking', True)
        self.declare_parameter('use_cutie_tracking', True)
        self.declare_parameter('cutie_seg_threshold', 0.1)
        self.declare_parameter('track_mask_iou_min', 0.5)
        self.declare_parameter('track_area_ratio_min', 0.3)
        self.declare_parameter('track_area_ratio_max', 3.0)
        self.declare_parameter('track_depth_std_min', 0.003)
        self.declare_parameter('track_global_detect_interval', 10)
        self.declare_parameter('track_stable_discovery_interval', 15)
        self.declare_parameter('track_empty_discovery_interval', 5)
        self.declare_parameter('track_skip_discovery_when_stable', True)
        self.declare_parameter('track_stable_min_age', 3)
        self.declare_parameter('track_discovery_skip_reinit_iou', 0.65)
        self.declare_parameter('track_discovery_update_iou_min', 0.2)
        self.declare_parameter('track_discovery_batch_size', 2)
        self.declare_parameter('track_discovery_max_batches_per_frame', 0)
        self.declare_parameter('track_discovery_use_batches', True)
        self.declare_parameter('track_discovery_gap_fill', True)
        self.declare_parameter('track_discovery_gap_fill_only', True)
        self.declare_parameter('track_gap_fill_discovery_interval', 5)
        self.declare_parameter('scene_preserve_gdino_prompts', True)
        self.declare_parameter('scene_prompt_never_detected_grace_sec', 120.0)
        self.declare_parameter('track_spatial_blacklist_frames', 100)
        self.declare_parameter('track_spatial_blacklist_iou', 0.3)
        self.declare_parameter('track_anchor_area_ratio_max', 2.5)
        self.declare_parameter('track_table_leak_max', 0.72)
        self.declare_parameter('track_table_leak_min_mask_ratio', 0.01)
        self.declare_parameter('track_log_mask_ratio_warn', 0.12)
        self.declare_parameter('track_log_bbox_ratio_warn', 12.0)
        self.declare_parameter('log_bbox_geometry', True)
        self.declare_parameter('annotate_draw_3d_projection', False)
        self.declare_parameter('track_bbox_min_fill_ratio', 0.10)
        self.declare_parameter('track_bbox_area_jump_max', 2.5)
        self.declare_parameter('track_bbox_ema', 0.8)
        self.declare_parameter('track_mask_open_kernel', 3)
        self.declare_parameter('track_lost_max_frames', 5)
        self.declare_parameter('track_roi_fail_max_frames', 15)
        self.declare_parameter('track_roi_blacklist_frames', 100)
        self.declare_parameter('track_max_mask_ratio', 0.25)
        self.declare_parameter('track_assoc_iou_min', 0.3)
        self.declare_parameter('track_assoc_require_label_match', True)
        self.declare_parameter('track_label_lock', True)
        self.declare_parameter('track_score_ema', 0.7)
        self.declare_parameter('use_layer2_roi_refine', True)
        self.declare_parameter('track_roi_expand_ratio', 2.0)
        self.declare_parameter('track_refine_box_threshold', 0.25)
        self.declare_parameter('track_refine_score_min', 0.35)
        self.declare_parameter('track_refine_iou_min', 0.5)
        self.declare_parameter('track_refine_iou_hard_min', 0.25)
        self.declare_parameter('sam2_score_min', 0.7)
        self.declare_parameter('vlm_nms_iou', 0.5)
        self.declare_parameter('vlm_max_area_ratio', 0.25)
        self.declare_parameter('track_label_vote_window', 5)
        self.declare_parameter('publish_track_debug', False)
        self.declare_parameter('slop', 0.05)
        self.declare_parameter('device', '')
        self.declare_parameter('sam2_checkpoint', SAM2_CHECKPOINT)
        self.declare_parameter('sam2_model_config', 'configs/sam2.1/sam2.1_hiera_t.yaml')
        self.declare_parameter('gdino_config', GDINO_CONFIG)
        self.declare_parameter('gdino_checkpoint', GDINO_CHECKPOINT)
        self.declare_parameter('use_stereo_depth', False)
        self.declare_parameter('stereo_plan_path', STEREO_PLAN_PATH)
        self.declare_parameter('stereo_baseline', 0.095)
        self.declare_parameter('stereo_max_age_sec', 0.25)
        self.declare_parameter('ir_optical_frame', '')
        self.declare_parameter('color_optical_frame', '')
        self.declare_parameter('right_ir_optical_frame', '')
        self.declare_parameter('calib_file', '')
        self.declare_parameter('workbench_frame_id', 'world')
        self.declare_parameter('publish_static_tf', True)
        self.declare_parameter('enable_world_roi_filter', True)
        self.declare_parameter('world_roi_mode', 'and')
        self.declare_parameter('world_forward_max_m', 1.0)
        self.declare_parameter('workbench_z', -1.0)
        self.declare_parameter('workbench_surface_tol_m', 0.04)
        self.declare_parameter('workbench_max_height_m', 0.35)
        self.declare_parameter('enable_ransac_workbench_plane', True)
        self.declare_parameter('workbench_z_prior_tol_m', 0.15)
        self.declare_parameter('workbench_plane_ransac_iters', 200)
        self.declare_parameter('workbench_plane_inlier_thresh_m', 0.015)
        self.declare_parameter('workbench_plane_min_inliers', 500)
        self.declare_parameter('workbench_plane_sample_stride', 4)
        self.declare_parameter('workbench_plane_roi_v_frac', 0.55)
        self.declare_parameter('workbench_plane_normal_max_tilt_deg', 12.0)
        self.declare_parameter('workbench_plane_ema', 0.85)
        self.declare_parameter('workbench_plane_update_interval', 3)
        self.declare_parameter('workbench_plane_exclude_objects', True)
        self.declare_parameter('workbench_plane_exclude_dilate_px', 8)
        self.declare_parameter('publish_workbench_plane_marker', True)
        self.declare_parameter('workbench_plane_marker_size_m', 0.8)
        self.declare_parameter('workbench_plane_marker_thickness_m', 0.004)
        self.declare_parameter('enable_hybrid_surface_mesh', True)
        self.declare_parameter('publish_hybrid_surface_marker', True)
        self.declare_parameter('hybrid_surface_min_points', 0)
        self.declare_parameter('hybrid_surface_top_frac', 0.45)
        self.declare_parameter('hybrid_surface_max_top_points', 600)
        self.declare_parameter('hybrid_surface_max_triangle_edge_m', 0.04)
        self.declare_parameter('hybrid_surface_min_triangle_normal_z', 0.15)
        self.declare_parameter('hybrid_surface_marker_alpha', 0.55)
        self.declare_parameter('hybrid_surface_rim_search_m', 0.012)
        self.declare_parameter('hybrid_surface_aabb_bottom', False)
        self.declare_parameter('hybrid_surface_outlier_filter', True)
        self.declare_parameter('hybrid_surface_iqr_k', 1.5)
        self.declare_parameter('hybrid_surface_sor_k', 8)
        self.declare_parameter('hybrid_surface_sor_std', 1.5)
        self.declare_parameter('publish_camera_bbox', False)
        self.declare_parameter('publish_workbench_bbox', True)
        self.declare_parameter('resize_width', 0)
        self.declare_parameter('resize_height', 0)
        self.declare_parameter('min_mask_pixels', 100)
        self.declare_parameter('min_depth_points', 50)
        self.declare_parameter('depth_min_m', 0.01)
        self.declare_parameter('depth_max_m', 1.5)
        self.declare_parameter('subscribe_detection_prompts', True)
        self.declare_parameter('detection_prompts_topic', DEFAULT_DETECTION_PROMPTS_TOPIC)
        self.declare_parameter('publish_surface_mesh_topic', True)
        self.declare_parameter('publish_rviz_markers', True)
        self.declare_parameter('rviz_show_fill', False)
        self.declare_parameter('rviz_fill_alpha', 0.15)
        self.declare_parameter('use_vlm_detect', False)
        self.declare_parameter('vlm_base_url', 'http://127.0.0.1:8000/v1')
        self.declare_parameter('vlm_api_key', 'local')
        self.declare_parameter('vlm_model', default_vlm_serve_model_id())
        self.declare_parameter('vlm_debug_dir', '')
        self.declare_parameter('use_scene_understand', False)
        self.declare_parameter('scene_prompt_config', '')
        self.declare_parameter('scene_change_check_interval', 10)
        self.declare_parameter('scene_change_threshold', 0.15)
        self.declare_parameter('scene_change_pixel_threshold', 30)
        self.declare_parameter('scene_max_objects', 10)
        self.declare_parameter('scene_understand_cooldown', 10.0)
        self.declare_parameter('scene_use_prior_hints', True)
        self.declare_parameter('log_scene_prompts', True)
        self.declare_parameter('scene_first_run_retry_sec', 2.0)
        self.declare_parameter('scene_prompt_stale_sec', 45.0)
        self.declare_parameter('scene_force_refresh_empty_frames', 20)
        self.declare_parameter('log_frame_timing', True)
        self.declare_parameter('publish_object_pointcloud', False)
        self.declare_parameter('publish_mask_image', False)
        self.declare_parameter('use_pixel3d', False)
        self.declare_parameter('pixel3d_model_path', PIXEL3D_MODEL_PATH)
        self.declare_parameter('pixel3d_low_vram', True)
        self.declare_parameter('pixel3d_trigger_age', 5)
        self.declare_parameter('pixel3d_max_concurrent', 1)
        self.declare_parameter('pixel3d_cache_dir', '')
        self.declare_parameter('pixel3d_use_label_cache', True)
        self.declare_parameter('pixel3d_offline_config', '')

        self.handlers = []
        self._pending_detection_prompts = None
        self._subscribe_detection_prompts = (
            self.get_parameter('subscribe_detection_prompts').get_parameter_value().bool_value)
        if self._subscribe_detection_prompts:
            prompts_topic = (
                self.get_parameter('detection_prompts_topic')
                .get_parameter_value().string_value)
            self.create_subscription(
                StringVec, prompts_topic, self._on_detection_prompts, 10)
            self.get_logger().info(
                f'[detection_bbox] Listening on {prompts_topic} (buffering during init)')

        self.args = type('Args', (), {
            'cam_ns': _get_string_list_param(self, 'cam_ns'),
            'allow_freeform_prompts': self.get_parameter('allow_freeform_prompts').get_parameter_value().bool_value,
            'auto_open_scene_prompts': self.get_parameter(
                'auto_open_scene_prompts').get_parameter_value().bool_value,
            'auto_industry_scene_prompts': self.get_parameter(
                'auto_industry_scene_prompts').get_parameter_value().bool_value,
            'auto_lego_scene_prompts': self.get_parameter(
                'auto_lego_scene_prompts').get_parameter_value().bool_value,
            'accept_unmatched_detections': self.get_parameter('accept_unmatched_detections').get_parameter_value().bool_value,
            'always_publish_annotated': self.get_parameter('always_publish_annotated').get_parameter_value().bool_value,
            'fps': self.get_parameter('fps').get_parameter_value().double_value,
            'box_threshold': self.get_parameter('box_threshold').get_parameter_value().double_value,
            'text_threshold': self.get_parameter('text_threshold').get_parameter_value().double_value,
            'min_detection_score': self.get_parameter('min_detection_score').get_parameter_value().double_value,
            'use_temporal_tracking': self.get_parameter(
                'use_temporal_tracking').get_parameter_value().bool_value,
            'use_cutie_tracking': self.get_parameter('use_cutie_tracking').get_parameter_value().bool_value,
            'cutie_seg_threshold': self.get_parameter(
                'cutie_seg_threshold').get_parameter_value().double_value,
            'track_mask_iou_min': self.get_parameter(
                'track_mask_iou_min').get_parameter_value().double_value,
            'track_area_ratio_min': self.get_parameter(
                'track_area_ratio_min').get_parameter_value().double_value,
            'track_area_ratio_max': self.get_parameter(
                'track_area_ratio_max').get_parameter_value().double_value,
            'track_depth_std_min': self.get_parameter(
                'track_depth_std_min').get_parameter_value().double_value,
            'track_global_detect_interval': self.get_parameter(
                'track_global_detect_interval').get_parameter_value().integer_value,
            'track_stable_discovery_interval': self.get_parameter(
                'track_stable_discovery_interval').get_parameter_value().integer_value,
            'track_empty_discovery_interval': self.get_parameter(
                'track_empty_discovery_interval').get_parameter_value().integer_value,
            'track_skip_discovery_when_stable': self.get_parameter(
                'track_skip_discovery_when_stable').get_parameter_value().bool_value,
            'track_stable_min_age': self.get_parameter(
                'track_stable_min_age').get_parameter_value().integer_value,
            'track_discovery_skip_reinit_iou': self.get_parameter(
                'track_discovery_skip_reinit_iou').get_parameter_value().double_value,
            'track_discovery_update_iou_min': self.get_parameter(
                'track_discovery_update_iou_min').get_parameter_value().double_value,
            'track_discovery_batch_size': self.get_parameter(
                'track_discovery_batch_size').get_parameter_value().integer_value,
            'track_discovery_max_batches_per_frame': self.get_parameter(
                'track_discovery_max_batches_per_frame').get_parameter_value().integer_value,
            'track_discovery_use_batches': self.get_parameter(
                'track_discovery_use_batches').get_parameter_value().bool_value,
            'track_discovery_gap_fill': self.get_parameter(
                'track_discovery_gap_fill').get_parameter_value().bool_value,
            'track_discovery_gap_fill_only': self.get_parameter(
                'track_discovery_gap_fill_only').get_parameter_value().bool_value,
            'track_gap_fill_discovery_interval': self.get_parameter(
                'track_gap_fill_discovery_interval').get_parameter_value().integer_value,
            'scene_preserve_gdino_prompts': self.get_parameter(
                'scene_preserve_gdino_prompts').get_parameter_value().bool_value,
            'scene_prompt_never_detected_grace_sec': self.get_parameter(
                'scene_prompt_never_detected_grace_sec').get_parameter_value().double_value,
            'track_spatial_blacklist_frames': self.get_parameter(
                'track_spatial_blacklist_frames').get_parameter_value().integer_value,
            'track_spatial_blacklist_iou': self.get_parameter(
                'track_spatial_blacklist_iou').get_parameter_value().double_value,
            'track_anchor_area_ratio_max': self.get_parameter(
                'track_anchor_area_ratio_max').get_parameter_value().double_value,
            'track_table_leak_max': self.get_parameter(
                'track_table_leak_max').get_parameter_value().double_value,
            'track_table_leak_min_mask_ratio': self.get_parameter(
                'track_table_leak_min_mask_ratio').get_parameter_value().double_value,
            'track_log_mask_ratio_warn': self.get_parameter(
                'track_log_mask_ratio_warn').get_parameter_value().double_value,
            'track_log_bbox_ratio_warn': self.get_parameter(
                'track_log_bbox_ratio_warn').get_parameter_value().double_value,
            'log_bbox_geometry': self.get_parameter(
                'log_bbox_geometry').get_parameter_value().bool_value,
            'annotate_draw_3d_projection': self.get_parameter(
                'annotate_draw_3d_projection').get_parameter_value().bool_value,
            'track_bbox_min_fill_ratio': self.get_parameter(
                'track_bbox_min_fill_ratio').get_parameter_value().double_value,
            'track_bbox_area_jump_max': self.get_parameter(
                'track_bbox_area_jump_max').get_parameter_value().double_value,
            'track_bbox_ema': self.get_parameter(
                'track_bbox_ema').get_parameter_value().double_value,
            'track_mask_open_kernel': self.get_parameter(
                'track_mask_open_kernel').get_parameter_value().integer_value,
            'track_lost_max_frames': self.get_parameter(
                'track_lost_max_frames').get_parameter_value().integer_value,
            'track_roi_fail_max_frames': self.get_parameter(
                'track_roi_fail_max_frames').get_parameter_value().integer_value,
            'track_roi_blacklist_frames': self.get_parameter(
                'track_roi_blacklist_frames').get_parameter_value().integer_value,
            'track_max_mask_ratio': self.get_parameter(
                'track_max_mask_ratio').get_parameter_value().double_value,
            'track_assoc_iou_min': self.get_parameter(
                'track_assoc_iou_min').get_parameter_value().double_value,
            'track_assoc_require_label_match': self.get_parameter(
                'track_assoc_require_label_match').get_parameter_value().bool_value,
            'track_label_lock': self.get_parameter(
                'track_label_lock').get_parameter_value().bool_value,
            'track_score_ema': self.get_parameter(
                'track_score_ema').get_parameter_value().double_value,
            'use_layer2_roi_refine': self.get_parameter(
                'use_layer2_roi_refine').get_parameter_value().bool_value,
            'track_roi_expand_ratio': self.get_parameter(
                'track_roi_expand_ratio').get_parameter_value().double_value,
            'track_refine_box_threshold': self.get_parameter(
                'track_refine_box_threshold').get_parameter_value().double_value,
            'track_refine_score_min': self.get_parameter(
                'track_refine_score_min').get_parameter_value().double_value,
            'track_refine_iou_min': self.get_parameter(
                'track_refine_iou_min').get_parameter_value().double_value,
            'track_refine_iou_hard_min': self.get_parameter(
                'track_refine_iou_hard_min').get_parameter_value().double_value,
            'sam2_score_min': self.get_parameter(
                'sam2_score_min').get_parameter_value().double_value,
            'vlm_nms_iou': self.get_parameter(
                'vlm_nms_iou').get_parameter_value().double_value,
            'vlm_max_area_ratio': self.get_parameter(
                'vlm_max_area_ratio').get_parameter_value().double_value,
            'track_label_vote_window': self.get_parameter(
                'track_label_vote_window').get_parameter_value().integer_value,
            'publish_track_debug': self.get_parameter(
                'publish_track_debug').get_parameter_value().bool_value,
            'slop': self.get_parameter('slop').get_parameter_value().double_value,
            'device': self.get_parameter('device').get_parameter_value().string_value or None,
            'sam2_checkpoint': self.get_parameter('sam2_checkpoint').get_parameter_value().string_value,
            'sam2_model_config': self.get_parameter('sam2_model_config').get_parameter_value().string_value,
            'gdino_config': self.get_parameter('gdino_config').get_parameter_value().string_value,
            'gdino_checkpoint': self.get_parameter('gdino_checkpoint').get_parameter_value().string_value,
            'use_stereo_depth': self.get_parameter('use_stereo_depth').get_parameter_value().bool_value,
            'stereo_plan_path': self.get_parameter('stereo_plan_path').get_parameter_value().string_value,
            'stereo_baseline': self.get_parameter('stereo_baseline').get_parameter_value().double_value,
            'stereo_max_age_sec': self.get_parameter(
                'stereo_max_age_sec').get_parameter_value().double_value,
            'ir_optical_frame': self.get_parameter('ir_optical_frame').get_parameter_value().string_value or None,
            'color_optical_frame': self.get_parameter('color_optical_frame').get_parameter_value().string_value or None,
            'right_ir_optical_frame': self.get_parameter('right_ir_optical_frame').get_parameter_value().string_value or None,
            'workbench_frame_id': self.get_parameter('workbench_frame_id').get_parameter_value().string_value,
            'publish_static_tf': self.get_parameter(
                'publish_static_tf').get_parameter_value().bool_value,
            'enable_world_roi_filter': self.get_parameter(
                'enable_world_roi_filter').get_parameter_value().bool_value,
            'world_roi_mode': self.get_parameter('world_roi_mode').get_parameter_value().string_value,
            'world_forward_max_m': self.get_parameter(
                'world_forward_max_m').get_parameter_value().double_value,
            'workbench_z': self.get_parameter('workbench_z').get_parameter_value().double_value,
            'workbench_surface_tol_m': self.get_parameter(
                'workbench_surface_tol_m').get_parameter_value().double_value,
            'workbench_max_height_m': self.get_parameter(
                'workbench_max_height_m').get_parameter_value().double_value,
            'enable_ransac_workbench_plane': self.get_parameter(
                'enable_ransac_workbench_plane').get_parameter_value().bool_value,
            'workbench_z_prior_tol_m': self.get_parameter(
                'workbench_z_prior_tol_m').get_parameter_value().double_value,
            'workbench_plane_ransac_iters': self.get_parameter(
                'workbench_plane_ransac_iters').get_parameter_value().integer_value,
            'workbench_plane_inlier_thresh_m': self.get_parameter(
                'workbench_plane_inlier_thresh_m').get_parameter_value().double_value,
            'workbench_plane_min_inliers': self.get_parameter(
                'workbench_plane_min_inliers').get_parameter_value().integer_value,
            'workbench_plane_sample_stride': self.get_parameter(
                'workbench_plane_sample_stride').get_parameter_value().integer_value,
            'workbench_plane_roi_v_frac': self.get_parameter(
                'workbench_plane_roi_v_frac').get_parameter_value().double_value,
            'workbench_plane_normal_max_tilt_deg': self.get_parameter(
                'workbench_plane_normal_max_tilt_deg').get_parameter_value().double_value,
            'workbench_plane_ema': self.get_parameter(
                'workbench_plane_ema').get_parameter_value().double_value,
            'workbench_plane_update_interval': self.get_parameter(
                'workbench_plane_update_interval').get_parameter_value().integer_value,
            'workbench_plane_exclude_objects': self.get_parameter(
                'workbench_plane_exclude_objects').get_parameter_value().bool_value,
            'workbench_plane_exclude_dilate_px': self.get_parameter(
                'workbench_plane_exclude_dilate_px').get_parameter_value().integer_value,
            'publish_workbench_plane_marker': self.get_parameter(
                'publish_workbench_plane_marker').get_parameter_value().bool_value,
            'workbench_plane_marker_size_m': self.get_parameter(
                'workbench_plane_marker_size_m').get_parameter_value().double_value,
            'workbench_plane_marker_thickness_m': self.get_parameter(
                'workbench_plane_marker_thickness_m').get_parameter_value().double_value,
            'enable_hybrid_surface_mesh': self.get_parameter(
                'enable_hybrid_surface_mesh').get_parameter_value().bool_value,
            'publish_hybrid_surface_marker': self.get_parameter(
                'publish_hybrid_surface_marker').get_parameter_value().bool_value,
            'hybrid_surface_min_points': self.get_parameter(
                'hybrid_surface_min_points').get_parameter_value().integer_value,
            'hybrid_surface_top_frac': self.get_parameter(
                'hybrid_surface_top_frac').get_parameter_value().double_value,
            'hybrid_surface_max_top_points': self.get_parameter(
                'hybrid_surface_max_top_points').get_parameter_value().integer_value,
            'hybrid_surface_max_triangle_edge_m': self.get_parameter(
                'hybrid_surface_max_triangle_edge_m').get_parameter_value().double_value,
            'hybrid_surface_min_triangle_normal_z': self.get_parameter(
                'hybrid_surface_min_triangle_normal_z').get_parameter_value().double_value,
            'hybrid_surface_marker_alpha': self.get_parameter(
                'hybrid_surface_marker_alpha').get_parameter_value().double_value,
            'hybrid_surface_rim_search_m': self.get_parameter(
                'hybrid_surface_rim_search_m').get_parameter_value().double_value,
            'hybrid_surface_aabb_bottom': self.get_parameter(
                'hybrid_surface_aabb_bottom').get_parameter_value().bool_value,
            'hybrid_surface_outlier_filter': self.get_parameter(
                'hybrid_surface_outlier_filter').get_parameter_value().bool_value,
            'hybrid_surface_iqr_k': self.get_parameter(
                'hybrid_surface_iqr_k').get_parameter_value().double_value,
            'hybrid_surface_sor_k': self.get_parameter(
                'hybrid_surface_sor_k').get_parameter_value().integer_value,
            'hybrid_surface_sor_std': self.get_parameter(
                'hybrid_surface_sor_std').get_parameter_value().double_value,
            'workbench_plane_state': None,
            'publish_camera_bbox': self.get_parameter('publish_camera_bbox').get_parameter_value().bool_value,
            'publish_workbench_bbox': self.get_parameter('publish_workbench_bbox').get_parameter_value().bool_value,
            'resize_width': self.get_parameter('resize_width').get_parameter_value().integer_value,
            'resize_height': self.get_parameter('resize_height').get_parameter_value().integer_value,
            'min_mask_pixels': self.get_parameter('min_mask_pixels').get_parameter_value().integer_value,
            'min_depth_points': self.get_parameter('min_depth_points').get_parameter_value().integer_value,
            'depth_min_m': self.get_parameter('depth_min_m').get_parameter_value().double_value,
            'depth_max_m': self.get_parameter('depth_max_m').get_parameter_value().double_value,
            'publish_surface_mesh_topic': self.get_parameter('publish_surface_mesh_topic').get_parameter_value().bool_value,
            'publish_rviz_markers': self.get_parameter('publish_rviz_markers').get_parameter_value().bool_value,
            'rviz_show_fill': self.get_parameter('rviz_show_fill').get_parameter_value().bool_value,
            'rviz_fill_alpha': self.get_parameter('rviz_fill_alpha').get_parameter_value().double_value,
            'use_vlm_detect': self.get_parameter('use_vlm_detect').get_parameter_value().bool_value,
            'vlm_base_url': self.get_parameter('vlm_base_url').get_parameter_value().string_value,
            'vlm_api_key': self.get_parameter('vlm_api_key').get_parameter_value().string_value,
            'vlm_model': self.get_parameter('vlm_model').get_parameter_value().string_value,
            'vlm_debug_dir': self.get_parameter('vlm_debug_dir').get_parameter_value().string_value,
            'use_scene_understand': self.get_parameter('use_scene_understand').get_parameter_value().bool_value,
            'scene_change_check_interval': self.get_parameter(
                'scene_change_check_interval').get_parameter_value().integer_value,
            'scene_change_threshold': self.get_parameter(
                'scene_change_threshold').get_parameter_value().double_value,
            'scene_change_pixel_threshold': self.get_parameter(
                'scene_change_pixel_threshold').get_parameter_value().integer_value,
            'scene_max_objects': self.get_parameter(
                'scene_max_objects').get_parameter_value().integer_value,
            'scene_understand_cooldown': self.get_parameter(
                'scene_understand_cooldown').get_parameter_value().double_value,
            'scene_use_prior_hints': self.get_parameter(
                'scene_use_prior_hints').get_parameter_value().bool_value,
            'log_scene_prompts': self.get_parameter(
                'log_scene_prompts').get_parameter_value().bool_value,
            'scene_first_run_retry_sec': self.get_parameter(
                'scene_first_run_retry_sec').get_parameter_value().double_value,
            'scene_prompt_stale_sec': self.get_parameter(
                'scene_prompt_stale_sec').get_parameter_value().double_value,
            'scene_force_refresh_empty_frames': self.get_parameter(
                'scene_force_refresh_empty_frames').get_parameter_value().integer_value,
            'log_frame_timing': self.get_parameter(
                'log_frame_timing').get_parameter_value().bool_value,
            'publish_object_pointcloud': self.get_parameter('publish_object_pointcloud').get_parameter_value().bool_value,
            'publish_mask_image': self.get_parameter('publish_mask_image').get_parameter_value().bool_value,
            'use_pixel3d': self.get_parameter('use_pixel3d').get_parameter_value().bool_value,
            'pixel3d_model_path': self.get_parameter('pixel3d_model_path').get_parameter_value().string_value,
            'pixel3d_low_vram': self.get_parameter('pixel3d_low_vram').get_parameter_value().bool_value,
            'pixel3d_trigger_age': self.get_parameter('pixel3d_trigger_age').get_parameter_value().integer_value,
            'pixel3d_max_concurrent': self.get_parameter('pixel3d_max_concurrent').get_parameter_value().integer_value,
            'pixel3d_cache_dir': self.get_parameter('pixel3d_cache_dir').get_parameter_value().string_value,
            'pixel3d_use_label_cache': self.get_parameter('pixel3d_use_label_cache').get_parameter_value().bool_value,
            'pixel3d_offline_config': self.get_parameter('pixel3d_offline_config').get_parameter_value().string_value,
        })()

        device = self.args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._grounding_model = None
        self._vlm_detector = None
        self._scene_prompt_config = None
        if self.args.use_scene_understand or self.args.use_vlm_detect:
            self._scene_prompt_config = load_scene_prompt_config(
                self.get_parameter('scene_prompt_config').get_parameter_value().string_value,
                logger=self.get_logger(),
            )

        if self.args.use_vlm_detect:
            if self.args.use_layer2_roi_refine:
                self.get_logger().warn(
                    '[detection_bbox] use_vlm_detect=true: Layer2 ROI refine still uses '
                    'VLM on crop (GDINO not loaded)')
            self.get_logger().info('[detection_bbox] Loading SAM2 (VLM detect mode, skip GDINO)...')
            self._sam2_predictor = load_sam2_predictor(
                self.args.sam2_model_config,
                self.args.sam2_checkpoint,
                device,
            )
            self.get_logger().info('[detection_bbox] Initializing VLM detector...')
            self._vlm_detector = VLMDetector(
                api_key=self.args.vlm_api_key,
                base_url=self.args.vlm_base_url,
                model=self.args.vlm_model,
                debug=bool(self.args.vlm_debug_dir),
                scene_prompt_config=self._scene_prompt_config,
            )
            self._vlm_detector.warmup()
            self.get_logger().info(
                f'[detection_bbox] VLM detect enabled: {self.args.vlm_model} @ '
                f'{self.args.vlm_base_url}')
        else:
            self.get_logger().info('[detection_bbox] Loading Grounding DINO + SAM2...')
            self._grounding_model, self._sam2_predictor = load_gdino_sam_models(
                self.args.gdino_config,
                self.args.gdino_checkpoint,
                self.args.sam2_model_config,
                self.args.sam2_checkpoint,
                device,
            )
            # 场景理解需要 VLM (仅做物体发现，不做 bbox 检测)
            if getattr(self.args, 'use_scene_understand', False):
                self.get_logger().info(
                    '[detection_bbox] Initializing VLM for scene understanding...')
                self._vlm_detector = VLMDetector(
                    api_key=self.args.vlm_api_key,
                    base_url=self.args.vlm_base_url,
                    model=self.args.vlm_model,
                    debug=bool(self.args.vlm_debug_dir),
                    scene_prompt_config=self._scene_prompt_config,
                )
                self._vlm_detector.warmup()
                self.get_logger().info(
                    f'[detection_bbox] VLM scene understand: {self.args.vlm_model} @ '
                    f'{self.args.vlm_base_url}')
        self.get_logger().info(f'[detection_bbox] Models loaded on {device}')

        calib_file = self.get_parameter('calib_file').get_parameter_value().string_value.strip()
        if not calib_file:
            calib_file = default_calib_file()
        self.T_world_cam = None
        try:
            self.T_world_cam = load_T_world_cam(calib_file)
            if self.T_world_cam is not None:
                self.get_logger().info(
                    f'[detection_bbox] Loaded T_world_cam from {calib_file} '
                    f'(world frame: {self.args.workbench_frame_id})')
        except ValueError as e:
            self.get_logger().error(f'[detection_bbox] Failed to load calib: {e}')

        if self.T_world_cam is None:
            self.get_logger().warn(
                f'[detection_bbox] calib_file not available: {calib_file!r}, '
                'world 3D bbox / ROI filter disabled')
            if self.args.enable_world_roi_filter:
                self.get_logger().warn(
                    '[detection_bbox] enable_world_roi_filter=true but no calib — disabling ROI filter')
                self.args.enable_world_roi_filter = False
            if self.args.publish_workbench_bbox:
                self.get_logger().warn(
                    '[detection_bbox] publish_workbench_bbox=true but no calib — only 2D boxes')

        self._stereo_engine = None
        if self.args.use_stereo_depth:
            if not self.args.stereo_plan_path or not os.path.isfile(self.args.stereo_plan_path):
                self.get_logger().warn(
                    f'[detection_bbox] stereo plan not found: {self.args.stereo_plan_path}, '
                    'falling back to hardware depth')
                self.args.use_stereo_depth = False
            else:
                try:
                    self.get_logger().info(
                        f'[detection_bbox] Loading stereo engine: {self.args.stereo_plan_path}')
                    self._stereo_engine = load_stereo_engine(self.args.stereo_plan_path)
                except (FileNotFoundError, ImportError) as e:
                    self.get_logger().warn(
                        f'[detection_bbox] Stereo engine load failed: {e}, '
                        'falling back to hardware depth')
                    self.args.use_stereo_depth = False

        for cam_ns in self.args.cam_ns:
            ns = cam_ns if cam_ns.startswith('/') else '/' + cam_ns
            handler = CameraBBoxHandler(
                cam_ns=ns,
                node=self,
                args=self.args,
                grounding_model=self._grounding_model,
                sam2_predictor=self._sam2_predictor,
                stereo_engine=self._stereo_engine,
                T_world_cam=self.T_world_cam,
                device=device,
                vlm_detector=self._vlm_detector,
            )
            self.handlers.append(handler)

        self._current_targets_key = None
        self._apply_initial_targets()

    def _targets_key(self, targets):
        return tuple((t['label'], t['prompt']) for t in targets)

    def _activate_targets(self, targets):
        key = self._targets_key(targets)
        if key == self._current_targets_key:
            return
        if not targets:
            return
        for handler in self.handlers:
            try:
                handler.set_active_targets(targets)
            except ValueError as e:
                self.get_logger().warn(f'[detection_bbox] Target activation failed: {e}')
                return
        self._current_targets_key = key
        labels = [t['label'] for t in targets]
        self.get_logger().info(f'[detection_bbox] Activated targets: {labels}')

    def _activate_text_prompts(self, text_prompts):
        """Activate arbitrary text prompts (open-vocabulary, label = prompt)."""
        targets = targets_from_text_prompts(text_prompts)
        self._activate_targets(targets)

    def _apply_initial_targets(self):
        text_prompts = _get_string_list_param(self, 'text_prompts')
        tags = _get_string_list_param(self, 'tags')

        if self._pending_detection_prompts:
            self.get_logger().info(
                f'[detection_bbox] Applying buffered detection_prompts: '
                f'{self._pending_detection_prompts}')
            self._activate_from_strings(self._pending_detection_prompts)
        elif text_prompts:
            self._activate_text_prompts(text_prompts)
        elif tags:
            self._activate_from_strings(tags)
        elif self.args.auto_industry_scene_prompts:
            self.get_logger().info(
                f'[detection_bbox] Auto industry-scene prompts '
                f'({len(DEFAULT_INDUSTRY_SCENE_PROMPTS)} classes): '
                f'{DEFAULT_INDUSTRY_SCENE_PROMPTS[:4]}...')
            self._activate_text_prompts(list(DEFAULT_INDUSTRY_SCENE_PROMPTS))
        elif self.args.auto_lego_scene_prompts:
            self.get_logger().info(
                f'[detection_bbox] Auto lego-scene prompts '
                f'({len(DEFAULT_LEGO_SCENE_PROMPTS)} classes): '
                f'{DEFAULT_LEGO_SCENE_PROMPTS[:4]}...')
            self._activate_text_prompts(list(DEFAULT_LEGO_SCENE_PROMPTS))
        elif self.args.auto_open_scene_prompts:
            self.get_logger().info(
                f'[detection_bbox] Auto open-scene prompts ({len(DEFAULT_OPEN_SCENE_PROMPTS)} classes): '
                f'{DEFAULT_OPEN_SCENE_PROMPTS[:5]}... '
                '(GDINO only detects categories in caption, not arbitrary unknown objects)')
            self._activate_text_prompts(list(DEFAULT_OPEN_SCENE_PROMPTS))
        else:
            self.get_logger().warn(
                '[detection_bbox] No detection targets configured. '
                'Set text_prompts/tags at launch, enable auto_industry_scene_prompts, '
                'auto_lego_scene_prompts, auto_open_scene_prompts, '
                f'or publish to {DEFAULT_DETECTION_PROMPTS_TOPIC}')

    def _activate_from_strings(self, strings):
        """Activate from tag names and/or freeform strings (hybrid)."""
        try:
            targets = resolve_detection_targets(
                strings, allow_freeform=self.args.allow_freeform_prompts)
        except ValueError as e:
            self.get_logger().warn(
                f'[detection_bbox] Target activation failed: {e}. '
                'Open mode accepts arbitrary text prompts when allow_freeform_prompts=true.')
            return
        if not targets:
            self.get_logger().warn('[detection_bbox] No valid targets in detection_prompts message')
            return
        self._activate_targets(targets)

    def _on_detection_prompts(self, msg):
        strings = [str(t).strip() for t in msg.data if t and str(t).strip()]
        if not strings:
            self.get_logger().warn('[detection_bbox] Ignoring empty detection_prompts message')
            return
        self._pending_detection_prompts = strings
        if not self.handlers:
            self.get_logger().info(
                f'[detection_bbox] Buffered detection_prompts during model init: {strings}')
            return
        self.get_logger().info(f'[detection_bbox] detection_prompts received: {strings}')
        self._activate_from_strings(strings)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionBBoxNode()
    try:
        if node.args.use_stereo_depth:
            from rclpy.executors import MultiThreadedExecutor
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('[detection_bbox] Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
