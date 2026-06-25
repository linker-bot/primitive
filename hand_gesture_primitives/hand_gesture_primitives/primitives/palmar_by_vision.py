"""掌心抓握原语 — 五指全包络力控闭合，掌心为对侧支撑面。

抓取类型: 0 vs 2-5 (掌心 vs 食指+中指+无名指+小指，拇指同步包络)

适用场景: 强力抓取、掌心包络多面体或球体，
五指从各个方向均匀包络物体，掌心提供稳定支撑。

两阶段算法:
1. 预成型 (Pre-shape): 五指张开准备包络，根据物体尺寸调整拇指旋转
2. 同时力控闭合: 四指和拇指同时收紧，各自独立检测接触后停止
"""

import logging
import os
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..mesh_paths import default_mesh_model_dir
from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL, HAND_CONFIGS, HandConfig,
)

_logger = logging.getLogger(__name__)

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# --- 拇指定位参数 ---
_THUMB_ABD = 180
_THUMB_ROT_DEFAULT = 200
_THUMB_BASE = 80

# --- 预成型张开角度 ---
_PRESHAPE = {
    "thumb_tip": 50,
    "index_base": 70,
    "index_tip": 30,
    "middle_base": 70,
    "middle_tip": 30,
    "ring_base": 70,
    "ring_tip": 30,
    "pinky_base": 70,
    "pinky_tip": 30,
}

# --- 力控闭合参数 ---
FINGER_STEP = 8
THUMB_STEP = 8
PRESHAPE_DURATION = 0.5

# --- 物体尺寸 → thumb_rot 映射 ---
GRASP_WIDTH_MIN = 15.0
GRASP_WIDTH_MAX = 80.0
THUMB_ROT_SMALL = 220
THUMB_ROT_LARGE = 130

def _build_finger_joints(cfg):
    return [
        (cfg.index_base, cfg.index_tip),
        (cfg.middle_base, cfg.middle_tip),
        (cfg.ring_base, cfg.ring_tip),
        (cfg.pinky_base, cfg.pinky_tip),
    ]


def _width_to_thumb_rot(width_mm: float) -> int:
    t = (width_mm - GRASP_WIDTH_MIN) / (GRASP_WIDTH_MAX - GRASP_WIDTH_MIN)
    t = max(0.0, min(1.0, t))
    return int(THUMB_ROT_SMALL + t * (THUMB_ROT_LARGE - THUMB_ROT_SMALL))


def _load_mesh(label: str):
    try:
        import trimesh
    except ImportError:
        return None
    mesh_path = os.path.join(default_mesh_model_dir(), label, "model.obj")
    if not os.path.isfile(mesh_path):
        return None
    try:
        mesh = trimesh.load(mesh_path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(
                tuple(g for g in mesh.geometry.values())
            )
        return mesh
    except Exception:
        return None


class _Phase(Enum):
    PRESHAPE = 0
    CLOSE_FINGERS = 1
    CLOSE_THUMB = 2
    HOLD = 3


class PalmarByVision(HandGesturePrimitive):
    """掌心抓握 — 五指全包络力控闭合 (0 vs 2-5)。

    掌心作为对侧稳定支撑面，五指均匀包络物体。
    四指从正面包围物体，拇指从对侧包裹，
    适合需要掌心支撑的强力抓取场景。

    支持 O20 (20-DOF) 和 L25 (25-DOF) 两种手型。
    """

    def __init__(self):
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._finger_stopped: List[bool] = [False, False, False, False]
        self._thumb_stopped = False
        self._thumb_rot: Optional[int] = None
        self._cfg: Optional[HandConfig] = None

    @property
    def name(self) -> str:
        return "palmar_by_vision"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._finger_stopped = [False, False, False, False]
        self._thumb_stopped = False
        self._thumb_rot = None
        self._cfg = None

    def _get_cfg(self, ctx: PrimitiveContext) -> HandConfig:
        if self._cfg is None:
            self._cfg = HAND_CONFIGS[ctx.hand_type]
        return self._cfg

    def _compute_thumb_rot(self, ctx: PrimitiveContext) -> int:
        label = ctx.object_label
        if not label:
            return _THUMB_ROT_DEFAULT

        mesh = _load_mesh(label)
        if mesh is None:
            return _THUMB_ROT_DEFAULT

        if ctx.object_pose is not None:
            quat = ctx.object_pose.orientation
            R_obj = Rotation.from_quat(quat).as_matrix()
            verts = np.asarray(mesh.vertices) @ R_obj.T
        else:
            verts = np.asarray(mesh.vertices)

        has_real_tcp = False
        if ctx.tcp_pose is not None and ctx.object_pose is not None:
            dist = float(np.linalg.norm(ctx.tcp_pose.position - ctx.object_pose.position))
            if dist > 0.01:
                has_real_tcp = True

        if has_real_tcp:
            quat_tcp = ctx.tcp_pose.orientation
            R_tcp = Rotation.from_quat(quat_tcp).as_matrix()
            grasp_axis = R_tcp[:, 1]
            projections = verts @ grasp_axis
            width_mm = (projections.max() - projections.min()) * 1000.0
        else:
            aabb_min = verts.min(axis=0)
            aabb_max = verts.max(axis=0)
            extents = (aabb_max - aabb_min) * 1000.0
            # 取最大边，掌心包络需要容纳物体最大截面
            width_mm = max(extents)

        rot = _width_to_thumb_rot(width_mm)
        _logger.warning("palmar: 物体=%s, 宽度=%.1fmm → thumb_rot=%d", label, width_mm, rot)
        return rot

    def _build_preshape_target(self, cfg: HandConfig) -> List[float]:
        angles = [0.0] * cfg.num_joints
        # 拇指预成型
        angles[cfg.thumb_base] = _THUMB_BASE
        angles[cfg.thumb_abd] = _THUMB_ABD
        angles[cfg.thumb_rot] = self._thumb_rot if self._thumb_rot is not None else _THUMB_ROT_DEFAULT
        angles[cfg.thumb_tip] = _PRESHAPE["thumb_tip"]
        # 四指预成型
        angles[cfg.index_base] = _PRESHAPE["index_base"]
        angles[cfg.index_tip] = _PRESHAPE["index_tip"]
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        angles[cfg.middle_tip] = _PRESHAPE["middle_tip"]
        angles[cfg.ring_base] = _PRESHAPE["ring_base"]
        angles[cfg.ring_tip] = _PRESHAPE["ring_tip"]
        angles[cfg.pinky_base] = _PRESHAPE["pinky_base"]
        angles[cfg.pinky_tip] = _PRESHAPE["pinky_tip"]
        # L25 root2
        if cfg.thumb_root2 >= 0:
            angles[cfg.thumb_root2] = 40
        if cfg.middle_root2 >= 0:
            angles[cfg.middle_root2] = 30
        # 侧摆归中，四指收拢以增强包络稳定性
        for idx in cfg.abd_indices:
            if idx != cfg.thumb_abd:
                angles[idx] = cfg.abd_neutral
        return angles

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        cfg = self._get_cfg(ctx)
        finger_joints = _build_finger_joints(cfg)

        if ctx.tcp_pose is None:
            return self._hold("缺少 tcp_pose")

        # 无 object_pose 时用默认值，跳过可达性检查
        if ctx.object_pose is not None:
            tcp_to_obj = ctx.object_pose.position - ctx.tcp_pose.position
            tcp_dist = float(np.linalg.norm(tcp_to_obj))

            tcp_quat = ctx.tcp_pose.orientation
            if not np.allclose(tcp_quat, [0, 0, 0, 1]):
                R_tcp = Rotation.from_quat(tcp_quat).as_matrix()
                palm_forward = R_tcp[:, 2]
                forward_dist = float(np.dot(tcp_to_obj, palm_forward))
                if forward_dist < PALM_FORWARD_MIN or forward_dist > PALM_FORWARD_MAX:
                    return self._hold("物体不在掌心前方 2–15cm")
            else:
                if tcp_dist > REACH_THRESHOLD:
                    return self._hold("TCP到物体距离过远")

        # === Phase 1: 预成型 ===
        if self._phase == _Phase.PRESHAPE:
            if self._thumb_rot is None:
                self._thumb_rot = self._compute_thumb_rot(ctx)

            target = self._build_preshape_target(cfg)
            t = elapsed / PRESHAPE_DURATION
            if t >= 1.0:
                self._phase = _Phase.CLOSE_FINGERS
                self._close_angles = list(target)
                self._baseline_currents = list(ctx.joint_currents)
                _logger.warning("palmar: 进入四指闭合阶段")
                return self._move(list(target))
            return self._move(lerp_angles(self._start_angles, target, t))

        # === Phase 2: 四指同时力控闭合 ===
        if self._phase == _Phase.CLOSE_FINGERS:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles))

            base = self._baseline_currents

            # 四指各自独立力控
            for i, (base_idx, tip_idx) in enumerate(finger_joints):
                if self._finger_stopped[i]:
                    continue
                base_delta = currents[base_idx] - base[base_idx]
                tip_delta = currents[tip_idx] - base[tip_idx]
                if base_delta > ctx.contact_thresholds.current_delta_narrow or tip_delta > ctx.contact_thresholds.current_delta_narrow:
                    self._finger_stopped[i] = True
                    finger_names = ["食指", "中指", "无名指", "小指"]
                    _logger.warning("palmar: %s接触!", finger_names[i])
                else:
                    if self._close_angles[base_idx] < 255:
                        self._close_angles[base_idx] = min(
                            self._close_angles[base_idx] + FINGER_STEP, 255)
                    else:
                        self._finger_stopped[i] = True
                    if self._close_angles[tip_idx] < 255:
                        self._close_angles[tip_idx] = min(
                            self._close_angles[tip_idx] + FINGER_STEP, 255)

            if cfg.middle_root2 >= 0 and not self._finger_stopped[1]:
                if self._close_angles[cfg.middle_root2] < 255:
                    self._close_angles[cfg.middle_root2] = min(
                        self._close_angles[cfg.middle_root2] + FINGER_STEP, 255)

            # 四指都停 → 进入拇指闭合
            if all(self._finger_stopped):
                _logger.warning("palmar: 四指闭合完成 → 进入拇指闭合")
                self._phase = _Phase.CLOSE_THUMB
                self._settle_count = 0

            return self._move(list(self._close_angles))

        # === Phase 3: 拇指闭合 ===
        if self._phase == _Phase.CLOSE_THUMB:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles))

            base = self._baseline_currents

            if not self._thumb_stopped:
                thumb_tip_delta = currents[cfg.thumb_tip] - base[cfg.thumb_tip]
                thumb_base_delta = currents[cfg.thumb_base] - base[cfg.thumb_base]
                if thumb_tip_delta > ctx.contact_thresholds.current_delta_narrow or thumb_base_delta > ctx.contact_thresholds.current_delta_narrow:
                    self._thumb_stopped = True
                    _logger.warning("palmar: 拇指接触!")
                else:
                    if self._close_angles[cfg.thumb_tip] < 255:
                        self._close_angles[cfg.thumb_tip] = min(
                            self._close_angles[cfg.thumb_tip] + THUMB_STEP, 255)
                    else:
                        self._thumb_stopped = True
                    if self._close_angles[cfg.thumb_base] < 255:
                        self._close_angles[cfg.thumb_base] = min(
                            self._close_angles[cfg.thumb_base] + THUMB_STEP, 255)
                    if cfg.thumb_root2 >= 0 and self._close_angles[cfg.thumb_root2] < 255:
                        self._close_angles[cfg.thumb_root2] = min(
                            self._close_angles[cfg.thumb_root2] + THUMB_STEP, 255)

            if self._thumb_stopped:
                _logger.warning("palmar: 拇指闭合完成 → HOLD")
                self._phase = _Phase.HOLD

            return self._move(list(self._close_angles))

        # === Phase 4: 保持 ===
        currents = ctx.joint_currents
        monitored = [cfg.thumb_base, cfg.thumb_tip]
        for base_idx, tip_idx in finger_joints:
            monitored.extend([base_idx, tip_idx])
        if cfg.thumb_root2 >= 0:
            monitored.append(cfg.thumb_root2)
        if cfg.middle_root2 >= 0:
            monitored.append(cfg.middle_root2)
        for idx in monitored:
            if currents[idx] > ctx.contact_thresholds.hold_safe_current and self._close_angles[idx] > 0:
                over = currents[idx] - ctx.contact_thresholds.hold_safe_current
                step = max(1, int(over / 30))
                self._close_angles[idx] = max(0, self._close_angles[idx] - step)
        return self._move(list(self._close_angles))

    @property
    def done(self) -> bool:
        return False
