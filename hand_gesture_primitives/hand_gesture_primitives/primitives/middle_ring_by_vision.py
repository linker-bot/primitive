"""中指环形包络原语 — 拇指与中指上下错开，同时力控闭合自适应抓取。

两阶段算法:
1. 预成型 (Pre-shape): 根据物体尺寸自动选择 thumb_rot，快速 lerp 到预备姿态
2. 同时力控闭合: 中指和拇指同时收紧，各自独立检测接触后停止
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
PALM_FORWARD_MIN = 0.02  # 物体至少在掌心前方2cm
PALM_FORWARD_MAX = 0.15  # 物体最远不超过掌心前方15cm

# --- 拇指定位参数 ---
_THUMB_ABD = 160
_THUMB_ROT_DEFAULT = 200  # 无感知数据时的默认值
_THUMB_BASE = 50

# --- 预成型张开角度 ---
_PRESHAPE = {
    "thumb_tip": 30,
    "middle_base": 100,
    "middle_tip": 30,
}

# --- 力控闭合参数 ---
MIDDLE_STEP = 8
THUMB_STEP = 8
PRESHAPE_DURATION = 0.5

# --- 物体尺寸 → thumb_rot 映射参数 ---
GRASP_WIDTH_MIN = 15.0   # mm, 低于此用最小错开
GRASP_WIDTH_MAX = 60.0   # mm, 高于此用最大错开
THUMB_ROT_SMALL = 240    # 小错开（薄物体）
THUMB_ROT_LARGE = 150    # 大错开（粗物体）

def _width_to_thumb_rot(width_mm: float) -> int:
    """物体夹取方向投影尺寸 (mm) → thumb_rot 值。"""
    t = (width_mm - GRASP_WIDTH_MIN) / (GRASP_WIDTH_MAX - GRASP_WIDTH_MIN)
    t = max(0.0, min(1.0, t))
    return int(THUMB_ROT_SMALL + t * (THUMB_ROT_LARGE - THUMB_ROT_SMALL))


def _load_mesh(label: str):
    """加载离线 mesh，返回 trimesh 对象或 None。"""
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


class MiddleRingByVision(HandGesturePrimitive):
    """拇指与中指上下错开、同时力控闭合抓取。

    自动根据物体形状选择 thumb_rot（上下错开距离）：
    - 有 TCP 真实位姿时：沿夹取轴计算物体投影宽度
    - 无 TCP 时：用物体 bounding box 最短边估计
    - 无 mesh 时：使用默认值

    支持 O20 (20-DOF) 和 L25 (25-DOF) 两种手型，通过 ctx.hand_type 自动切换。
    """

    def __init__(self):
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._middle_stopped = False
        self._thumb_stopped = False
        self._thumb_rot: Optional[int] = None
        self._cfg: Optional[HandConfig] = None

    @property
    def name(self) -> str:
        return "middle_ring_by_vision"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._middle_stopped = False
        self._thumb_stopped = False
        self._thumb_rot = None
        self._cfg = None

    def _get_cfg(self, ctx: PrimitiveContext) -> HandConfig:
        if self._cfg is None:
            self._cfg = HAND_CONFIGS[ctx.hand_type]
        return self._cfg

    def _compute_thumb_rot(self, ctx: PrimitiveContext) -> int:
        """根据物体几何和TCP位姿计算最佳 thumb_rot。"""
        label = ctx.object_label
        if not label:
            _logger.warning("无物体标签, 使用默认 thumb_rot=%d", _THUMB_ROT_DEFAULT)
            return _THUMB_ROT_DEFAULT

        mesh = _load_mesh(label)
        if mesh is None:
            _logger.warning("无法加载 mesh: %s, 使用默认 thumb_rot=%d", label, _THUMB_ROT_DEFAULT)
            return _THUMB_ROT_DEFAULT

        # 用物体朝向旋转 mesh
        if ctx.object_pose is not None:
            quat = ctx.object_pose.orientation  # [qx, qy, qz, qw]
            R_obj = Rotation.from_quat(quat).as_matrix()
            verts = np.asarray(mesh.vertices) @ R_obj.T
        else:
            verts = np.asarray(mesh.vertices)

        # 判断是否有真实 TCP（tcp和object位置不同说明是真实的）
        has_real_tcp = False
        if ctx.tcp_pose is not None and ctx.object_pose is not None:
            dist = float(np.linalg.norm(ctx.tcp_pose.position - ctx.object_pose.position))
            if dist > 0.01:
                has_real_tcp = True

        if has_real_tcp:
            # 路径1: 用TCP朝向确定夹取轴
            # TCP y轴近似为手指闭合方向（拇指→中指方向）
            quat_tcp = ctx.tcp_pose.orientation
            R_tcp = Rotation.from_quat(quat_tcp).as_matrix()
            grasp_axis = R_tcp[:, 1]  # TCP y轴
            # 物体在夹取轴方向的投影
            projections = verts @ grasp_axis
            width_mm = (projections.max() - projections.min()) * 1000.0
            _logger.warning("路径1(TCP): 物体=%s, 夹取轴投影=%.1fmm", label, width_mm)
        else:
            # 路径2: 用 bounding box 最短边
            aabb_min = verts.min(axis=0)
            aabb_max = verts.max(axis=0)
            extents = (aabb_max - aabb_min) * 1000.0  # mm
            extents_sorted = sorted(extents)
            width_mm = extents_sorted[0]  # 最短边
            _logger.warning("路径2(AABB): 物体=%s, extents=[%.1f, %.1f, %.1f]mm, 最短边=%.1fmm",
                            label, extents[0], extents[1], extents[2], width_mm)

        rot = _width_to_thumb_rot(width_mm)
        _logger.warning("物体=%s, 夹取宽度=%.1fmm → thumb_rot=%d", label, width_mm, rot)
        return rot

    def _build_preshape_target(self, cfg: HandConfig) -> List[float]:
        angles = [0.0] * cfg.num_joints
        angles[cfg.thumb_base] = _THUMB_BASE
        angles[cfg.thumb_abd] = _THUMB_ABD
        angles[cfg.thumb_rot] = self._thumb_rot if self._thumb_rot is not None else _THUMB_ROT_DEFAULT
        angles[cfg.thumb_tip] = _PRESHAPE["thumb_tip"]
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        angles[cfg.middle_tip] = _PRESHAPE["middle_tip"]
        # L25 的 root2 关节也给一个初始弯曲
        if cfg.thumb_root2 >= 0:
            angles[cfg.thumb_root2] = 30
        if cfg.middle_root2 >= 0:
            angles[cfg.middle_root2] = 30
        # 侧摆归中
        for idx in cfg.abd_indices:
            if idx != cfg.thumb_abd:
                angles[idx] = cfg.abd_neutral
        return angles

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        cfg = self._get_cfg(ctx)

        if ctx.tcp_pose is None:
            return self._hold("缺少 tcp_pose")

        # 无 object_pose 时用默认 thumb_rot，跳过可达性检查
        # (auto_grasp_node 等外部节点已完成抓取判定)
        if ctx.object_pose is not None:
            # 判断物体是否在掌心前方可抓取范围内
            tcp_to_obj = ctx.object_pose.position - ctx.tcp_pose.position
            tcp_dist = float(np.linalg.norm(tcp_to_obj))

            # 有真实TCP朝向时，检查物体是否在掌心前方
            tcp_quat = ctx.tcp_pose.orientation
            if not np.allclose(tcp_quat, [0, 0, 0, 1]):
                R_tcp = Rotation.from_quat(tcp_quat).as_matrix()
                palm_forward = R_tcp[:, 2]  # TCP z轴 = 掌心朝向
                forward_dist = float(np.dot(tcp_to_obj, palm_forward))
                if forward_dist < PALM_FORWARD_MIN or forward_dist > PALM_FORWARD_MAX:
                    return self._hold("物体不在掌心前方 2–15cm")
            else:
                # 假位姿，只做简单距离检查
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
                _logger.warning("middle_ring: 进入中指闭合阶段")
                return self._move(list(target))
            return self._move(lerp_angles(self._start_angles, target, t))

        # === Phase 2: 先闭合中指 ===
        if self._phase == _Phase.CLOSE_FINGERS:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles))

            base = self._baseline_currents

            if not self._middle_stopped:
                mid_base_delta = currents[cfg.middle_base] - base[cfg.middle_base]
                mid_tip_delta = currents[cfg.middle_tip] - base[cfg.middle_tip]
                if mid_base_delta > ctx.contact_thresholds.current_delta or mid_tip_delta > ctx.contact_thresholds.current_delta:
                    self._middle_stopped = True
                    _logger.warning("middle_ring: 中指接触!")
                else:
                    if self._close_angles[cfg.middle_base] < 255:
                        self._close_angles[cfg.middle_base] = min(self._close_angles[cfg.middle_base] + MIDDLE_STEP, 255)
                    else:
                        self._middle_stopped = True
                    if self._close_angles[cfg.middle_tip] < 255:
                        self._close_angles[cfg.middle_tip] = min(self._close_angles[cfg.middle_tip] + MIDDLE_STEP, 255)
                    if cfg.middle_root2 >= 0 and self._close_angles[cfg.middle_root2] < 255:
                        self._close_angles[cfg.middle_root2] = min(self._close_angles[cfg.middle_root2] + MIDDLE_STEP, 255)

            if self._middle_stopped:
                _logger.warning("middle_ring: 中指闭合完成 → 进入拇指闭合")
                self._phase = _Phase.CLOSE_THUMB
                self._settle_count = 0

            return self._move(list(self._close_angles))

        # === Phase 3: 再闭合拇指 ===
        if self._phase == _Phase.CLOSE_THUMB:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles))

            base = self._baseline_currents

            if not self._thumb_stopped:
                thumb_delta = currents[cfg.thumb_tip] - base[cfg.thumb_tip]
                if thumb_delta > ctx.contact_thresholds.current_delta:
                    self._thumb_stopped = True
                    _logger.warning("middle_ring: 拇指接触!")
                else:
                    if self._close_angles[cfg.thumb_tip] < 255:
                        self._close_angles[cfg.thumb_tip] = min(self._close_angles[cfg.thumb_tip] + THUMB_STEP, 255)
                    else:
                        self._thumb_stopped = True
                    if cfg.thumb_root2 >= 0 and self._close_angles[cfg.thumb_root2] < 255:
                        self._close_angles[cfg.thumb_root2] = min(self._close_angles[cfg.thumb_root2] + THUMB_STEP, 255)

            if self._thumb_stopped:
                _logger.warning("middle_ring: 拇指闭合完成 → HOLD")
                self._phase = _Phase.HOLD

            return self._move(list(self._close_angles))

        # === Phase 4: 保持 ===
        currents = ctx.joint_currents
        monitored = [cfg.middle_base, cfg.middle_tip, cfg.thumb_tip]
        if cfg.middle_root2 >= 0:
            monitored.append(cfg.middle_root2)
        if cfg.thumb_root2 >= 0:
            monitored.append(cfg.thumb_root2)
        for idx in monitored:
            if currents[idx] > ctx.contact_thresholds.hold_safe_current and self._close_angles[idx] > 0:
                over = currents[idx] - ctx.contact_thresholds.hold_safe_current
                step = max(1, int(over / 30))
                self._close_angles[idx] = max(0, self._close_angles[idx] - step)
        return self._move(list(self._close_angles))

    @property
    def done(self) -> bool:
        return False
