"""三指捏取原语 — 拇指与食指+中指指尖对捏，形成三角支撑。

抓取类型: 1 vs 2-3 (拇指 vs 食指+中指)

适用场景: 精细捏取中等物体，三指形成稳定三角支撑，
如捏笔、捏螺丝刀柄、捏小立方体等。
拇指从一侧、食指和中指并拢从对侧对捏，无名指和小指自然弯曲避让。

两阶段算法:
1. 预成型 (Pre-shape): 食指+中指张开准备，拇指预就位
2. 同时力控闭合: 食指、中指和拇指同时收紧，各自独立检测接触后停止
"""

import logging
import os
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL, HAND_CONFIGS, HandConfig,
)

_logger = logging.getLogger(__name__)

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# --- 拇指定位参数 ---
_THUMB_ABD = 155
_THUMB_ROT_DEFAULT = 190
_THUMB_BASE = 45

# --- 预成型张开角度 ---
_PRESHAPE = {
    "thumb_tip": 30,
    "index_base": 85,
    "index_tip": 25,
    "middle_base": 85,
    "middle_tip": 25,
}

# --- 力控闭合参数 ---
CONTACT_DELTA = 270
FINGER_STEP = 7
THUMB_STEP = 7

PRESHAPE_DURATION = 0.5
SETTLE_FRAMES = 10
HOLD_SAFE_CURRENT = 800

# --- 物体尺寸 → thumb_rot 映射 ---
GRASP_WIDTH_MIN = 5.0
GRASP_WIDTH_MAX = 40.0
THUMB_ROT_SMALL = 210
THUMB_ROT_LARGE = 140

MESH_MODEL_DIR = "/botclaw/mesh_model"


TRIPOD_ANGLES = [
    139,          # [0]  thumb_base: 伸直
    184,          # [1]  index_base: 伸直
    163,          # [2]  middle_base: 伸直
    0,          # [3]  ring_base: 伸直
    0,          # [4]  pinky_base: 伸直
    227,        # [5]  thumb_abd: 外展张开对向四指
    193,         # [6]  index_abd: 内收并拢
    255,         # [7]  middle_abd: 内收并拢
    105,         # [8]  ring_abd: 内收并拢
    52,         # [9]  pinky_abd: 内收并拢
    110,        # [10] thumb_rot: 旋转使指腹平行于四指面
    0, 0, 0, 0,  # [11-14] rsv
    143,          # [15] thumb_tip: 伸直
    0,          # [16] index_tip: 伸直
    80,          # [17] middle_tip: 伸直
    0,          # [18] ring_tip: 伸直
    0,          # [19] pinky_tip: 伸直
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
    mesh_path = os.path.join(MESH_MODEL_DIR, label, "model.obj")
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


class Tripod(HandGesturePrimitive):
    """拇指与食指+中指三指对捏 (1 vs 2-3)。

    三角支撑精密捏取：食指和中指并拢从正面抵住物体，
    拇指从对侧精确对捏。三指同时力控闭合，
    各自独立检测接触后停止。

    与 ring 的区别: tripod_by_vision 是精密指尖捏取 (fingertip pinch)，
    ring 是环形包络 (envelope grasp)，手指弯曲更深。

    支持 O20 (20-DOF) 和 L25 (25-DOF) 两种手型。
    """

    TRANSITION_DURATION = 0.6

    def __init__(self):
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._index_stopped = False
        self._middle_stopped = False
        self._thumb_stopped = False
        self._thumb_rot: Optional[int] = None
        self._cfg: Optional[HandConfig] = None

    @property
    def name(self) -> str:
        return "tripod"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._index_stopped = False
        self._middle_stopped = False
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
            extents_sorted = sorted(extents)
            # 三指捏取取中间边
            width_mm = extents_sorted[1]

        rot = _width_to_thumb_rot(width_mm)
        _logger.warning("tripod_by_vision: 物体=%s, 宽度=%.1fmm → thumb_rot=%d", label, width_mm, rot)
        return rot

    def _build_preshape_target(self, cfg: HandConfig) -> List[float]:
        angles = [0.0] * cfg.num_joints
        # 拇指预成型
        angles[cfg.thumb_base] = _THUMB_BASE
        angles[cfg.thumb_abd] = _THUMB_ABD
        angles[cfg.thumb_rot] = self._thumb_rot if self._thumb_rot is not None else _THUMB_ROT_DEFAULT
        angles[cfg.thumb_tip] = _PRESHAPE["thumb_tip"]
        # 食指预成型
        angles[cfg.index_base] = _PRESHAPE["index_base"]
        angles[cfg.index_tip] = _PRESHAPE["index_tip"]
        # 中指预成型（与食指并拢）
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        angles[cfg.middle_tip] = _PRESHAPE["middle_tip"]
        # 无名指和小指自然弯曲避让
        angles[cfg.ring_base] = 30
        angles[cfg.ring_tip] = 30
        angles[cfg.pinky_base] = 30
        angles[cfg.pinky_tip] = 30
        # L25 root2
        if cfg.thumb_root2 >= 0:
            angles[cfg.thumb_root2] = 25
        if cfg.middle_root2 >= 0:
            angles[cfg.middle_root2] = 25
        # 食指+中指并拢 (侧摆内收)，abd_indices = [thumb_abd, index_abd, middle_abd, ring_abd, pinky_abd] or O20
        for idx in cfg.abd_indices:
            if idx == cfg.thumb_abd:
                continue
            # index_abd(6) 和 middle_abd(7) 略内收使食指中指并拢
            if idx in (6, 7):
                angles[idx] = 100
            else:
                angles[idx] = cfg.abd_neutral
        return angles

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:

        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(TRIPOD_ANGLES))
        return self._move(lerp_angles(self._start_angles, TRIPOD_ANGLES, t))


    @property
    def done(self) -> bool:
        return False
