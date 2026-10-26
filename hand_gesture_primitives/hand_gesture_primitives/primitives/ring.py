"""食中指环形包络原语 — 拇指与食指+中指力控闭合自适应抓取。

O6/L6：config sequential_force_close + SequentialForceCloseEngine（MCP 力控）。
O20/L25：legacy tip+rot 力控路径；写入 tip/rot 时用 set_joint（欠驱动索引为 -1）。
"""

import logging
import os
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..angle_bridge import AngleUnits
from ..contact_config import scale_per_tick_delta, scale_tick_count, scaled_hold_retreat_step
from ..contact_resolver import current_index_for_s30, hw_current_delta
from ..gesture_engine import SequentialForceCloseEngine, make_ring_engine
from ..gesture_params import load_ring_params
from ..hand_config import HandConfig as HardwareHandConfig
from ..primitive_base import (
    SEMANTIC_DIM,
    HandConfig,
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    apply_four_finger_pip,
    bump_joint,
    digit_flex_close_step,
    flex_chain_indices,
    lerp_angles,
    present_joints,
    semantic_pose_config,
    set_joint,
)
from ..semantic_30 import as_semantic_30

_logger = logging.getLogger(__name__)

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# --- 拇指定位参数 ---
_THUMB_ABD = 160
_THUMB_ROT_DEFAULT = 180
_THUMB_BASE = 40

# --- 预成型张开角度 ---
_PRESHAPE = {
    "thumb_tip": 30,
    "index_base": 90,
    "index_tip": 30,
    "middle_base": 90,
    "middle_tip": 30,
}

# --- 力控闭合参数 ---
CONTACT_DELTA = 270
FINGER_STEP = 8
THUMB_STEP = 8

PRESHAPE_DURATION = 0.5
SETTLE_FRAMES = 10
HOLD_SAFE_CURRENT = 800

# --- 物体尺寸 → thumb_rot 映射参数 ---
GRASP_WIDTH_MIN = 15.0
GRASP_WIDTH_MAX = 70.0
THUMB_ROT_SMALL = 220
THUMB_ROT_LARGE = 130

MESH_MODEL_DIR = "/botclaw/mesh_model"


def _width_to_thumb_rot(width_mm: float) -> float:
    """物体夹取方向投影尺寸 (mm) → thumb_rot 值。"""
    t = (width_mm - GRASP_WIDTH_MIN) / (GRASP_WIDTH_MAX - GRASP_WIDTH_MIN)
    t = max(0.0, min(1.0, t))
    return float(THUMB_ROT_SMALL + t * (THUMB_ROT_LARGE - THUMB_ROT_SMALL))


def _load_mesh(label: str):
    """加载离线 mesh，返回 trimesh 对象或 None。"""
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
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
        return mesh
    except Exception:
        return None


class _Phase(Enum):
    PRESHAPE = 0
    CLOSE_FINGERS = 1
    CLOSE_THUMB = 2
    HOLD = 3


def _default_thumb_rot_ring(hand_type: str) -> float:
    """L25 invert 下勿用 O20 默认 180（hw≈75 过度旋转）。"""
    if hand_type.lower() in ("l25", "g20"):
        return 140.0
    return _THUMB_ROT_DEFAULT


class Ring(HandGesturePrimitive):
    """拇指与食指+中指力控闭合抓取 (1 vs 2-3)。

    O6 为 MCP 顺序力控包络；O20/L25 保留独立指尖 + thumb_rot 算法。
    """

    def __init__(self):
        self._engine: Optional[SequentialForceCloseEngine] = None
        self._engine_hand_type: str = ""
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._index_stopped = False
        self._middle_stopped = False
        self._thumb_stopped = False
        self._thumb_rot: Optional[float] = None
        self._cfg: Optional[HandConfig] = None
        self._hw_cfg: Optional[HardwareHandConfig] = None
        self._grasp_state: str = "approaching"

    @property
    def name(self) -> str:
        return "ring"

    @property
    def grasp_state(self) -> str:
        if self._engine is not None:
            return self._engine.grasp_state
        return self._grasp_state

    def _ensure_engine(self, ctx: PrimitiveContext) -> SequentialForceCloseEngine:
        if self._engine is None or self._engine_hand_type != ctx.hand_type:
            self._engine = make_ring_engine(ctx.hand_type, radian_converter=ctx.radian_converter)
            if self._engine is None:
                raise RuntimeError(f"ring config engine missing for {ctx.hand_type}")
            self._engine.reset(self._start_angles)
            self._engine_hand_type = ctx.hand_type
        return self._engine

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._engine_hand_type = ""
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._index_stopped = False
        self._middle_stopped = False
        self._thumb_stopped = False
        self._thumb_rot = None
        self._cfg = None
        self._hw_cfg = None
        self._grasp_state = "approaching"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        if load_ring_params(ctx.hand_type) is not None:
            engine = self._ensure_engine(ctx)
            raw = engine.compute(elapsed, ctx)
            return self._move(raw, space=u.space)
        return self._compute_legacy(current_angles, elapsed, ctx)

    def _get_cfg(self, ctx: PrimitiveContext) -> HandConfig:
        if self._cfg is None:
            self._cfg = semantic_pose_config(ctx.hand_type)
            self._hw_cfg = HardwareHandConfig(ctx.hand_type)
        return self._cfg

    def _compute_thumb_rot(self, ctx: PrimitiveContext) -> float:
        """根据物体几何和TCP位姿计算最佳 thumb_rot。"""
        label = ctx.object_label
        if not label:
            return _default_thumb_rot_ring(ctx.hand_type)

        mesh = _load_mesh(label)
        if mesh is None:
            return _default_thumb_rot_ring(ctx.hand_type)

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
            width_mm = extents_sorted[1]  # 中间边，食中指包络覆盖面更大

        rot = _width_to_thumb_rot(width_mm)
        _logger.warning("ring: 物体=%s, 夹取宽度=%.1fmm → thumb_rot=%d", label, width_mm, rot)
        return rot

    def _build_preshape_target(self, cfg: HandConfig) -> List[float]:
        angles = [0.0] * SEMANTIC_DIM
        angles[cfg.thumb_base] = _THUMB_BASE
        angles[cfg.thumb_abd] = _THUMB_ABD
        set_joint(
            angles,
            cfg.thumb_rot,
            self._thumb_rot if self._thumb_rot is not None else _THUMB_ROT_DEFAULT,
        )
        set_joint(angles, cfg.thumb_tip, _PRESHAPE["thumb_tip"])
        # 食指预成型
        angles[cfg.index_base] = _PRESHAPE["index_base"]
        set_joint(angles, cfg.index_tip, _PRESHAPE["index_tip"])
        # 中指预成型
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        set_joint(angles, cfg.middle_tip, _PRESHAPE["middle_tip"])
        if cfg.thumb_root2 >= 0:
            angles[cfg.thumb_root2] = 30
        # O30 等独立 PIP：食/中/环/小统一预弯（勿只写 middle_root2）
        apply_four_finger_pip(angles, cfg, 30.0)
        # 侧摆归中
        for idx in cfg.abd_indices:
            if idx != cfg.thumb_abd and 0 <= idx < len(angles):
                angles[idx] = cfg.abd_neutral
        return angles

    def _compute_legacy(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        finger_step = scale_per_tick_delta(FINGER_STEP, ctx.motion_control_rate)
        thumb_step = scale_per_tick_delta(THUMB_STEP, ctx.motion_control_rate)
        cfg = self._get_cfg(ctx)
        hw = self._hw_cfg
        assert hw is not None

        # === Phase 1: 预成型 ===
        if self._phase == _Phase.PRESHAPE:
            if self._thumb_rot is None:
                self._thumb_rot = self._compute_thumb_rot(ctx)

            _raw_preshape = self._build_preshape_target(cfg)
            # 已是 S30 下标向量，禁止 [:20] 再当 O20 投影
            target = u.pose(as_semantic_30(_raw_preshape))
            t = elapsed / PRESHAPE_DURATION
            if t >= 1.0:
                self._phase = _Phase.CLOSE_FINGERS
                self._close_angles = list(target)
                self._baseline_currents = list(ctx.joint_currents)
                self._grasp_state = "closing"
                _logger.warning("ring: 进入手指闭合阶段（食指+中指先合）")
                return self._move(list(target), space=u.space)
            self._grasp_state = "approaching"
            return self._move(lerp_angles(self._start_angles, target, t), space=u.space)

        # === Phase 2: 先闭合食指和中指 ===
        if self._phase == _Phase.CLOSE_FINGERS:
            currents = ctx.joint_currents

            settle_frames = scale_tick_count(SETTLE_FRAMES, ctx.control_rate)
            if self._settle_count < settle_frames:
                self._settle_count += 1
                if self._settle_count == settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles), space=u.space)

            base = self._baseline_currents

            # 食/中屈伸链力控（含 O30 index_pip / middle_pip）
            if not self._index_stopped:
                status = digit_flex_close_step(
                    self._close_angles,
                    cfg,
                    "index",
                    currents,
                    base,
                    hw,
                    CONTACT_DELTA,
                    u.step(cfg.index_base, finger_step),
                    u.close_max(cfg.index_base),
                )
                if status != "continue":
                    self._index_stopped = True
                    if status == "contact":
                        _logger.warning("ring: 食指接触!")
            if not self._middle_stopped:
                status = digit_flex_close_step(
                    self._close_angles,
                    cfg,
                    "middle",
                    currents,
                    base,
                    hw,
                    CONTACT_DELTA,
                    u.step(cfg.middle_base, finger_step),
                    u.close_max(cfg.middle_base),
                )
                if status != "continue":
                    self._middle_stopped = True
                    if status == "contact":
                        _logger.warning("ring: 中指接触!")

            # 食指和中指都停 → 进入拇指闭合阶段
            if self._index_stopped and self._middle_stopped:
                _logger.warning("ring: 食指+中指闭合完成 → 进入拇指闭合")
                self._phase = _Phase.CLOSE_THUMB
                self._settle_count = 0

            self._grasp_state = "closing"
            return self._move(list(self._close_angles), space=u.space)

        # === Phase 3: 再闭合拇指（合到食指+中指中间） ===
        if self._phase == _Phase.CLOSE_THUMB:
            currents = ctx.joint_currents

            settle_frames = scale_tick_count(SETTLE_FRAMES, ctx.control_rate)
            if self._settle_count < settle_frames:
                self._settle_count += 1
                if self._settle_count == settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles), space=u.space)

            base = self._baseline_currents

            if not self._thumb_stopped:
                thumb_delta = max(
                    hw_current_delta(currents, base, hw, cfg.thumb_base),
                    hw_current_delta(currents, base, hw, cfg.thumb_tip),
                )
                if thumb_delta > CONTACT_DELTA:
                    self._thumb_stopped = True
                    _logger.warning("ring: 拇指接触!")
                else:
                    all_at_max = True
                    for ji in present_joints(cfg.thumb_base, cfg.thumb_tip):
                        if not bump_joint(
                            self._close_angles,
                            ji,
                            u.step(ji, thumb_step),
                            max_val=u.close_max(ji),
                        ):
                            all_at_max = False
                    if cfg.thumb_root2 >= 0 and self._close_angles[cfg.thumb_root2] < u.close_max(
                        cfg.thumb_root2
                    ):
                        self._close_angles[cfg.thumb_root2] = min(
                            self._close_angles[cfg.thumb_root2]
                            + u.step(cfg.thumb_root2, thumb_step),
                            u.close_max(cfg.thumb_root2),
                        )
                        all_at_max = False
                    if all_at_max:
                        self._thumb_stopped = True

            if self._thumb_stopped:
                _logger.warning("ring: 拇指闭合完成 → HOLD")
                self._phase = _Phase.HOLD
                self._grasp_state = "contact"
            else:
                self._grasp_state = "closing"

            return self._move(list(self._close_angles), space=u.space)

        # === Phase 4: 保持 ===
        self._grasp_state = "contact"
        currents = ctx.joint_currents
        monitored = []
        for digit in ("index", "middle"):
            monitored.extend(flex_chain_indices(cfg, digit))
        monitored.extend(present_joints(cfg.thumb_base, cfg.thumb_tip))
        if cfg.thumb_root2 >= 0:
            monitored.append(cfg.thumb_root2)
        for idx in monitored:
            hi = current_index_for_s30(hw, idx)
            if hi is None or hi < 0 or hi >= len(currents):
                continue
            if currents[hi] > HOLD_SAFE_CURRENT and self._close_angles[idx] > 0:
                over = currents[hi] - HOLD_SAFE_CURRENT
                step = u.step(idx, scaled_hold_retreat_step(over, ctx.control_rate))
                self._close_angles[idx] = max(u.value(idx, 0), self._close_angles[idx] - step)
        return self._move(list(self._close_angles), space=u.space)

    @property
    def done(self) -> bool:
        return False
