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

from ..angle_bridge import AngleUnits
from ..contact_config import scale_per_tick_delta, scaled_hold_retreat_step
from ..contact_resolver import current_index_for_s30, hw_current_delta
from ..gesture_engine import (
    PhasedLerpEngine,
    SequentialForceCloseEngine,
    make_palmar_by_vision_engine,
    make_palmar_by_vision_sfce_engine,
)
from ..gesture_params import load_palmar_by_vision_params
from ..hand_config import HandConfig as HardwareHandConfig
from ..mesh_paths import default_mesh_model_dir
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
    joint_present,
    lerp_angles,
    present_joints,
    semantic_pose_config,
    set_joint,
    tip_or_base,
)

_logger = logging.getLogger(__name__)

_CLOSE_DIGITS = ("index", "middle", "ring", "pinky")
_CLOSE_DIGIT_LABELS = ("食指", "中指", "无名指", "小指")

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
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
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
        self._engine: Optional[PhasedLerpEngine] = None
        self._sfce_engine: Optional[SequentialForceCloseEngine] = None
        self._engine_hand_type: str = ""
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._finger_stopped: List[bool] = [False, False, False, False]
        self._thumb_stopped = False
        self._thumb_rot: Optional[int] = None
        self._cfg: Optional[HandConfig] = None
        self._grasp_state: str = "approaching"

    @property
    def name(self) -> str:
        return "palmar_by_vision"

    @property
    def grasp_state(self) -> str:
        if self._engine is not None:
            return self._engine.grasp_state
        if self._sfce_engine is not None:
            return self._sfce_engine.grasp_state
        return self._grasp_state

    def _ensure_engine(self, ctx: PrimitiveContext) -> PhasedLerpEngine:
        if self._engine is None or self._engine_hand_type != ctx.hand_type:
            self._engine = make_palmar_by_vision_engine(
                ctx.hand_type, radian_converter=ctx.radian_converter
            )
            if self._engine is None:
                raise RuntimeError(f"palmar_by_vision config engine missing for {ctx.hand_type}")
            self._engine.reset(self._start_angles)
            self._engine_hand_type = ctx.hand_type
        return self._engine

    def _ensure_sfce_engine(self, ctx: PrimitiveContext) -> Optional[SequentialForceCloseEngine]:
        if self._sfce_engine is None or self._engine_hand_type != ctx.hand_type:
            self._sfce_engine = make_palmar_by_vision_sfce_engine(
                ctx.hand_type, radian_converter=ctx.radian_converter
            )
            if self._sfce_engine is not None:
                self._sfce_engine.reset(self._start_angles)
                self._engine_hand_type = ctx.hand_type
        return self._sfce_engine

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._sfce_engine = None
        self._engine_hand_type = ""
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._finger_stopped = [False, False, False, False]
        self._thumb_stopped = False
        self._thumb_rot = None
        self._cfg = None
        self._hw_cfg = None
        self._grasp_state = "approaching"

    def _get_cfg(self, ctx: PrimitiveContext) -> HandConfig:
        if self._cfg is None:
            self._cfg = semantic_pose_config(ctx.hand_type)
            self._hw_cfg = HardwareHandConfig(ctx.hand_type)
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
        angles = [0.0] * SEMANTIC_DIM
        # 拇指预成型
        angles[cfg.thumb_base] = _THUMB_BASE
        angles[cfg.thumb_abd] = _THUMB_ABD
        set_joint(
            angles,
            cfg.thumb_rot,
            self._thumb_rot if self._thumb_rot is not None else _THUMB_ROT_DEFAULT,
        )
        set_joint(angles, cfg.thumb_tip, _PRESHAPE["thumb_tip"])
        # 四指预成型
        angles[cfg.index_base] = _PRESHAPE["index_base"]
        set_joint(angles, cfg.index_tip, _PRESHAPE["index_tip"])
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        set_joint(angles, cfg.middle_tip, _PRESHAPE["middle_tip"])
        angles[cfg.ring_base] = _PRESHAPE["ring_base"]
        set_joint(angles, cfg.ring_tip, _PRESHAPE["ring_tip"])
        angles[cfg.pinky_base] = _PRESHAPE["pinky_base"]
        set_joint(angles, cfg.pinky_tip, _PRESHAPE["pinky_tip"])
        # L25 root2
        if cfg.thumb_root2 >= 0:
            angles[cfg.thumb_root2] = 40
        apply_four_finger_pip(angles, cfg, 30)
        # 侧摆归中，四指收拢以增强包络稳定性
        for idx in cfg.abd_indices:
            if idx != cfg.thumb_abd:
                angles[idx] = cfg.abd_neutral
        return angles

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        finger_step = scale_per_tick_delta(FINGER_STEP, ctx.motion_control_rate)
        thumb_step = scale_per_tick_delta(THUMB_STEP, ctx.motion_control_rate)
        if load_palmar_by_vision_params(ctx.hand_type) is not None:
            engine = self._ensure_engine(ctx)
            raw = engine.compute(current_angles, elapsed, ctx)
            return self._move(raw, space=u.space)

        sfce = self._ensure_sfce_engine(ctx)
        if sfce is not None:
            raw = sfce.compute(elapsed, ctx)
            return self._move(raw, space=u.space)

        cfg = self._get_cfg(ctx)

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

            # fallback 常量为 legacy 255，须经 u.pose 转到与 start_angles 同空间再 lerp
            target = u.pose(self._build_preshape_target(cfg))
            t = elapsed / PRESHAPE_DURATION
            if t >= 1.0:
                self._phase = _Phase.CLOSE_FINGERS
                self._close_angles = list(target)
                self._baseline_currents = list(ctx.joint_currents)
                self._grasp_state = "closing"
                _logger.warning("palmar: 进入四指闭合阶段")
                return self._move(list(target), space=u.space)
            return self._move(lerp_angles(self._start_angles, target, t), space=u.space)

        # === Phase 2: 四指同时力控闭合 ===
        if self._phase == _Phase.CLOSE_FINGERS:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles), space=u.space)

            base = self._baseline_currents

            # 四指各自独立力控
            for i, digit in enumerate(_CLOSE_DIGITS):
                if self._finger_stopped[i]:
                    continue
                status = digit_flex_close_step(
                    self._close_angles,
                    cfg,
                    digit,
                    currents,
                    base,
                    self._hw_cfg,
                    ctx.contact_thresholds.current_delta_narrow,
                    u.step(getattr(cfg, f"{digit}_base"), finger_step),
                    u.close_max(getattr(cfg, f"{digit}_base")),
                )
                if status != "continue":
                    self._finger_stopped[i] = True
                    if status == "contact":
                        _logger.warning("palmar: %s接触!", _CLOSE_DIGIT_LABELS[i])

            # 四指都停 → 进入拇指闭合
            if all(self._finger_stopped):
                _logger.warning("palmar: 四指闭合完成 → 进入拇指闭合")
                self._phase = _Phase.CLOSE_THUMB
                self._settle_count = 0

            return self._move(list(self._close_angles), space=u.space)

        # === Phase 3: 拇指闭合 ===
        if self._phase == _Phase.CLOSE_THUMB:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles), space=u.space)

            base = self._baseline_currents

            if not self._thumb_stopped:
                thumb_j = tip_or_base(cfg.thumb_tip, cfg.thumb_base)
                thumb_tip_delta = hw_current_delta(currents, base, self._hw_cfg, thumb_j)
                thumb_base_delta = hw_current_delta(currents, base, self._hw_cfg, cfg.thumb_base)
                if (
                    thumb_tip_delta > ctx.contact_thresholds.current_delta_narrow
                    or thumb_base_delta > ctx.contact_thresholds.current_delta_narrow
                ):
                    self._thumb_stopped = True
                    _logger.warning("palmar: 拇指接触!")
                else:
                    if bump_joint(
                        self._close_angles,
                        thumb_j,
                        u.step(thumb_j, thumb_step),
                        max_val=u.close_max(thumb_j),
                    ):
                        # 无 tip 时 thumb_j==base，达上限即停；有 tip 时 tip 达上限才停
                        if thumb_j == cfg.thumb_tip or not joint_present(cfg.thumb_tip):
                            self._thumb_stopped = True
                    if joint_present(cfg.thumb_tip) and self._close_angles[
                        cfg.thumb_base
                    ] < u.close_max(cfg.thumb_base):
                        self._close_angles[cfg.thumb_base] = min(
                            self._close_angles[cfg.thumb_base]
                            + u.step(cfg.thumb_base, thumb_step),
                            u.close_max(cfg.thumb_base),
                        )
                    if cfg.thumb_root2 >= 0 and self._close_angles[cfg.thumb_root2] < u.close_max(
                        cfg.thumb_root2
                    ):
                        self._close_angles[cfg.thumb_root2] = min(
                            self._close_angles[cfg.thumb_root2]
                            + u.step(cfg.thumb_root2, thumb_step),
                            u.close_max(cfg.thumb_root2),
                        )

            if self._thumb_stopped:
                _logger.warning("palmar: 拇指闭合完成 → HOLD")
                self._phase = _Phase.HOLD
                self._grasp_state = "contact"

            elif self._phase == _Phase.CLOSE_THUMB:
                self._grasp_state = "closing"

            return self._move(list(self._close_angles), space=u.space)

        # === Phase 4: 保持 ===
        self._grasp_state = "contact"
        currents = ctx.joint_currents
        monitored = present_joints(cfg.thumb_base, tip_or_base(cfg.thumb_tip, cfg.thumb_base))
        for digit in _CLOSE_DIGITS:
            monitored.extend(flex_chain_indices(cfg, digit))
        if cfg.thumb_root2 >= 0:
            monitored.append(cfg.thumb_root2)
        for idx in monitored:
            hi = current_index_for_s30(self._hw_cfg, idx)
            if hi is None or hi < 0 or hi >= len(currents):
                continue
            if (
                currents[hi] > ctx.contact_thresholds.hold_safe_current
                and self._close_angles[idx] > 0
            ):
                over = currents[hi] - ctx.contact_thresholds.hold_safe_current
                step = u.step(idx, scaled_hold_retreat_step(over, ctx.control_rate))
                self._close_angles[idx] = max(u.value(idx, 0), self._close_angles[idx] - step)
        return self._move(list(self._close_angles), space=u.space)

    @property
    def done(self) -> bool:
        return False
