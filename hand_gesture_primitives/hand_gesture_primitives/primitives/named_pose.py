"""命名示教姿态回放 — PoseLibrary + KeyframeRunner。

支持：
- 单点旧姿态（无 keyframes）
- 多关键帧 ``keyframes[]`` 顺序 AUTO
- ``pose <name> <kf> [progress]`` 段内 progress 细控

hardware 落盘：在完整硬件 DOF 上插值直出（避免 L25 经 from_hardware 丢指尖）。
semantic 落盘：有 Converter 时在 semantic_rad 插值；否则 legacy255 S30。
"""

from typing import List, Optional

from ..angle_bridge import (
    SPACE_HARDWARE,
    SPACE_LEGACY_SEMANTIC255,
    SPACE_SEMANTIC_RAD,
    start_angles_for_rad_engine,
)
from ..hand_config import HandConfig, semantic_legacy255_to_hardware
from ..keyframe_runner import DriveMode, KeyframeRunner, KeyframeSegment, MotionProgram
from ..pose_library import load_pose, motion_program_from_entry
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
)
from ..semantic_30 import SEMANTIC_DIM, as_semantic_30, project_o20_to_s30


class NamedPose(HandGesturePrimitive):
    """回放用户示教姿态（hardware / semantic_v3 / semantic_o20；可多关键帧）。"""

    def __init__(
        self,
        pose_name: str,
        *,
        keyframe: Optional[str] = None,
        progress: Optional[float] = None,
        auto_segment: bool = False,
        entry=None,
    ) -> None:
        from copy import deepcopy
        self._entry_snapshot = deepcopy(entry)
        self._pose_name = (pose_name or "").strip().lower()
        self._request_keyframe = (keyframe or "").strip().lower() or None
        self._request_progress = None if progress is None else max(0.0, min(1.0, float(progress)))
        if self._request_progress is not None:
            import math

            if not math.isfinite(self._request_progress):
                self._request_progress = None
        # 指定关键帧且无 progress：AUTO 只跑该段后 done（不进入 PROGRESS hold）
        self._auto_segment = bool(auto_segment) and bool(self._request_keyframe)
        if self._auto_segment and self._request_progress is not None:
            # 显式 progress 优先，退回细控
            self._auto_segment = False
        self._runner: Optional[KeyframeRunner] = None
        self._hold_reason: str = ""
        self._done = False
        self._use_hardware = False
        self._result_space = SPACE_SEMANTIC_RAD

    @property
    def name(self) -> str:
        return f"named_pose:{self._pose_name}"

    @property
    def pose_name(self) -> str:
        return self._pose_name

    @property
    def request_keyframe(self) -> Optional[str]:
        return self._request_keyframe

    @property
    def active_keyframe(self) -> Optional[str]:
        """当前段名：优先 runner，否则请求的 keyframe。"""
        if self._runner is not None:
            return self._runner.keyframe
        return self._request_keyframe

    @property
    def runner_ready(self) -> bool:
        return self._runner is not None

    @property
    def motion_status(self) -> str:
        if self._runner is None:
            kf = self._request_keyframe or "-"
            p = 0.0 if self._request_progress is None else float(self._request_progress)
            if self._auto_segment:
                drive = DriveMode.AUTO.value
            elif self._request_progress is not None or self._request_keyframe:
                drive = DriveMode.PROGRESS.value
            else:
                drive = DriveMode.AUTO.value
            return f"drive={drive}|keyframe={kf}|progress={p:.3f}"
        return (
            f"drive={self._runner.drive_mode}"
            f"|keyframe={self._runner.keyframe}"
            f"|progress={self._runner.progress:.3f}"
        )

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._runner = None
        self._hold_reason = ""
        self._done = False
        self._use_hardware = False
        self._result_space = SPACE_SEMANTIC_RAD

    def supports_progress(self) -> bool:
        return True

    @property
    def progress_driven(self) -> bool:
        if self._auto_segment:
            return False
        if self._runner is not None:
            return self._runner.drive_mode == DriveMode.PROGRESS.value
        return self._request_progress is not None or bool(self._request_keyframe)

    def set_progress(self, p: float) -> None:
        """同段热更新 progress（供 motion_progress topic / pose 指令）。

        runner 尚未由首帧 compute 创建时，先写入 ``_request_progress``，
        避免 ``pose …`` 后立刻发 motion_progress 被静默丢弃。
        """
        import math

        value = float(p)
        if not math.isfinite(value):
            return
        clamped = max(0.0, min(1.0, value))
        self._request_progress = clamped
        self._done = False
        if self._runner is not None:
            self._runner.set_drive_progress(clamped)

    def switch_keyframe(
        self,
        keyframe: str,
        current_q: List[float],
        *,
        progress: float = 0.0,
        elapsed: float = 0.0,
    ) -> None:
        """同 pose 内换段：重锚到 current_q，进入 PROGRESS。"""
        if self._runner is None:
            raise RuntimeError("KeyframeRunner 尚未创建")
        name = (keyframe or "").strip().lower()
        if not name:
            raise ValueError("keyframe 为空")
        p = max(0.0, min(1.0, float(progress)))
        self._request_keyframe = name
        self._request_progress = p
        self._runner.select_segment(
            name,
            current_q,
            elapsed=elapsed,
            drive=DriveMode.PROGRESS,
            progress=p,
        )
        self._done = False

    def _legacy_start_from_context(self, ctx: PrimitiveContext) -> List[float]:
        """on_enter 缓存可能是 semantic_rad，回退到 legacy255 再 to_hardware。"""
        start = list(self._start_angles)
        conv = ctx.radian_converter
        unit = str(ctx.angle_unit or "").lower()
        if conv is not None and unit in (SPACE_SEMANTIC_RAD, "radians", "rad"):
            return conv.semantic_rad_to_legacy255(as_semantic_30(start))
        return as_semantic_30(start)

    def _convert_target_vector(
        self, positions: List[float], entry_space: str, entry_unit: str, ctx: PrimitiveContext
    ) -> List[float]:
        if entry_space == "hardware":
            return [float(v) for v in positions]
        if entry_space == "semantic_v3":
            semantic = [float(v) for v in positions]
            while len(semantic) < SEMANTIC_DIM:
                semantic.append(0.0)
            semantic = semantic[:SEMANTIC_DIM]
        else:
            o20 = [float(v) for v in positions]
            while len(o20) < 20:
                o20.append(0.0)
            semantic = project_o20_to_s30(o20[:20])

        unit = str(entry_unit or "legacy255").lower()
        conv = ctx.radian_converter
        if unit == "rad":
            return semantic
        if conv is not None:
            return conv.legacy255_to_semantic_rad(semantic)
        return semantic

    def _build_start_vector(self, entry, ctx: PrimitiveContext) -> List[float]:
        hand = HandConfig(ctx.hand_type)
        if entry.space == "hardware":
            hw_cmd = ctx.hand_cmd_angles
            hw_fb = ctx.hand_hw_angles
            if isinstance(hw_cmd, list) and len(hw_cmd) == hand.num_joints:
                hw_start = [float(v) for v in hw_cmd]
            elif isinstance(hw_fb, list) and len(hw_fb) == hand.num_joints:
                hw_start = [float(v) for v in hw_fb]
            else:
                hw_start = semantic_legacy255_to_hardware(
                    hand, self._legacy_start_from_context(ctx)
                )
            n = hand.num_joints
            while len(hw_start) < n:
                hw_start.append(0.0)
            return hw_start[:n]

        conv = ctx.radian_converter
        start = start_angles_for_rad_engine(
            self._start_angles,
            radian_converter=conv,
            start_unit=str(ctx.angle_unit or SPACE_LEGACY_SEMANTIC255),
        )
        while len(start) < SEMANTIC_DIM:
            start.append(0.0)
        return start[:SEMANTIC_DIM]

    def _ensure_runner(self, ctx: PrimitiveContext) -> Optional[KeyframeRunner]:
        if self._runner is not None:
            return self._runner
        if not self._pose_name:
            self._hold_reason = "姿态名为空"
            return None
        entry = self._entry_snapshot or load_pose(
            ctx.hand_type,
            self._pose_name,
            hand_side=ctx.hand_side or None,
        )
        if entry is None:
            self._hold_reason = f"姿态不存在: {self._pose_name}"
            return None

        if entry.space == "hardware":
            self._use_hardware = True
            self._result_space = SPACE_HARDWARE
        else:
            self._use_hardware = False
            unit = str(getattr(entry, "unit", "legacy255") or "legacy255").lower()
            conv = ctx.radian_converter
            if unit == "rad" and conv is None:
                self._hold_reason = (
                    f"姿态 {self._pose_name} 为 unit=rad，需要 URDF radian_converter"
                )
                return None
            self._result_space = (
                SPACE_SEMANTIC_RAD if conv is not None else SPACE_LEGACY_SEMANTIC255
            )

        program = motion_program_from_entry(entry)
        # 运行时目标空间：hardware 直出；semantic 转 rad/legacy
        converted_segs = []

        for seg in program.segments:
            tgt = self._convert_target_vector(list(seg.target), entry.space, entry.unit, ctx)
            converted_segs.append(
                KeyframeSegment(name=seg.name, target=tuple(tgt), duration=seg.duration)
            )
        program = MotionProgram(
            ref=program.ref,
            space=entry.space,
            unit=entry.unit,
            segments=converted_segs,
        )

        start = self._build_start_vector(entry, ctx)
        runner = KeyframeRunner()
        seg_index = 0
        if self._request_keyframe:
            try:
                seg_index = program.find_segment_index(self._request_keyframe)
            except KeyError:
                available = program.segment_names()
                self._hold_reason = f"关键帧不存在: {self._request_keyframe}；可用: {available}"
                return None

        if self._request_progress is not None:
            drive = DriveMode.PROGRESS
            progress = float(self._request_progress)
        elif self._auto_segment:
            drive = DriveMode.AUTO
            progress = 0.0
        elif self._request_keyframe:
            drive = DriveMode.PROGRESS
            progress = 0.0
        else:
            drive = DriveMode.AUTO
            progress = 0.0

        runner.reset(
            program,
            start,
            drive=drive,
            segment_index=seg_index,
            progress=progress,
            elapsed=0.0,
            stop_after_segment=self._auto_segment,
        )
        self._runner = runner
        return self._runner

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        runner = self._ensure_runner(ctx)
        if runner is None:
            return self._hold(self._hold_reason or "named_pose 加载失败")
        target = runner.compute(elapsed)
        self._done = bool(runner.done)
        if self._use_hardware:
            result = self._move(target, space=SPACE_HARDWARE)
        else:
            result = self._move(target, space=self._result_space)
        result.trajectory_finished = bool(runner.done)
        result.keyframe = runner.keyframe
        result.progress = runner.progress
        result.drive_mode = runner.drive_mode
        return result

    @property
    def done(self) -> bool:
        return self._done
