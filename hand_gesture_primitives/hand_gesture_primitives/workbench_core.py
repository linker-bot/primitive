# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Host-independent, validated pose and catalog services.

Positions at the editor boundary use the host's controller joint order and radians.
The host supplies joint limits and its hardware conversion; persistent poses keep the
existing semantic_v3/rad representation. No ROS or robot connection is created here.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

import numpy as np
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field


class PrimitiveRejectedError(ValueError):
    def __init__(self, reason: str, code: str = "primitive_unavailable"):
        super().__init__(reason)
        self.code = code
        self.reason = reason


class PoseFrame(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$", max_length=64)
    positions: list[float] = Field(min_length=1, max_length=64)
    duration: float = Field(default=0.6, ge=0.2, le=60)


class PoseDraft(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9_]+$", max_length=64)
    description: str = Field(default="", max_length=2000)
    keyframes: list[PoseFrame] = Field(min_length=1, max_length=100)
    revision: str = ""  # Empty means create, never silently overwrite.


@dataclass
class JointAdapter:
    """Controller conversion and limits supplied by the embedding application."""

    model: str
    side: str
    joint_names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    converter: object
    hardware_to_controller: Callable | None = None

    def __post_init__(self):
        from .hand_profile import load_hand_profile

        self.model = self.model.lower()
        if self.side not in {"left", "right"}:
            raise ValueError("未知手型或侧别")
        load_hand_profile(self.model)
        self.joint_names = list(self.joint_names)
        self.lower = np.asarray(self.lower, dtype=float).copy()
        self.upper = np.asarray(self.upper, dtype=float).copy()
        if (
            self.lower.shape != (len(self.joint_names),)
            or self.upper.shape != self.lower.shape
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.lower > self.upper)
            or len(set(self.joint_names)) != len(self.joint_names)
            or len(self.converter.limits) != len(self.joint_names)
        ):
            raise ValueError("关节名称、限位与换算器不匹配")

    @classmethod
    def from_urdf(cls, model, side, urdf):
        from .joint_state_mapping import Ros2ControlRadianConverter

        converter = Ros2ControlRadianConverter(urdf, model, side)
        names = [name.split("_hand_", 1)[-1] for name in converter.joint_names]
        lower, upper = np.asarray(converter.limits, dtype=float).T
        return cls(model, side, names, lower, upper, converter)

    @classmethod
    def from_limits(cls, model, side, joint_names, lower, upper):
        from .joint_state_mapping import ros2_control_joint_names

        full_names = ros2_control_joint_names(model, side)
        supplied = list(joint_names)
        suffixes = [name.split("_hand_", 1)[-1] for name in full_names]
        if supplied != suffixes and supplied != full_names:
            raise ValueError("关节顺序与当前手型控制器不匹配")
        if len(lower) != len(full_names) or len(upper) != len(full_names):
            raise ValueError("关节限位维度不符")
        root = ElementTree.Element("robot", name=f"{model}_{side}_workbench")
        ElementTree.SubElement(root, "link", name="base")
        for name, lo, hi in zip(full_names, lower, upper):
            ElementTree.SubElement(root, "link", name=name + "_link")
            joint = ElementTree.SubElement(root, "joint", name=name, type="revolute")
            ElementTree.SubElement(joint, "parent", link="base")
            ElementTree.SubElement(joint, "child", link=name + "_link")
            ElementTree.SubElement(
                joint, "limit", lower=str(lo), upper=str(hi), effort="1", velocity="1"
            )
        adapter = cls.from_urdf(model, side, ElementTree.tostring(root, encoding="unicode"))
        adapter.joint_names = supplied
        return adapter

    def validate_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all():
            raise ValueError("关节目标维度不符或含非有限数值")
        if np.any(q < self.lower - 1e-7) or np.any(q > self.upper + 1e-7):
            raise ValueError("关节目标超出当前手型限位")
        return q

    def joints(self):
        return [
            {"name": name, "lower": float(lo), "upper": float(hi)}
            for name, lo, hi in zip(self.joint_names, self.lower, self.upper)
        ]

    def from_hardware(self, positions):
        if self.hardware_to_controller is not None:
            return self.hardware_to_controller(positions)
        from .hand_profile import load_hand_profile
        from .joint_state_mapping import semantic_to_controller_hardware

        hardware = load_hand_profile(self.model).hardware
        semantic = hardware.to_semantic(positions)
        ordered = semantic_to_controller_hardware(self.model, semantic, hardware)
        return self.converter.hardware_uint8_to_radians(ordered)


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def snapshot(entry):
    from .workbench_support import pose_revision

    return {"revision": pose_revision(entry), "pose": asdict(entry)}


class WorkbenchService:
    """Synchronous library service; hosts run disk operations outside control ticks."""

    def __init__(self, adapter: JointAdapter):
        self.adapter = adapter
        self.model, self.side = adapter.model, adapter.side
        self.converter = adapter.converter

    def validate_q(self, q):
        return self.adapter.validate_q(q)

    def joints(self):
        return self.adapter.joints()

    def catalog(self):
        from .workbench_catalog import builtin_catalog

        return builtin_catalog(self.model)

    def draft_entry(self, draft: PoseDraft | dict):
        from .pose_library import PoseEntry, PoseKeyframe, validate_pose_name

        draft = draft if isinstance(draft, PoseDraft) else PoseDraft.model_validate(draft)
        error = validate_pose_name(draft.name)
        if error:
            raise ValueError(error)
        names = [frame.name for frame in draft.keyframes]
        if len(names) != len(set(names)):
            raise ValueError("关键帧名称不能重复")
        frames = tuple(
            PoseKeyframe(
                frame.name,
                self.converter.controller_to_semantic_rad(
                    self.validate_q(frame.positions).tolist()
                ),
                frame.duration,
            )
            for frame in draft.keyframes
        )
        return PoseEntry(
            name=draft.name,
            positions=frames[-1].positions,
            space="semantic_v3",
            unit="rad",
            duration=frames[-1].duration,
            num_joints=30,
            description=draft.description,
            source="workbench",
            keyframes=frames,
        )

    def entry_draft(self, entry):
        from .semantic_30 import project_o20_to_s30
        from .workbench_support import pose_revision

        frames = []
        for frame in entry.effective_keyframes():
            if entry.space == "hardware":
                q = self.adapter.from_hardware(frame.positions)
            else:
                sem = (
                    project_o20_to_s30(frame.positions)
                    if entry.space == "semantic_o20"
                    else frame.positions
                )
                q = (
                    self.converter.semantic_rad_to_controller(sem)
                    if entry.unit == "rad"
                    else self.converter.semantic_to_radians(sem)
                )
            frames.append(
                {
                    "name": frame.name,
                    "positions": self.validate_q(q).tolist(),
                    "duration": frame.duration,
                }
            )
        return {
            "name": entry.name,
            "description": entry.description,
            "keyframes": frames,
            "revision": pose_revision(entry),
        }

    def entries(self):
        from .pose_library import list_poses, load_pose

        return {
            name: load_pose(self.model, name, hand_side=self.side)
            for name in list_poses(self.model, hand_side=self.side)
        }

    def save(self, draft):
        from .pose_library import load_pose, upsert_pose

        draft = draft if isinstance(draft, PoseDraft) else PoseDraft.model_validate(draft)
        entry = self.draft_entry(draft)
        error = upsert_pose(
            self.model,
            entry.name,
            entry.positions,
            duration=entry.duration,
            description=entry.description,
            source=entry.source,
            space=entry.space,
            unit=entry.unit,
            hand_side=self.side,
            keyframes=list(entry.keyframes),
            replace_multi_keyframe=True,
            expected_revision=draft.revision,
        )
        if error:
            raise PrimitiveRejectedError(
                error,
                "revision_conflict" if error.startswith("revision_conflict") else "invalid_pose",
            )
        return self.entry_draft(load_pose(self.model, entry.name, hand_side=self.side))

    def delete(self, name, revision):
        from .pose_library import delete_pose

        if not revision:
            raise ValueError("删除需要姿态版本")
        try:
            if not delete_pose(self.model, name, hand_side=self.side, expected_revision=revision):
                raise ValueError("姿态不存在")
        except ValueError as exc:
            raise PrimitiveRejectedError(str(exc), "revision_conflict") from exc

    def menu_preferences(self, names=None):
        from .pose_library import pose_dir

        path = pose_dir() / "workbench-menus.json"
        if names is None and not path.exists():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=1.0):
            data = json.loads(path.read_text()) if path.exists() else {}
            if names is not None:
                data[f"{self.model}:{self.side}"] = list(dict.fromkeys(names))
                atomic_write_json(path, data)
            return data.get(f"{self.model}:{self.side}")

    def load_library(self):
        from .pose_library import pose_dir

        library = {name: entry for name, entry in self.entries().items() if entry is not None}
        errors, drafts = {}, []
        for name, entry in library.items():
            try:
                drafts.append(self.entry_draft(entry))
            except Exception as exc:
                errors[name] = f"保存姿态无效：{exc}"
        return {
            "entries": library,
            "errors": errors,
            "preferences": self.menu_preferences(),
            "public": {
                "model": self.model,
                "side": self.side,
                "joints": self.joints(),
                "directory": str(pose_dir()),
                "poses": drafts,
                "errors": errors,
            },
        }
