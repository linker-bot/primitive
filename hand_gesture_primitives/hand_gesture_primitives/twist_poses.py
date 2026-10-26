"""twist* 原语共享：用 semantic_30 通道名构姿（P4，禁止裸 20 维表）。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping

from .gesture_params import _load_raw_gestures
from .primitive_base import ABD_NEUTRAL, sync_pip_from_dip
from .semantic_30 import s30_pose_from_channels

# 兼容 re-export（历史 import 路径）
__all__ = [
    "TwistPoseSet",
    "clear_twist_poses_cache",
    "load_twist_poses",
    "s30_pose_from_channels",
]


def _channels_from_sparse(raw: Mapping[str, Any], *, abd_neutral: bool) -> List[float]:
    channels: Dict[str, float] = {}
    for key, val in raw.items():
        channels[str(key)] = float(val)
    return s30_pose_from_channels(channels, abd_neutral=abd_neutral)


@dataclass(frozen=True)
class TwistPoseSet:
    """单 twist 原语的三段姿态（均为 S30）。"""

    ready: List[float]
    end: List[float]
    twist_delta: List[float]


# 内置默认（无 YAML 段时回退；与历史 O20 表投影等价）
_DEFAULT_TWIST: Dict[str, TwistPoseSet] = {
    "twist2digit": TwistPoseSet(
        ready=s30_pose_from_channels(
            {
                "thumb_mcp_flex": 0.0,
                "index_mcp_flex": 77.0,
                "middle_mcp_flex": 255.0,
                "ring_mcp_flex": 255.0,
                "pinky_mcp_flex": 255.0,
                "thumb_rot": 128.0,
                "thumb_dip": 90.0,
                "index_dip": 77.0,
                "middle_dip": 255.0,
                "ring_dip": 255.0,
                "pinky_dip": 255.0,
            }
        ),
        end=s30_pose_from_channels(
            {
                "thumb_mcp_flex": 102.0,
                "index_mcp_flex": 166.0,
                "middle_mcp_flex": 255.0,
                "ring_mcp_flex": 255.0,
                "pinky_mcp_flex": 255.0,
                "thumb_rot": 128.0,
                "thumb_dip": 90.0,
                "index_dip": 133.0,
                "middle_dip": 255.0,
                "ring_dip": 255.0,
                "pinky_dip": 255.0,
            }
        ),
        twist_delta=s30_pose_from_channels(
            {
                "thumb_mcp_flex": -26.0,
                "index_mcp_flex": 20.0,
                "thumb_mcp_abd": -200.0,
                "index_mcp_abd": -300.0,
                "thumb_dip": 13.0,
            },
            abd_neutral=False,
        ),
    ),
    "twist3digit": TwistPoseSet(
        ready=s30_pose_from_channels(
            {
                "thumb_mcp_flex": 0.0,
                "index_mcp_flex": 77.0,
                "middle_mcp_flex": 77.0,
                "ring_mcp_flex": 255.0,
                "pinky_mcp_flex": 255.0,
                "thumb_mcp_abd": 180.0,
                "index_mcp_abd": ABD_NEUTRAL + 50,
                "thumb_rot": 128.0,
                "thumb_dip": 128.0,
                "index_dip": 77.0,
                "middle_dip": 102.0,
                "ring_dip": 255.0,
                "pinky_dip": 255.0,
            }
        ),
        end=s30_pose_from_channels(
            {
                "thumb_mcp_flex": 102.0,
                "index_mcp_flex": 179.0,
                "middle_mcp_flex": 179.0,
                "ring_mcp_flex": 255.0,
                "pinky_mcp_flex": 255.0,
                "thumb_mcp_abd": 180.0,
                "index_mcp_abd": ABD_NEUTRAL + 50,
                "thumb_rot": 128.0,
                "thumb_dip": 128.0,
                "index_dip": 133.0,
                "middle_dip": 158.0,
                "ring_dip": 255.0,
                "pinky_dip": 255.0,
            }
        ),
        twist_delta=s30_pose_from_channels(
            {
                "thumb_mcp_flex": -26.0,
                "index_mcp_flex": -26.0,
                "middle_mcp_flex": 26.0,
                "thumb_mcp_abd": -150.0,
                "index_mcp_abd": -200.0,
                "middle_mcp_abd": -200.0,
                "thumb_dip": 13.0,
            },
            abd_neutral=False,
        ),
    ),
    "twist5digit": TwistPoseSet(
        ready=s30_pose_from_channels(
            {
                "thumb_mcp_flex": 0.0,
                "index_mcp_flex": 102.0,
                "middle_mcp_flex": 77.0,
                "ring_mcp_flex": 89.0,
                "pinky_mcp_flex": 128.0,
                "thumb_mcp_abd": 220.0,
                "index_mcp_abd": ABD_NEUTRAL + 50,
                "middle_mcp_abd": ABD_NEUTRAL,
                "ring_mcp_abd": ABD_NEUTRAL - 25,
                "pinky_mcp_abd": ABD_NEUTRAL - 50,
                "thumb_rot": 150.0,
                "thumb_dip": 77.0,
                "index_dip": 51.0,
                "middle_dip": 77.0,
                "ring_dip": 77.0,
                "pinky_dip": 77.0,
            }
        ),
        end=s30_pose_from_channels(
            {
                "thumb_mcp_flex": 102.0,
                "index_mcp_flex": 191.0,
                "middle_mcp_flex": 179.0,
                "ring_mcp_flex": 191.0,
                "pinky_mcp_flex": 255.0,
                "thumb_mcp_abd": 220.0,
                "index_mcp_abd": ABD_NEUTRAL + 50,
                "middle_mcp_abd": ABD_NEUTRAL,
                "ring_mcp_abd": ABD_NEUTRAL - 25,
                "pinky_mcp_abd": ABD_NEUTRAL - 50,
                "thumb_rot": 150.0,
                "thumb_dip": 77.0,
                "index_dip": 80.0,
                "middle_dip": 80.0,
                "ring_dip": 90.0,
                "pinky_dip": 100.0,
            }
        ),
        twist_delta=s30_pose_from_channels(
            {
                "thumb_mcp_flex": -16.0,
                "index_mcp_flex": -35.0,
                "middle_mcp_flex": 26.0,
                "ring_mcp_flex": 35.0,
                "thumb_mcp_abd": -150.0,
                "index_mcp_abd": -200.0,
                "middle_mcp_abd": -200.0,
                "ring_mcp_abd": -100.0,
                "pinky_mcp_abd": -50.0,
                "thumb_dip": 13.0,
            },
            abd_neutral=False,
        ),
    ),
}


def _parse_twist_block(block: dict, primitive: str) -> TwistPoseSet:
    angles = block.get("angles") or {}
    if not isinstance(angles, dict):
        raise ValueError(f"{primitive} angles 必须是对象")
    space = str(angles.get("space", "semantic_v3")).lower()
    if space not in ("semantic_v3", "semantic_o20"):
        raise ValueError(f"{primitive} angles.space 无效: {space!r}")

    def _one(key: str, *, abd_neutral: bool) -> List[float]:
        raw = angles.get(key)
        if raw is None:
            raise KeyError(key)
        if not isinstance(raw, dict):
            raise ValueError(f"{primitive} angles.{key} 必须是通道名映射")
        return _channels_from_sparse(raw, abd_neutral=abd_neutral)

    return TwistPoseSet(
        ready=_one("ready", abd_neutral=True),
        end=_one("end", abd_neutral=True),
        twist_delta=_one("twist_delta", abd_neutral=False),
    )


@lru_cache(maxsize=32)
def load_twist_poses(hand_type: str, primitive: str) -> TwistPoseSet:
    """从 ``gestures.{primitive}`` 加载 twist 姿态；缺省回退内置默认。"""
    key = primitive.lower().strip()
    if key not in _DEFAULT_TWIST:
        raise ValueError(f"未知 twist 原语: {primitive!r}")
    gestures = _load_raw_gestures(hand_type.lower())
    block = gestures.get(key)
    if not isinstance(block, dict) or not block.get("angles"):
        poses = _DEFAULT_TWIST[key]
    else:
        poses = _parse_twist_block(block, key)
    # O30 等独立 PIP：ready/end 的 pip 跟随 dip；无独立轴由 mask 消化
    return TwistPoseSet(
        ready=sync_pip_from_dip(poses.ready),
        end=sync_pip_from_dip(poses.end),
        twist_delta=poses.twist_delta,
    )


def clear_twist_poses_cache() -> None:
    load_twist_poses.cache_clear()
