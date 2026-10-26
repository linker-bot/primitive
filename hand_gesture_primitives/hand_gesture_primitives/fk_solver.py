"""正运动学求解器 — 基于 URDF + Pinocchio 计算手指尖末端位置。"""

import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ROS Jazzy 的 pinocchio 扩展按 NumPy 1.x ABI 编译。用户目录若装了 NumPy 2.x
# （常见于 pip install --user），import pinocchio 会 ImportError 甚至 SIGSEGV。
# 必须在 import 前拦截，不能依赖 try/except（segfault 杀进程）。
_PINOCCHIO_UNAVAILABLE_REASON = ""
pin = None  # type: ignore
_PINOCCHIO_AVAILABLE = False


def _numpy_major() -> int:
    try:
        return int(str(np.__version__).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _try_import_pinocchio() -> None:
    """延迟导入 pinocchio；NumPy≥2 时直接跳过，避免 ABI 崩溃。"""
    global pin, _PINOCCHIO_AVAILABLE, _PINOCCHIO_UNAVAILABLE_REASON
    if _PINOCCHIO_AVAILABLE or _PINOCCHIO_UNAVAILABLE_REASON:
        return
    if _numpy_major() >= 2:
        _PINOCCHIO_UNAVAILABLE_REASON = (
            f"NumPy {np.__version__}（路径 {getattr(np, '__file__', '?')}）与 "
            "ROS pinocchio（NumPy 1.x ABI）不兼容。若因 pyzed/ZED 需要保留 NumPy 2，"
            "请仅对 gesture_node 隔离用户 site：`PYTHONNOUSERSITE=1`（机载 launch 已默认设置），"
            "勿全局卸载 NumPy 2。"
        )
        return
    try:
        import pinocchio as pinocchio_mod

        pin = pinocchio_mod
        _PINOCCHIO_AVAILABLE = True
    except ImportError as exc:
        _PINOCCHIO_UNAVAILABLE_REASON = f"pinocchio 不可用: {exc}"
        pin = None


# 配置目录 (source tree)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _parse_fk_joint_map(raw: dict) -> Dict[int, str]:
    """解析 FK 关节映射 → **O20 下标**（MOCK / ros2_control 兼容）。

    键可为：
    - ``int`` / 数字字符串：O20 通道下标（现网 ``o20_to_urdf_joint``）
    - ``semantic_30`` 通道名：经 s30→o20 投影写入（无 O20 对应的通道跳过）
    """
    from .semantic_30 import SEMANTIC_30_IDX, s30_joint_index_to_o20

    out: Dict[int, str] = {}
    for key, value in (raw or {}).items():
        jname = str(value)
        if isinstance(key, bool):
            raise ValueError(f"FK joint map 无效键: {key!r}")
        if isinstance(key, int) or (isinstance(key, str) and key.strip().isdigit()):
            out[int(key)] = jname
            continue
        name = str(key).strip()
        if name not in SEMANTIC_30_IDX:
            raise ValueError(f"FK joint map 未知通道/下标: {key!r}")
        o20 = s30_joint_index_to_o20(SEMANTIC_30_IDX[name])
        if o20 < 0:
            continue
        out[o20] = jname
    return out


def _angles_as_o20(angles: List[float], *, fold_pip_into_dip: bool = False) -> List[float]:
    """FK 入口：S30(≥30) 投影为 O20；已是 20 维则原样。

    ``fold_pip_into_dip``：O20 mesh 无独立 PIP 时，把 o20[11..14]（← S30 pip）
    折入对应 dip，使 root2 仍能拉动指尖 FK。
    """
    from .semantic_30 import project_s30_to_o20

    if len(angles) >= 30:
        out = project_s30_to_o20(angles)
    else:
        out = [float(v) for v in angles[:20]]
        while len(out) < 20:
            out.append(0.0)
    if fold_pip_into_dip:
        for pip_i, dip_i in ((11, 16), (12, 17), (13, 18), (14, 19)):
            out[dip_i] = min(255.0, float(out[dip_i]) + float(out[pip_i]))
    return out


def _load_fk_config(hand_joint: str) -> Optional[dict]:
    """从 YAML 配置加载 FK 段。返回 None 如果不存在。"""
    from .hand_config import resolve_hand_config_key

    filename = f"{resolve_hand_config_key(hand_joint)}.yaml"
    # 尝试 ament share
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("hand_gesture_primitives")
        path = os.path.join(share, "config", filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("fk")
    except Exception:
        pass
    # source tree fallback
    path = os.path.join(_CONFIG_DIR, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("fk")
    return None


def _resolve_urdf_path(model: str, side: str, urdf_path: str = "") -> str:
    """查找 URDF 文件。优先使用 urdf_utils (若可导入)，否则本地搜索。"""
    if urdf_path and os.path.isfile(urdf_path):
        return urdf_path
    try:
        from .urdf_utils import resolve_urdf_path

        return resolve_urdf_path(model, side)
    except (ImportError, FileNotFoundError):
        pass
    # 本地 fallback（与 urdf_utils 保持一致：支持版本化文件名）
    try:
        from .urdf_utils import _find_urdf_in_dir, _normalize_model, list_search_dirs

        model_n = _normalize_model(model)
        for base_dir in list_search_dirs():
            hit = _find_urdf_in_dir(os.path.join(base_dir, model_n, side), model_n, side)
            if hit:
                return hit
    except ImportError:
        pass
    return ""


def _frame_id(model: "pin.Model", name: str) -> int:
    """Resolve a URDF link/frame name to a Pinocchio frame id."""
    if hasattr(model, "existFrame") and not model.existFrame(name):
        raise ValueError(f"URDF frame/link 不存在: {name}")
    fid = model.getFrameId(name)
    if fid >= model.nframes:
        raise ValueError(f"URDF frame/link 不存在: {name}")
    return fid


class HandFKSolver:
    """基于 Pinocchio 的 URDF 正运动学求解器。

    将 O20 内部表示 (20-DOF, 0-255) 转为 URDF 关节弧度，
    通过正运动学计算 5 个指尖在基座坐标系下的 xyz 位置。

    用法:
        fk = HandFKSolver("l25", "left")
        tips = fk.compute_fingertips(o20_angles)  # shape [5, 3]
    """

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    def __init__(
        self,
        hand_model: str = "l25",
        hand_side: str = "left",
        urdf_path: str = "",
        fk_config: Optional[dict] = None,
    ):
        """初始化 FK 求解器。

        Args:
            hand_model: URDF 型号 (l25/l20/o20)
            hand_side: left/right
            urdf_path: URDF 文件路径 (空=自动搜索)
            fk_config: FK 配置字典 (从 YAML 加载), None 时使用默认
        """
        _try_import_pinocchio()
        if not _PINOCCHIO_AVAILABLE:
            raise ImportError(
                _PINOCCHIO_UNAVAILABLE_REASON or "pinocchio 未安装，无法使用 FK 求解器"
            )

        self._hand_model = hand_model.lower()
        self._hand_side = hand_side.lower()

        # 加载 FK 配置: 优先参数传入, 其次 YAML, 最后内置默认
        if fk_config is None:
            fk_config = _load_fk_config(self._hand_model)
        if fk_config is None:
            fk_config = _DEFAULT_FK_CONFIGS.get(self._hand_model)
        if fk_config is None:
            raise ValueError(
                f"不支持的 FK 型号: {self._hand_model}, 可用: {list(_DEFAULT_FK_CONFIGS.keys())}"
            )

        # 应用侧别覆盖 (左右手 URDF 关节名/限位可能不同)
        side_overrides = fk_config.get("side_overrides", {}).get(self._hand_side)
        if side_overrides:
            fk_config = dict(fk_config)
            if "base_link" in side_overrides:
                fk_config["base_link"] = side_overrides["base_link"]
            if "fingertip_links" in side_overrides:
                fk_config["fingertip_links"] = side_overrides["fingertip_links"]
            if "invert_joints" in side_overrides:
                fk_config["invert_joints"] = side_overrides["invert_joints"]
            if "tip_offsets" in side_overrides:
                fk_config["tip_offsets"] = side_overrides["tip_offsets"]
            if "o20_to_urdf_joint" in side_overrides:
                merged = dict(fk_config["o20_to_urdf_joint"])
                merged.update({int(k): v for k, v in side_overrides["o20_to_urdf_joint"].items()})
                fk_config["o20_to_urdf_joint"] = merged

        self._base_link: str = fk_config["base_link"]
        self._fingertip_links: List[str] = list(fk_config["fingertip_links"])
        # 关节映射：优先合并 semantic 通道名，再叠 O20 数字键（同槽后者覆盖）
        joint_map: Dict[int, str] = {}
        if fk_config.get("semantic_to_urdf_joint"):
            joint_map.update(_parse_fk_joint_map(fk_config["semantic_to_urdf_joint"]))
        if fk_config.get("o20_to_urdf_joint"):
            joint_map.update(_parse_fk_joint_map(fk_config["o20_to_urdf_joint"]))
        self._joint_map: Dict[int, str] = joint_map
        # 需要反转映射方向的关节 (URDF 正方向与 O20 约定相反)
        self._invert_joints: set = set(fk_config.get("invert_joints", []))
        # 指肚接触面偏移: link 局部坐标系下从 link origin 到实际接触点的偏移
        # 格式: {link_name: [dx, dy, dz]} — 无配置时为零偏移 (即 DIP 关节位置)
        raw_offsets = fk_config.get("tip_offsets", {})
        self._tip_offsets: Dict[str, np.ndarray] = {
            link: np.array(off, dtype=np.float64) for link, off in raw_offsets.items()
        }
        # O30 mesh 无 PIP 折入：独立 index_pip 关节（fold_pip_into_dip 在 o30.yaml 为 false）
        self._fold_pip_into_dip: bool = bool(fk_config.get("fold_pip_into_dip", False))

        # 加载 URDF → Pinocchio model
        urdf_model = fk_config.get("urdf_model", self._hand_model)
        path = _resolve_urdf_path(urdf_model, self._hand_side, urdf_path)
        if not path:
            raise FileNotFoundError(f"找不到 URDF: model={urdf_model}, side={self._hand_side}")

        self._model = pin.buildModelFromUrdf(path)
        self._data = self._model.createData()
        self._q = pin.neutral(self._model)
        self._fk_lock = threading.Lock()

        # 关节名 → (q_index, lower, upper)；仅 1-DoF 可动关节
        self._joint_q_index: Dict[str, int] = {}
        self._joint_limits: Dict[str, Tuple[float, float]] = {}
        for jid in range(1, self._model.njoints):
            joint = self._model.joints[jid]
            if joint.nq != 1:
                continue
            jname = self._model.names[jid]
            q_idx = joint.idx_q
            self._joint_q_index[jname] = q_idx
            self._joint_limits[jname] = (
                float(self._model.lowerPositionLimit[q_idx]),
                float(self._model.upperPositionLimit[q_idx]),
            )

        # 验证映射表关节存在于 URDF actuated joints
        actuated = set(self._joint_q_index.keys())
        self._active_mapping: Dict[int, str] = {}
        for o20_idx, jname in self._joint_map.items():
            if jname in actuated and jname in self._joint_limits:
                self._active_mapping[o20_idx] = jname

        self._base_frame_id = _frame_id(self._model, self._base_link)
        self._tip_frame_ids = [_frame_id(self._model, link) for link in self._fingertip_links]

        self._urdf_path = path

    @property
    def urdf_path(self) -> str:
        return self._urdf_path

    @property
    def base_link(self) -> str:
        return self._base_link

    @property
    def fingertip_links(self) -> List[str]:
        return list(self._fingertip_links)

    @property
    def joint_names(self) -> List[str]:
        """URDF 中被映射的 actuated joint names。"""
        return list(self._active_mapping.values())

    @property
    def available(self) -> bool:
        return True

    @property
    def has_tip_offsets(self) -> bool:
        """是否配置了指肚接触面偏移。"""
        return len(self._tip_offsets) > 0

    def o20_to_radians(self, o20_angles: List[float]) -> Dict[str, float]:
        """语义角度 [0-255] → URDF 关节弧度（接受 O20 或 S30）。"""
        cfg = {}
        for o20_idx, angle in self.o20_to_indexed_radians(o20_angles).items():
            jname = self._active_mapping[o20_idx]
            cfg[jname] = angle
        return cfg

    def semantic_to_radians(self, angles: List[float]) -> Dict[str, float]:
        """``o20_to_radians`` 别名：强调可吃 semantic_30。"""
        return self.o20_to_radians(angles)

    def o20_rad_to_radians(self, o20_rad: List[float]) -> Dict[str, float]:
        """semantic_rad（O20 或 S30 槽，已是 URDF 弧度）→ 关节名字典。

        不再做 255 比例/invert（invert 已在 255→rad 适配层处理）。
        """
        cfg = {}
        for o20_idx, angle in self.o20_rad_to_indexed_radians(o20_rad).items():
            jname = self._active_mapping[o20_idx]
            cfg[jname] = angle
        return cfg

    def o20_to_indexed_radians(self, o20_angles: List[float]) -> Dict[int, float]:
        """语义通道 → URDF 弧度，以 **O20** 通道索引返回（MOCK 兼容）。

        入参可为 20 维 O20 或 ≥30 维 S30（自动投影）。
        """
        o20_angles = _angles_as_o20(
            o20_angles,
            fold_pip_into_dip=bool(getattr(self, "_fold_pip_into_dip", False)),
        )
        cfg: Dict[int, float] = {}
        for o20_idx, jname in self._active_mapping.items():
            val = float(o20_angles[o20_idx]) if o20_idx < len(o20_angles) else 0.0
            val = max(0.0, min(255.0, val))
            lo, hi = self._joint_limits[jname]
            if jname in self._invert_joints:
                cfg[o20_idx] = hi - (val / 255.0) * (hi - lo)
            else:
                cfg[o20_idx] = lo + (val / 255.0) * (hi - lo)
        return cfg

    def o20_rad_to_indexed_radians(self, o20_rad: List[float]) -> Dict[int, float]:
        """semantic_rad → 按 O20 槽索引钳位到 URDF 限位（无 255/invert 再变换）。

        入参可为 20 维 O20 或 ≥30 维 S30（投影为 O20；rad 路径不做 255 pip 折入）。
        """
        o20_rad = _angles_as_o20(o20_rad, fold_pip_into_dip=False)
        cfg: Dict[int, float] = {}
        for o20_idx, jname in self._active_mapping.items():
            val = float(o20_rad[o20_idx]) if o20_idx < len(o20_rad) else 0.0
            lo, hi = self._joint_limits[jname]
            if hi < lo:
                lo, hi = hi, lo
            cfg[o20_idx] = max(lo, min(hi, val))
        return cfg

    def indexed_radians_to_o20(self, radians: Dict[int, float]) -> List[float]:
        """按 O20 通道索引给出的 URDF 弧度 → O20 0~255 语义空间。"""
        out = [0.0] * 20
        for o20_idx, angle in radians.items():
            jname = self._active_mapping.get(int(o20_idx))
            if jname is None or o20_idx >= len(out):
                continue
            lo, hi = self._joint_limits[jname]
            if hi <= lo:
                continue
            value = max(lo, min(hi, float(angle)))
            if jname in self._invert_joints:
                ratio = (hi - value) / (hi - lo)
            else:
                ratio = (value - lo) / (hi - lo)
            out[o20_idx] = max(0.0, min(255.0, ratio * 255.0))
        return out

    def compute_fingertips(self, o20_angles: List[float]) -> np.ndarray:
        """计算 5 指肚接触面位置。

        如配置了 tip_offsets，返回 link origin + 局部偏移旋转到世界系后的位置;
        无偏移时退化为 link origin (DIP 关节位置)。

        Args:
            o20_angles: O20 (20) 或 semantic_30 (≥30)，值域 0-255

        Returns:
            [5, 3] ndarray — 指肚接触面 xyz 坐标 (基座 link 系, 米)
            顺序: thumb, index, middle, ring, pinky
        """
        return self._compute_fingertips_from_cfg(self.o20_to_radians(o20_angles))

    def compute_fingertips_rad(self, o20_rad: List[float]) -> np.ndarray:
        """同 ``compute_fingertips``，输入为 semantic_rad（O20 或 S30，单位 rad）。"""
        return self._compute_fingertips_from_cfg(self.o20_rad_to_radians(o20_rad))

    def _compute_fingertips_from_cfg(self, cfg: Dict[str, float]) -> np.ndarray:
        q = pin.neutral(self._model).copy()
        for jname, angle in cfg.items():
            q_idx = self._joint_q_index.get(jname)
            if q_idx is not None:
                q[q_idx] = angle

        with self._fk_lock:
            data = self._data
            pin.framesForwardKinematics(self._model, data, q)
            oMf_base = data.oMf[self._base_frame_id]

            positions = np.zeros((5, 3), dtype=np.float64)
            for i, (link_name, tip_fid) in enumerate(
                zip(self._fingertip_links, self._tip_frame_ids)
            ):
                T_base_tip = oMf_base.inverse() * data.oMf[tip_fid]
                origin = np.asarray(T_base_tip.translation, dtype=np.float64).reshape(3)
                offset = self._tip_offsets.get(link_name)
                if offset is not None:
                    R = np.asarray(T_base_tip.rotation, dtype=np.float64)
                    positions[i] = origin + R @ offset
                else:
                    positions[i] = origin
        return positions

    def compute_fingertip_dict(self, o20_angles: List[float]) -> Dict[str, np.ndarray]:
        """计算指尖位置，返回字典形式。"""
        positions = self.compute_fingertips(o20_angles)
        return {name: positions[i] for i, name in enumerate(self.FINGER_NAMES)}

    def fingertip_distances(self, o20_angles: List[float]) -> Dict[str, float]:
        """计算常用指对间距离 (米)。"""
        positions = self.compute_fingertips(o20_angles)
        result = {}
        pairs = [
            (0, 1, "thumb_index"),
            (0, 2, "thumb_middle"),
            (0, 3, "thumb_ring"),
            (0, 4, "thumb_pinky"),
            (1, 2, "index_middle"),
        ]
        for i, j, name in pairs:
            result[name] = float(np.linalg.norm(positions[i] - positions[j]))
        return result


# 内置默认 FK 配置 (无 YAML 时的 fallback)
_DEFAULT_FK_CONFIGS: Dict[str, dict] = {
    "l25": {
        "urdf_model": "l25",
        "base_link": "hand_base_link",
        "fingertip_links": [
            "thumb_distal",
            "index_distal",
            "middle_distal",
            "ring_distal",
            "pinky_distal",
        ],
        "tip_offsets": {
            "thumb_distal": [0.003, 0.0, 0.020],
            "index_distal": [0.010, 0.0, 0.015],
            "middle_distal": [0.010, 0.0, 0.015],
            "ring_distal": [0.010, 0.0, 0.015],
            "pinky_distal": [0.010, 0.0, 0.015],
        },
        "o20_to_urdf_joint": {
            0: "thumb_cmc_pitch",
            1: "index_mcp_pitch",
            2: "middle_mcp_pitch",
            3: "ring_mcp_pitch",
            4: "pinky_mcp_pitch",
            5: "thumb_cmc_yaw",
            6: "index_mcp_roll",
            7: "middle_mcp_roll",
            8: "ring_mcp_roll",
            9: "pinky_mcp_roll",
            10: "thumb_cmc_roll",
            15: "thumb_mcp",
            16: "index_pip",
            17: "middle_pip",
            18: "ring_pip",
            19: "pinky_pip",
        },
    },
    "l20": {
        "urdf_model": "l20",
        "base_link": "hand_base_link",
        "fingertip_links": [
            "thumb_distal",
            "index_distal",
            "middle_distal",
            "ring_distal",
            "pinky_distal",
        ],
        "tip_offsets": {
            "thumb_distal": [0.003, 0.0, 0.020],
            "index_distal": [0.010, 0.0, 0.015],
            "middle_distal": [0.010, 0.0, 0.015],
            "ring_distal": [0.010, 0.0, 0.015],
            "pinky_distal": [0.010, 0.0, 0.015],
        },
        "o20_to_urdf_joint": {
            0: "thumb_cmc_roll",
            1: "index_mcp_pitch",
            2: "middle_mcp_pitch",
            3: "ring_mcp_pitch",
            4: "pinky_mcp_pitch",
            5: "thumb_cmc_yaw",
            6: "index_mcp_roll",
            7: "middle_mcp_roll",
            8: "ring_mcp_roll",
            9: "pinky_mcp_roll",
            10: "thumb_cmc_pitch",
            15: "thumb_mcp",
            16: "index_pip",
            17: "middle_pip",
            18: "ring_pip",
            19: "pinky_pip",
        },
    },
    "o20": {
        "urdf_model": "o20",
        "base_link": "hand_link",
        "fingertip_links": [
            "thumb_proximal_Link",
            "index_distal_Link",
            "middle_distal_Link",
            "ring_distal_Link",
            "pinky_distal_Link",
        ],
        "invert_joints": ["thumb_cmc_yaw"],
        "tip_offsets": {
            "thumb_proximal_Link": [-0.003, 0.0, 0.027],
            "index_distal_Link": [0.006, 0.0, 0.030],
            "middle_distal_Link": [0.006, 0.0, 0.030],
            "ring_distal_Link": [0.006, 0.0, 0.030],
            "pinky_distal_Link": [0.006, 0.0, 0.030],
        },
        "o20_to_urdf_joint": {
            0: "thumb_cmc_roll",
            1: "index_mcp_pitch",
            2: "middle_mcp_pitch",
            3: "ring_mcp_pitch",
            4: "pinky_mcp_pitch",
            5: "thumb_cmc_yaw",
            6: "index_mcp_roll",
            7: "middle_mcp_roll",
            8: "ring_mcp_roll",
            9: "pinky_mcp_roll",
            10: "thumb_cmc_pitch",
            15: "thumb_mcp",
            16: "index_dip",
            17: "middle_dip",
            18: "ring_dip",
            19: "pinky_dip",
        },
        "side_overrides": {
            "left": {
                "fingertip_links": [
                    "thumb_distal_Link",
                    "index_distal_Link",
                    "middle_distal_Link",
                    "ring_distal_Link",
                    "pinky_distal_Link",
                ],
                "invert_joints": [],
                "tip_offsets": {
                    "thumb_distal_Link": [-0.003, 0.0, 0.027],
                    "index_distal_Link": [0.006, 0.0, 0.030],
                    "middle_distal_Link": [0.006, 0.0, 0.030],
                    "ring_distal_Link": [0.006, 0.0, 0.030],
                    "pinky_distal_Link": [0.006, 0.0, 0.030],
                },
                "o20_to_urdf_joint": {15: "thumb_dip"},
            },
        },
    },
    "l6": {
        "urdf_model": "l6",
        "base_link": "lh_hand_base_link",
        "invert_joints": ["lh_thumb_cmc_pitch"],
        "fingertip_links": [
            "lh_thumb_distal",
            "lh_index_distal",
            "lh_middle_distal",
            "lh_ring_distal",
            "lh_pinky_distal",
        ],
        "tip_offsets": {
            "lh_thumb_distal": [-0.003, 0.0, 0.013],
            "lh_index_distal": [0.005, 0.0, 0.020],
            "lh_middle_distal": [0.005, 0.0, 0.020],
            "lh_ring_distal": [0.005, 0.0, 0.020],
            "lh_pinky_distal": [0.005, 0.0, 0.020],
        },
        "o20_to_urdf_joint": {
            0: "lh_thumb_cmc_pitch",
            5: "lh_thumb_cmc_roll",
            1: "lh_index_mcp_pitch",
            2: "lh_middle_mcp_pitch",
            3: "lh_ring_mcp_pitch",
            4: "lh_pinky_mcp_pitch",
        },
        "side_overrides": {
            "right": {
                "base_link": "rh_hand_base_link",
                "invert_joints": ["rh_thumb_cmc_pitch"],
                "fingertip_links": [
                    "rh_thumb_distal",
                    "rh_index_distal",
                    "rh_middle_distal",
                    "rh_ring_distal",
                    "rh_pinky_distal",
                ],
                "tip_offsets": {
                    "rh_thumb_distal": [-0.003, 0.0, 0.013],
                    "rh_index_distal": [0.005, 0.0, 0.020],
                    "rh_middle_distal": [0.005, 0.0, 0.020],
                    "rh_ring_distal": [0.005, 0.0, 0.020],
                    "rh_pinky_distal": [0.005, 0.0, 0.020],
                },
                "o20_to_urdf_joint": {
                    0: "rh_thumb_cmc_pitch",
                    5: "rh_thumb_cmc_roll",
                    1: "rh_index_mcp_pitch",
                    2: "rh_middle_mcp_pitch",
                    3: "rh_ring_mcp_pitch",
                    4: "rh_pinky_mcp_pitch",
                },
            },
        },
    },
}

# hand_joint 参数 → FK 使用的配置名
_HAND_JOINT_TO_FK_MODEL = {
    "L25": "l25",
    "L20": "l20",
    "O20": "o20",
    "O6": "o6",
    "L6": "l6",
    "G20": "l25",  # 外观对齐 L20/L25
}


def create_fk_solver(
    hand_joint: str,
    hand_side: str,
    urdf_path: str = "",
    logger=None,
) -> Optional[HandFKSolver]:
    """安全创建 FK 求解器，失败时返回 None。

    优先从 YAML config 加载 FK 配置，无 YAML 时使用内置默认。
    """

    def _log_warn(msg: str) -> None:
        if logger is None:
            return
        # std logging 用 warning；rclpy Logger 常用 warn
        fn = getattr(logger, "warning", None) or getattr(logger, "warn", None)
        if fn is not None:
            fn(msg)

    _try_import_pinocchio()
    if not _PINOCCHIO_AVAILABLE:
        _log_warn(_PINOCCHIO_UNAVAILABLE_REASON or "pinocchio 未安装, FK 求解器不可用")
        return None

    # 尝试从 YAML 加载 FK 配置
    fk_config = _load_fk_config(hand_joint)

    # 无 YAML FK 配置: 使用 hand_joint → model 映射 + 内置默认
    if fk_config is None:
        model = _HAND_JOINT_TO_FK_MODEL.get(hand_joint.upper())
        if model is None:
            _log_warn(
                f"hand_joint={hand_joint!r} 无 FK 配置, "
                f"可用: {list(_HAND_JOINT_TO_FK_MODEL.keys())}"
            )
            return None
        fk_config = _DEFAULT_FK_CONFIGS.get(model)

    try:
        solver = HandFKSolver(
            fk_config.get("urdf_model", hand_joint.lower()),
            hand_side,
            urdf_path,
            fk_config=fk_config,
        )
        if logger:
            logger.info(
                f"FK 求解器初始化成功: hand_joint={hand_joint}, "
                f"model={solver._hand_model}, side={hand_side}, "
                f"base_link={solver.base_link}, "
                f"urdf={solver.urdf_path}, "
                f"actuated_joints={len(solver._active_mapping)}"
            )
        return solver
    except (FileNotFoundError, ImportError, ValueError, Exception) as e:
        _log_warn(f"FK 求解器初始化失败 ({e}), 指尖位置不可用")
        return None
