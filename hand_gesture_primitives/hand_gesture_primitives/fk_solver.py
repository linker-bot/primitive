"""正运动学求解器 — 基于 URDF 计算手指尖末端位置。"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import yourdfpy
    _YOURDFPY_AVAILABLE = True
except ImportError:
    _YOURDFPY_AVAILABLE = False


# 配置目录 (source tree)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _load_fk_config(hand_joint: str) -> Optional[dict]:
    """从 YAML 配置加载 FK 段。返回 None 如果不存在。"""
    filename = f"{hand_joint.lower()}.yaml"
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
    # 本地 fallback（与 urdf_utils._default_urdf_search_dirs 保持一致）
    try:
        from .urdf_utils import list_search_dirs
        search_dirs = list_search_dirs()
    except ImportError:
        search_dirs = [
            os.path.expanduser("~/ros_ws/src/linkerhand-urdf"),
            os.path.expanduser("~/gitea_ws/src/linkerhand-urdf"),
            os.path.expanduser("~/ros_ws/src/linkerhand-urdf"),
            "/opt/linkerhand/urdf",
        ]
    filename = f"linkerhand_{model}_{side}.urdf"
    for base_dir in search_dirs:
        path = os.path.join(base_dir, model, side, filename)
        if os.path.isfile(path):
            return path
    # ament_index fallback
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_dir = get_package_share_directory("linkerhand_description")
        path = os.path.join(pkg_dir, "urdf", model, side, filename)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    return ""


class HandFKSolver:
    """基于 URDF 正运动学求解器。

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
        if not _YOURDFPY_AVAILABLE:
            raise ImportError("yourdfpy 未安装，无法使用 FK 求解器")

        self._hand_model = hand_model.lower()
        self._hand_side = hand_side.lower()

        # 加载 FK 配置: 优先参数传入, 其次 YAML, 最后内置默认
        if fk_config is None:
            fk_config = _load_fk_config(self._hand_model)
        if fk_config is None:
            fk_config = _DEFAULT_FK_CONFIGS.get(self._hand_model)
        if fk_config is None:
            raise ValueError(
                f"不支持的 FK 型号: {self._hand_model}, "
                f"可用: {list(_DEFAULT_FK_CONFIGS.keys())}"
            )

        # 应用侧别覆盖 (左右手 URDF 关节名/限位可能不同)
        side_overrides = fk_config.get("side_overrides", {}).get(self._hand_side)
        if side_overrides:
            fk_config = dict(fk_config)
            if "fingertip_links" in side_overrides:
                fk_config["fingertip_links"] = side_overrides["fingertip_links"]
            if "invert_joints" in side_overrides:
                fk_config["invert_joints"] = side_overrides["invert_joints"]
            if "tip_offsets" in side_overrides:
                fk_config["tip_offsets"] = side_overrides["tip_offsets"]
            if "o20_to_urdf_joint" in side_overrides:
                merged = dict(fk_config["o20_to_urdf_joint"])
                merged.update(
                    {int(k): v for k, v in side_overrides["o20_to_urdf_joint"].items()}
                )
                fk_config["o20_to_urdf_joint"] = merged

        self._base_link: str = fk_config["base_link"]
        self._fingertip_links: List[str] = list(fk_config["fingertip_links"])
        # o20_to_urdf_joint: YAML 中 key 为 int, value 为 str
        raw_map = fk_config["o20_to_urdf_joint"]
        self._joint_map: Dict[int, str] = {int(k): v for k, v in raw_map.items()}
        # 需要反转映射方向的关节 (URDF 正方向与 O20 约定相反)
        self._invert_joints: set = set(fk_config.get("invert_joints", []))
        # 指肚接触面偏移: link 局部坐标系下从 link origin 到实际接触点的偏移
        # 格式: {link_name: [dx, dy, dz]} — 无配置时为零偏移 (即 DIP 关节位置)
        raw_offsets = fk_config.get("tip_offsets", {})
        self._tip_offsets: Dict[str, np.ndarray] = {
            link: np.array(off, dtype=np.float64)
            for link, off in raw_offsets.items()
        }

        # 加载 URDF
        urdf_model = fk_config.get("urdf_model", self._hand_model)
        path = _resolve_urdf_path(urdf_model, self._hand_side, urdf_path)
        if not path:
            raise FileNotFoundError(
                f"找不到 URDF: model={urdf_model}, side={self._hand_side}"
            )

        self._urdf = yourdfpy.URDF.load(path)

        # 缓存关节限位 {joint_name: (lower, upper)}
        self._joint_limits: Dict[str, Tuple[float, float]] = {}
        for j in self._urdf.robot.joints:
            if j.limit is not None:
                self._joint_limits[j.name] = (j.limit.lower, j.limit.upper)

        # 验证映射表关节存在于 URDF actuated joints
        actuated = set(self._urdf.actuated_joint_names)
        self._active_mapping: Dict[int, str] = {}
        for o20_idx, jname in self._joint_map.items():
            if jname in actuated and jname in self._joint_limits:
                self._active_mapping[o20_idx] = jname

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
        """O20 内部角度 [0-255] → URDF 关节弧度配置 (公开接口，供可视化使用)。"""
        cfg = {}
        for o20_idx, jname in self._active_mapping.items():
            val = float(o20_angles[o20_idx])
            val = max(0.0, min(255.0, val))
            lo, hi = self._joint_limits[jname]
            if jname in self._invert_joints:
                cfg[jname] = hi - (val / 255.0) * (hi - lo)
            else:
                cfg[jname] = lo + (val / 255.0) * (hi - lo)
        return cfg

    def compute_fingertips(self, o20_angles: List[float]) -> np.ndarray:
        """计算 5 指肚接触面位置。

        如配置了 tip_offsets，返回 link origin + 局部偏移旋转到世界系后的位置;
        无偏移时退化为 link origin (DIP 关节位置)。

        Args:
            o20_angles: O20 内部表示 (20 floats, 0-255)

        Returns:
            [5, 3] ndarray — 指肚接触面 xyz 坐标 (基座 link 系, 米)
            顺序: thumb, index, middle, ring, pinky
        """
        cfg = self.o20_to_radians(o20_angles)
        self._urdf.update_cfg(cfg)

        positions = np.zeros((5, 3), dtype=np.float64)
        for i, link_name in enumerate(self._fingertip_links):
            T = self._urdf.get_transform(link_name, self._base_link)
            origin = T[:3, 3]
            offset = self._tip_offsets.get(link_name)
            if offset is not None:
                R = T[:3, :3]
                positions[i] = origin + R @ offset
            else:
                positions[i] = origin
        return positions

    def compute_fingertip_dict(
        self, o20_angles: List[float]
    ) -> Dict[str, np.ndarray]:
        """计算指尖位置，返回字典形式。"""
        positions = self.compute_fingertips(o20_angles)
        return {
            name: positions[i] for i, name in enumerate(self.FINGER_NAMES)
        }

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
            "thumb_distal", "index_distal", "middle_distal",
            "ring_distal", "pinky_distal",
        ],
        "tip_offsets": {
            "thumb_distal": [0.003, 0.0, 0.020],
            "index_distal": [0.010, 0.0, 0.015],
            "middle_distal": [0.010, 0.0, 0.015],
            "ring_distal": [0.010, 0.0, 0.015],
            "pinky_distal": [0.010, 0.0, 0.015],
        },
        "o20_to_urdf_joint": {
            0: "thumb_cmc_roll", 1: "index_mcp_pitch",
            2: "middle_mcp_pitch", 3: "ring_mcp_pitch",
            4: "pinky_mcp_pitch", 5: "thumb_cmc_yaw",
            6: "index_mcp_roll", 7: "middle_mcp_roll",
            8: "ring_mcp_roll", 9: "pinky_mcp_roll",
            10: "thumb_cmc_pitch", 15: "thumb_mcp",
            16: "index_pip", 17: "middle_pip",
            18: "ring_pip", 19: "pinky_pip",
        },
    },
    "l20": {
        "urdf_model": "l20",
        "base_link": "hand_base_link",
        "fingertip_links": [
            "thumb_distal", "index_distal", "middle_distal",
            "ring_distal", "pinky_distal",
        ],
        "tip_offsets": {
            "thumb_distal": [0.003, 0.0, 0.020],
            "index_distal": [0.010, 0.0, 0.015],
            "middle_distal": [0.010, 0.0, 0.015],
            "ring_distal": [0.010, 0.0, 0.015],
            "pinky_distal": [0.010, 0.0, 0.015],
        },
        "o20_to_urdf_joint": {
            0: "thumb_cmc_roll", 1: "index_mcp_pitch",
            2: "middle_mcp_pitch", 3: "ring_mcp_pitch",
            4: "pinky_mcp_pitch", 5: "thumb_cmc_yaw",
            6: "index_mcp_roll", 7: "middle_mcp_roll",
            8: "ring_mcp_roll", 9: "pinky_mcp_roll",
            10: "thumb_cmc_pitch", 15: "thumb_mcp",
            16: "index_pip", 17: "middle_pip",
            18: "ring_pip", 19: "pinky_pip",
        },
    },
    "o20": {
        "urdf_model": "o20",
        "base_link": "hand_link",
        "fingertip_links": [
            "thumb_proximal_Link", "index_distal_Link",
            "middle_distal_Link", "ring_distal_Link",
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
            0: "thumb_cmc_roll", 1: "index_mcp_pitch",
            2: "middle_mcp_pitch", 3: "ring_mcp_pitch",
            4: "pinky_mcp_pitch", 5: "thumb_cmc_yaw",
            6: "index_mcp_roll", 7: "middle_mcp_roll",
            8: "ring_mcp_roll", 9: "pinky_mcp_roll",
            10: "thumb_cmc_pitch", 15: "thumb_mcp",
            16: "index_dip", 17: "middle_dip",
            18: "ring_dip", 19: "pinky_dip",
        },
        "side_overrides": {
            "left": {
                "fingertip_links": [
                    "thumb_distal_Link", "index_distal_Link",
                    "middle_distal_Link", "ring_distal_Link",
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
}

# hand_joint 参数 → FK 使用的配置名
_HAND_JOINT_TO_FK_MODEL = {
    "L25": "l25",
    "L20": "l20",
    "O20": "o20",
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
    if not _YOURDFPY_AVAILABLE:
        if logger:
            logger.warn("yourdfpy 未安装, FK 求解器不可用")
        return None

    # 尝试从 YAML 加载 FK 配置
    fk_config = _load_fk_config(hand_joint)

    # 无 YAML FK 配置: 使用 hand_joint → model 映射 + 内置默认
    if fk_config is None:
        model = _HAND_JOINT_TO_FK_MODEL.get(hand_joint.upper())
        if model is None:
            if logger:
                logger.warn(
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
        if logger:
            logger.warn(f"FK 求解器初始化失败 ({e}), 指尖位置不可用")
        return None
