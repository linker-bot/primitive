"""示教面板：产品手型（姿态库/执行）与 UI 布局手型（滑条）转换。

整机 G20（l25_urdf）机械上与 L25 同构（16 主动 + 5 被动尖），
示教 UI 用 L25 native25（含 mid、tip 耦合），落盘/回放仍用产品 G20 的 16 维 hardware。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..hand_config import HandConfig, resolve_hand_config_key
from ..joint_state_mapping import resolve_g20_l25_urdf


def panel_layout_hand_key(
    product_key: str,
    *,
    g20_joint_order: str = "",
    robot_description: str = "",
    hand_side: str = "left",
) -> str:
    """UI / tip 耦合用的手型键；整机 G20 → l25。"""
    key = resolve_hand_config_key(product_key)
    if key == "g20" and resolve_g20_l25_urdf(g20_joint_order, robot_description, hand_side):
        return "l25"
    return key


def ui_to_product_hardware(
    ui_positions: Sequence[float],
    *,
    product_key: str,
    layout_key: str,
) -> List[float]:
    """示教滑条向量 → 产品 hardware（姿态库 / NamedPose 量纲）。"""
    product = resolve_hand_config_key(product_key)
    layout = resolve_hand_config_key(layout_key)
    ui = [float(v) for v in ui_positions]
    if product == layout:
        return ui
    layout_cfg = HandConfig(layout)
    product_cfg = HandConfig(product)
    if len(ui) != layout_cfg.num_joints:
        raise ValueError(f"UI 关节数 {len(ui)} 与布局 {layout}({layout_cfg.num_joints}) 不符")
    if layout_cfg.has_from_semantic and product_cfg.has_from_semantic:
        return product_cfg.from_semantic(layout_cfg.to_semantic(ui))
    semantic = layout_cfg.from_hardware(ui)
    return product_cfg.to_hardware(semantic)


def product_hardware_to_ui(
    product_hw: Sequence[float],
    *,
    product_key: str,
    layout_key: str,
) -> List[float]:
    """产品 hardware → 示教滑条向量。"""
    product = resolve_hand_config_key(product_key)
    layout = resolve_hand_config_key(layout_key)
    hw = [float(v) for v in product_hw]
    if product == layout:
        return hw
    layout_cfg = HandConfig(layout)
    product_cfg = HandConfig(product)
    if len(hw) != product_cfg.num_joints:
        raise ValueError(f"产品关节数 {len(hw)} 与 {product}({product_cfg.num_joints}) 不符")
    if layout_cfg.has_from_semantic and product_cfg.has_from_semantic:
        return layout_cfg.from_semantic(product_cfg.to_semantic(hw))
    semantic = product_cfg.from_hardware(hw)
    return layout_cfg.to_hardware(semantic)


def maybe_ui_to_product(
    ui_positions: Optional[Sequence[float]],
    *,
    product_key: str,
    layout_key: str,
) -> Optional[List[float]]:
    if ui_positions is None:
        return None
    return ui_to_product_hardware(ui_positions, product_key=product_key, layout_key=layout_key)
