# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Shared model catalog for editors, teleoperation and execution guards."""

from functools import lru_cache

LABELS = {
    "init": "初始姿态",
    "open": "张开",
    "release": "释放",
    "relax_grip": "放松握持",
    "fist": "握拳",
    "pinch": "捏合",
    "point": "指向",
    "ok_sign": "OK 手势",
    "v_sign": "V 手势",
    "grasp_ready": "准备抓取",
    "hammer_grip": "握锤",
    "large_wrap": "包络抓取",
    "tripod": "三指捏合",
    "tripod_by_force": "三指力控捏合",
    "ring": "环形夹持",
    "middle_ring": "中指夹持",
    "thumb_adduction_grip": "拇指内收夹持",
    "index_middle_adduction_grip": "食中指内收夹持",
    "index_pinch": "食指捏合",
    "middle_pinch": "中指捏合",
    "parallel_extension": "平行伸展",
    "index_flick": "食指弹动",
    "index_middle_flick": "食中指弹动",
    "named_pose": "自定义姿态模板",
}


@lru_cache(maxsize=32)
def profile(model):
    from hand_gesture_primitives.hand_profile import load_hand_profile

    return load_hand_profile(model)


def builtin_catalog(model):
    from hand_gesture_primitives.primitive_catalog import PRIMITIVE_CATALOG
    from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY
    from hand_gesture_primitives.workbench_motion import hold_reason

    prof = profile(model)
    out = []
    for name in dict.fromkeys([*PRIMITIVE_CATALOG, *PRIMITIVE_REGISTRY]):
        entry = PRIMITIVE_CATALOG.get(name)
        category = getattr(entry, "category", "gesture")
        label = LABELS.get(name, name.replace("_", " "))
        if name.endswith("_by_vision"):
            label = (
                LABELS.get(
                    name.removesuffix("_by_vision"),
                    name.removesuffix("_by_vision").replace("_", " "),
                )
                + " · 视觉"
            )
        supported = prof.is_primitive_supported(name)
        out.append(
            {
                "name": name,
                "label": label,
                "supported": supported,
                "category": category,
                "group": "custom"
                if name == "named_pose"
                else "perception"
                if category in {"vision", "force"}
                else "default",
                "requires": sorted(getattr(entry, "requires", [])),
                "template": name == "named_pose",
                "supports_hold": not bool(hold_reason(model, name)),
                "hold_reason": hold_reason(model, name),
                "reason": "" if supported else prof.reject_user_message(name),
            }
        )
    return out


class PrimitiveMenu:
    """Ordered list of hand primitives selectable from buttons, shared by teleops and the UI.

    ``PRIMITIVE_ORDER`` is the stable id space (index = primitive_id in the action dict) and
    comes from hand_gesture_primitives' catalog; the per-hand *menu* is the subset the profile
    supports, in the same relative order.
    """

    DEFAULT_MENU = (
        "open",
        "init",
        "fist",
        "pinch",
        "point",
        "ok_sign",
        "v_sign",
        "relax_grip",
        "release",
        "grasp_ready",
    )

    def __init__(self, model: str | None = None, menu: list[str] | None = None) -> None:
        self.order: list[str] = self._catalog_order()
        wanted = list(menu) if menu else list(self.DEFAULT_MENU)
        if model is not None:
            try:
                from hand_gesture_primitives.hand_profile import load_hand_profile

                supported = set(load_hand_profile(model).supported_primitives())
                wanted = [p for p in wanted if p in supported]
            except Exception:  # hand package unavailable -> keep defaults
                pass
        self.menu: list[str] = [p for p in wanted if p in self.order]
        self._cursor = 0

    def configure(self, names: list[str], custom_names: list[str]) -> None:
        """Append IDs, never recycle them when a pose is removed during a session."""
        selected = self.current
        for name in sorted(custom_names):
            if name not in self.order:
                self.order.append(name)
        allowed = set(self._catalog_order()) - {"named_pose"} | set(custom_names)
        self.menu = list(dict.fromkeys(n for n in names if n in allowed))
        self._cursor = self.menu.index(selected) if selected in self.menu else 0

    @staticmethod
    def _catalog_order() -> list[str]:
        try:
            from hand_gesture_primitives.primitive_catalog import PRIMITIVE_CATALOG

            return list(PRIMITIVE_CATALOG.keys())
        except Exception:
            return list(PrimitiveMenu.DEFAULT_MENU)

    def id_of(self, name: str) -> int:
        return self.order.index(name) if name in self.order else -1

    def name_of(self, pid: int) -> str | None:
        return self.order[pid] if 0 <= pid < len(self.order) else None

    @property
    def current(self) -> str | None:
        return self.menu[self._cursor] if self.menu else None

    def step(self, delta: int) -> str | None:
        if self.menu:
            self._cursor = (self._cursor + delta) % len(self.menu)
        return self.current

    def select_index(self, idx: int) -> str | None:
        if 0 <= idx < len(self.menu):
            self._cursor = idx
        return self.current
