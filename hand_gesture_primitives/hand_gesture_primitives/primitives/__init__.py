"""预定义手势原语注册表。"""

from .open_hand import OpenHand
from .pinch import Pinch
from .fist import Fist
from .point import Point
from .init_hand import InitHand
from .relax_grip import RelaxGrip
from .ok_sign import OkSign
from .v_sign import VSign
from .release import Release
from .index_ring_by_vision import IndexRingByVision
from .large_wrap_by_vision import LargeWrapByVision
from .thumb_adduction_grip import ThumbAdductionGrip
from .index_middle_adduction_grip import IndexMiddleAdductionGrip
from .middle_ring_by_vision import MiddleRingByVision
from .middle_ring import MiddleRing
from .ring_by_vision import RingByVision
from .ring import Ring
from .small_warp_by_vision import SmallWarpByVision
from .no_index_warp_by_vision import NoIndexWarpByVision
from .hook_by_vision import HookByVision
from .index_pinch_by_vision import IndexPinchByVision
from .index_pinch import IndexPinch
from .middle_pinch_by_vision import MiddlePinchByVision
from .middle_pinch import MiddlePinch
from .tripod_by_vision import TripodByVision
from .tripod import Tripod
from .palmar_by_vision import PalmarByVision
from .parallel_extension_by_vision import ParallelExtensionByVision
from .parallel_extension import ParallelExtension
from .disk_by_vision import DiskByVision

PRIMITIVE_REGISTRY = {
    "open": OpenHand,
    "pinch": Pinch,
    "fist": Fist,
    "point": Point,
    "init": InitHand,
    "relax_grip": RelaxGrip,
    "ok_sign": OkSign,
    "v_sign": VSign,
    "release": Release,
    "index_ring_by_vision": IndexRingByVision,
    "large_wrap_by_vision": LargeWrapByVision,
    "thumb_adduction_grip": ThumbAdductionGrip,
    "index_middle_adduction_grip": IndexMiddleAdductionGrip,
    "middle_ring_by_vision": MiddleRingByVision,
    "middle_ring": MiddleRing,
    "ring_by_vision": RingByVision,
    "ring": Ring,
    "small_warp_by_vision": SmallWarpByVision,
    "no_index_warp_by_vision": NoIndexWarpByVision,
    "hook_by_vision": HookByVision,
    "index_pinch_by_vision": IndexPinchByVision,
    "index_pinch": IndexPinch,
    "middle_pinch_by_vision": MiddlePinchByVision,
    "middle_pinch": MiddlePinch,
    "tripod_by_vision": TripodByVision,
    "tripod": Tripod,
    "palmar_by_vision": PalmarByVision,
    "parallel_extension_by_vision": ParallelExtensionByVision,
    "parallel_extension": ParallelExtension,
    "disk_by_vision": DiskByVision,
}
