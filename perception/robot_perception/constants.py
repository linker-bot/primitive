LABEL_PROMPTS_MAP = {
    "6_12_green_lego": "big green lego",
    "2_10_red_bridge_lego": "red lego",
    "2_4_green_bridge_lego": "small green lego",
    "2_4_blue_lego": "blue lego",
    "2_10_orange_lego": "big yellow lego",
    "2_2_orange_head_lego": "small yellow lego",
    "lego_figure": "lego figure",
}

# Default GDINO/VLM prompts for production-line / workbench (narrow caption).
# Covers: robotic hand, joint motors, sheet-metal parts, and hand tools on table.
DEFAULT_INDUSTRY_SCENE_PROMPTS = [
    # robotic hand / gripper (loose parts on table)
    'robotic hand.',
    'robot gripper.',
    'dexterous hand.',
    # joint / servo motors
    'joint motor.',
    'servo motor.',
    'electric motor.',
    # sheet metal / stamped parts
    'sheet metal part.',
    'sheet metal bracket.',
    'stamped metal part.',
    'metal sheet.',
    # hand tools & fasteners
    'screwdriver.',
    'screw.',
    'pliers.',
    'wrench.',
    'hammer.',
    'hex key.',
    # generic workbench parts
    'metal workpiece.',
    'industrial part.',
    'fixture.',
]

# Default GDINO prompts for desk Lego detection (narrow caption → fewer false positives).
DEFAULT_LEGO_SCENE_PROMPTS = [
    'blue lego.',
    'red lego.',
    'green lego.',
    'yellow lego.',
    'orange lego.',
    'small yellow lego.',
    'big green lego.',
    'lego figure.',
    'lego brick.',
    'lego block.',
]

# Default GDINO caption classes for open-scene mode (prompt-driven, not closed-set YOLO).
# GDINO only detects categories present in the combined caption.
# 只保留桌面上可抓取物体类别，排除人/机器人/背景干扰。
DEFAULT_OPEN_SCENE_PROMPTS = [
    'bottle.',
    'cup.',
    'mug.',
    'bowl.',
    'plate.',
    'box.',
    'container.',
    'bag.',
    'can.',
    'book.',
    'pen.',
    'scissors.',
    'tool.',
    'phone.',
    'remote.',
    'block.',
    'toy.',
    'lego.',
    'figure.',
    'fruit.',
    'apple.',
    'orange.',
    'banana.',
    'object.',
]
