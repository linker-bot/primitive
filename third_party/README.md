# third_party/

External algorithm repositories used by the perception pipeline.

## Contents

- `robot_perception/` — Perception algorithm stack:
  - **Grounded-SAM-2** — Grounding DINO + SAM 2 detection/segmentation
  - **Cutie** — Video object segmentation / mask tracking (vendored, see `Cutie/LICENSE`)
  - **Pixal3D** — 3D mesh reconstruction (optional, **git submodule**)
  - **MiniCPM-V** — VLM reference docs (optional, **git submodule**)

> **FoundationStereo** (stereo depth via TensorRT) 已从 third_party 移除以精简开源发布。
> 如需使用，设置 `use_stereo_depth:=true` 并将 FoundationStereo 放回此目录或配置 `PERCEPTION_WEIGHTS_DIR`。

Clone 后初始化子模块：`git submodule update --init third_party/robot_perception/MiniCPM-V third_party/robot_perception/Pixal3D`

## Override

To use a different location (e.g., during development), set:

```bash
export ROBOT_PERCEPTION_DIR=/path/to/robot_perception
export PERCEPTION_WEIGHTS_DIR=/path/to/weights   # optional
```

If unset, the code defaults to this directory and `data/weights/` (or legacy `${PERCEPTION_WEIGHTS_DIR:-data/weights}/`).
