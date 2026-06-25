# Cutie (vendored)

This directory contains a **minimal vendored copy** of the [Cutie](https://github.com/hkchengrex/Cutie) video object segmentation library, used by `robot_perception` for temporal mask tracking (`use_cutie_tracking`).

Only the `cutie/` Python package is included (no GUI, demos, or training scripts from the upstream repository).

## Upstream

| Item | Value |
|------|-------|
| Project | [Putting the Object Back into Video Object Segmentation](https://hkchengrex.github.io/Cutie) |
| Authors | Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, Alexander Schwing |
| Repository | https://github.com/hkchengrex/Cutie |
| Reference commit | `ec5cdd4` (2024-11-08, upstream `main`) |
| License | MIT — see [LICENSE](./LICENSE) |

Paper: [arXiv:2310.12982](https://arxiv.org/abs/2310.12982) (CVPR 2024 Highlight)

## Local modifications

The following files differ from upstream for integration with `hand_gesture_primitives` / `robot_perception`:

| File | Change |
|------|--------|
| `cutie/inference/inference_core.py` | Add `segment_threshold` to `output_prob_to_mask()` (used by `cutie_track.py`) |
| `cutie/utils/download_models.py` | Weights path: `PERCEPTION_WEIGHTS_DIR/cutie` or `data/weights/cutie` |
| `cutie/utils/get_default_model.py` | Load weights on CPU first, then move to CUDA |

Model weights (`cutie-base-mega.pth`) are **not** bundled; see [third_party/DEPENDENCIES.md](../../DEPENDENCIES.md).

## Usage in this project

Imported via `perception/robot_perception/utils/cutie_track.py` when `detection_bbox` runs with `use_cutie_tracking:=true`.

For the full upstream project (training, interactive demo, GUI), clone:

```bash
git clone https://github.com/hkchengrex/Cutie.git
```
