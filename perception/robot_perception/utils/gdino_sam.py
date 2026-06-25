"""Grounding DINO detection + SAM2 segmentation."""
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.ops import box_convert

from robot_perception.utils.paths import ensure_gsam2_path, model_repo_hint

ensure_gsam2_path()

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
try:
    from groundingdino.util.inference import load_model, predict  # noqa: E402
    import groundingdino.datasets.transforms as GDT  # noqa: E402
except ModuleNotFoundError as exc:
    raise ImportError(
        'Cannot import groundingdino — third-party model repo not on PYTHONPATH. '
        f'{model_repo_hint()}. '
        'Fix: export ROBOT_PERCEPTION_DIR=/path/to/third_party/robot_perception '
        '(directory must contain Grounded-SAM-2/, Cutie/).'
    ) from exc

_GDINO_TRANSFORM = GDT.Compose([
    GDT.RandomResize([800], max_size=1333),
    GDT.ToTensor(),
    GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_sam2_predictor(sam2_config, sam2_checkpoint, device):
    """Load SAM2 image predictor only (no Grounding DINO)."""
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_model = sam2_model.half()
    return SAM2ImagePredictor(sam2_model)


def load_gdino_sam_models(gdino_config, gdino_checkpoint, sam2_config, sam2_checkpoint, device):
    """Load Grounding DINO and SAM2 models."""
    grounding_model = load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=device,
    )
    sam2_predictor = load_sam2_predictor(sam2_config, sam2_checkpoint, device)
    return grounding_model, sam2_predictor


@torch.no_grad()
def gdino_detect(grounding_model, image_rgb, caption, box_threshold, text_threshold, device):
    """Run Grounding DINO on one image.

    Returns list of dicts: phrase, score, box_xyxy (pixel coords).
    """
    h, w = image_rgb.shape[:2]
    pil_image = PILImage.fromarray(image_rgb)
    image_tensor, _ = _GDINO_TRANSFORM(pil_image, None)

    with torch.no_grad():
        boxes, confidences, phrases = predict(
            model=grounding_model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    if len(boxes) == 0:
        return []

    boxes_abs = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt='cxcywh', out_fmt='xyxy').cpu().numpy()
    scores = confidences.cpu().numpy()

    detections = []
    for i in range(len(boxes_xyxy)):
        detections.append({
            'phrase': phrases[i],
            'score': float(scores[i]),
            'box_xyxy': boxes_xyxy[i].astype(np.float32),
        })
    return detections


def expand_bbox_xyxy(bbox_xyxy, img_h, img_w, ratio=2.0):
    """Expand xyxy bbox around center, clamped to image bounds."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1, 8.0) * ratio
    bh = max(y2 - y1, 8.0) * ratio
    nx1 = max(0.0, cx - 0.5 * bw)
    ny1 = max(0.0, cy - 0.5 * bh)
    nx2 = min(float(img_w - 1), cx + 0.5 * bw)
    ny2 = min(float(img_h - 1), cy + 0.5 * bh)
    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        return 0, 0, img_w - 1, img_h - 1
    return int(nx1), int(ny1), int(nx2), int(ny2)


def gdino_detect_crop(grounding_model, image_rgb, x1, y1, x2, y2, caption,
                      box_threshold, text_threshold, device):
    """Run GDINO on a crop and map boxes back to full-image coordinates."""
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    detections = gdino_detect(
        grounding_model, crop, caption, box_threshold, text_threshold, device)
    offset = np.array([x1, y1, x1, y1], dtype=np.float32)
    for det in detections:
        det['box_xyxy'] = det['box_xyxy'] + offset
    return detections


def sam2_segment_boxes(sam2_predictor, image_rgb, boxes_xyxy, return_scores=False):
    """Segment each detection box with SAM2.

    Returns list of bool masks aligned with boxes_xyxy.
    If return_scores=True, returns (masks, scores) where scores is a list of
    SAM2 IoU prediction confidence per mask.
    """
    if len(boxes_xyxy) == 0:
        return ([], []) if return_scores else []

    with torch.autocast('cuda', dtype=torch.float16):
        sam2_predictor.set_image(image_rgb)
        masks, iou_preds, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if hasattr(iou_preds, 'cpu'):
        iou_preds = iou_preds.cpu().numpy()
    scores = iou_preds.flatten().tolist() if iou_preds is not None else [1.0] * len(boxes_xyxy)
    mask_list = [masks[i].astype(bool) for i in range(len(boxes_xyxy))]
    if return_scores:
        return mask_list, scores
    return mask_list


def sam2_segment_box(sam2_predictor, image_rgb, box_xyxy):
    """Segment a single box; returns bool mask HxW."""
    masks = sam2_segment_boxes(sam2_predictor, image_rgb, np.asarray([box_xyxy]))
    return masks[0] if masks else None
