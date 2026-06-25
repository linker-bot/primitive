import os
import cv2
import json
import copy
import torch
import shutil
import argparse
import tempfile
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded SAM2 local demo")
    parser.add_argument("--text-prompt", type=str, default="car. tire.",
                        help="text prompt for grounding (lowercased, dot-separated)")
    parser.add_argument("--img-path", type=str, default="notebooks/images/truck.jpg",
                        help="input image path or folder of images")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="output directory for binary masks (255=object, 0=bg). "
                        "If --img-path is a folder, defaults to sibling 'masks' dir.")
    parser.add_argument("--sam2-checkpoint", type=str,
                        default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--grounding-dino-config", type=str,
                        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding-dino-checkpoint", type=str,
                        default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default="outputs/grounded_sam2_local_demo")
    parser.add_argument("--device", type=str, default=None,
                        help="device (default: cuda if available)")
    parser.add_argument("--dump-json", action="store_true", default=True,
                        help="dump results as json")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--multimask-output", action="store_true", default=False)
    # Tracking mode arguments
    parser.add_argument("--tracking", action="store_true", default=False,
                        help="enable tracking mode for stable inter-frame segmentation")
    parser.add_argument("--redetect-interval", type=int, default=0,
                        help="re-run detection every N frames (0 = only detect on first frame)")
    return parser.parse_args()


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_single_image(img_path, sam2_predictor, grounding_model, args, device):
    """处理单张图片，返回合并后的二值 mask (H, W), uint8, 255=物体 0=背景"""
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    h, w, _ = image_source.shape

    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8), [], [], []

    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=args.multimask_output,
        )

    if args.multimask_output:
        best = np.argmax(scores, axis=1)
        masks = masks[np.arange(masks.shape[0]), best]

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 合并所有检测到的物体 mask 为一张二值图
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    return combined_mask, masks, confidences.numpy().tolist(), labels


def prepare_jpeg_frames(image_files, temp_dir):
    """将图片文件转换/复制为 SAM2 video predictor 要求的 JPEG 格式 (00000.jpg, 00001.jpg, ...)"""
    for i, img_file in enumerate(image_files):
        dst = os.path.join(temp_dir, f"{i:05d}.jpg")
        if img_file.suffix.lower() in ('.jpg', '.jpeg'):
            os.symlink(os.path.abspath(str(img_file)), dst)
        else:
            img = cv2.imread(str(img_file))
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def detect_on_frame(img_path, image_predictor, grounding_model, args, device):
    """在指定帧上运行检测，返回 masks (N,H,W) numpy bool 和 labels"""
    image_source, image = load_image(img_path)
    image_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    h, w, _ = image_source.shape

    if len(boxes) == 0:
        return None, [], h, w

    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return masks, labels, h, w


def process_folder_with_tracking(image_files, sam2_predictor, grounding_model, args, device):
    """使用 SAM2 video predictor 进行帧间 tracking，保证检测结果时序稳定"""
    mask_dir = Path(args.mask_dir) if args.mask_dir else image_files[0].parent.parent / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tracking Mode] Processing {len(image_files)} frames")
    print(f"  Redetect interval: {args.redetect_interval or 'disabled (first frame only)'}")

    # 准备 JPEG 帧目录（SAM2 video predictor 要求）
    temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        print(f"  Preparing JPEG frames in {temp_dir} ...")
        prepare_jpeg_frames(image_files, temp_dir)

        # 构建 video predictor
        video_predictor = build_sam2_video_predictor(
            args.sam2_model_config, args.sam2_checkpoint, device=device)

        # 初始化 video predictor state
        inference_state = video_predictor.init_state(
            video_path=temp_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )

        if args.redetect_interval <= 0:
            # 简单模式：只在第一帧检测，然后 propagate 到所有帧
            video_segments = _track_simple(
                image_files, video_predictor, inference_state,
                sam2_predictor, grounding_model, args, device)
        else:
            # 分段重检测模式：每 N 帧重新检测并合并 ID
            video_segments = _track_with_redetection(
                image_files, video_predictor, inference_state,
                sam2_predictor, grounding_model, args, device)

        # 保存结果
        print(f"  Saving masks to {mask_dir}")
        print(f"  Saving visualizations to {output_dir}")
        for i, image_file in enumerate(image_files):
            if i in video_segments:
                seg_masks = video_segments[i]  # list of (H, W) bool arrays
                if len(seg_masks) > 0:
                    combined = np.any(np.stack(seg_masks), axis=0).astype(np.uint8) * 255
                else:
                    img = cv2.imread(str(image_file))
                    combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                img = cv2.imread(str(image_file))
                combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            cv2.imwrite(str(mask_dir / image_file.name), combined)

            # 可视化
            img = cv2.imread(str(image_file))
            if i in video_segments and len(video_segments[i]) > 0:
                seg_masks_np = np.stack(video_segments[i])
                class_ids = np.arange(len(seg_masks_np))
                boxes_vis = []
                for m in seg_masks_np:
                    ys, xs = np.where(m)
                    if len(ys) > 0:
                        boxes_vis.append([xs.min(), ys.min(), xs.max(), ys.max()])
                    else:
                        boxes_vis.append([0, 0, 0, 0])
                detections = sv.Detections(
                    xyxy=np.array(boxes_vis),
                    mask=seg_masks_np,
                    class_id=class_ids,
                )
                box_annotator = sv.BoxAnnotator()
                annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
                mask_annotator = sv.MaskAnnotator()
                annotated = mask_annotator.annotate(scene=annotated, detections=detections)
                cv2.imwrite(str(output_dir / image_file.name), annotated)
            else:
                cv2.imwrite(str(output_dir / image_file.name), img)

            if (i + 1) % 10 == 0 or i == 0:
                n_fg = np.count_nonzero(combined)
                total = combined.shape[0] * combined.shape[1]
                print(f"  [{i+1}/{len(image_files)}] {image_file.name} "
                      f"-> fg {n_fg}/{total} ({100*n_fg/total:.1f}%)")

        print(f"Done. {len(image_files)} masks saved to {mask_dir}")
        print(f"Done. {len(image_files)} visualizations saved to {output_dir}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _track_simple(image_files, video_predictor, inference_state,
                  image_predictor, grounding_model, args, device):
    """简单 tracking：在第一帧检测，propagate 到所有帧"""
    img_path = str(image_files[0])
    masks, labels, h, w = detect_on_frame(
        img_path, image_predictor, grounding_model, args, device)

    if masks is None or len(masks) == 0:
        print("  WARNING: No objects detected on first frame!")
        return {}

    print(f"  Detected {len(masks)} object(s) on frame 0: {labels}")

    # 注册 mask 到 video predictor
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for obj_id, mask in enumerate(masks, start=1):
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask,
            )

        # Propagate
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            frame_masks = []
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                frame_masks.append(out_mask)
            video_segments[out_frame_idx] = frame_masks

    return video_segments


def _track_with_redetection(image_files, video_predictor, inference_state,
                            image_predictor, grounding_model, args, device):
    """分段重检测 tracking：每 redetect_interval 帧重新检测并用 IoU 匹配保持 ID 连续"""
    step = args.redetect_interval
    num_frames = len(image_files)
    all_segments = {}
    sam2_masks = MaskDictionaryModel()
    objects_count = 0

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for start_idx in range(0, num_frames, step):
            img_path = str(image_files[start_idx])
            masks, labels, h, w = detect_on_frame(
                img_path, image_predictor, grounding_model, args, device)

            if masks is None or len(masks) == 0:
                print(f"  No detection on frame {start_idx}, using previous tracking state")
                # 没有检测到物体时，如果之前有 tracking 状态就继续用
                if len(sam2_masks.labels) == 0:
                    for fi in range(start_idx, min(start_idx + step, num_frames)):
                        all_segments[fi] = []
                    continue
                mask_dict = sam2_masks
            else:
                print(f"  Frame {start_idx}: detected {len(masks)} object(s): {labels}")
                mask_dict = MaskDictionaryModel(promote_type="mask")
                input_boxes = []
                for m in masks:
                    ys, xs = np.where(m)
                    if len(ys) > 0:
                        input_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                    else:
                        input_boxes.append([0, 0, 0, 0])
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(device),
                    box_list=torch.tensor(input_boxes, dtype=torch.float32),
                    label_list=labels,
                )
                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks,
                    iou_threshold=0.8,
                    objects_count=objects_count,
                )

            if len(mask_dict.labels) == 0:
                for fi in range(start_idx, min(start_idx + step, num_frames)):
                    all_segments[fi] = []
                continue

            # Reset state and add masks for this chunk
            video_predictor.reset_state(inference_state)
            for object_id, object_info in mask_dict.labels.items():
                video_predictor.add_new_mask(
                    inference_state,
                    start_idx,
                    object_id,
                    object_info.mask,
                )

            # Propagate for this chunk
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                    inference_state, max_frame_num_to_track=step, start_frame_idx=start_idx):
                frame_masks = []
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    frame_masks.append(out_mask)
                    # 更新 sam2_masks 用于下一段的 IoU 匹配
                    if out_frame_idx == min(start_idx + step - 1, num_frames - 1):
                        obj_info = ObjectInfo(
                            instance_id=out_obj_id,
                            mask=out_mask_logits[i] > 0.0,
                            class_name=mask_dict.get_target_class_name(out_obj_id),
                        )
                        sam2_masks.labels[out_obj_id] = obj_info
                all_segments[out_frame_idx] = frame_masks

    return all_segments


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dump_json = args.dump_json and not args.no_dump_json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=args.grounding_dino_config,
        model_checkpoint_path=args.grounding_dino_checkpoint,
        device=device,
    )

    img_path = Path(args.img_path)

    # 判断输入是文件夹还是单张图片
    if img_path.is_dir():
        image_files = sorted([
            f for f in img_path.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        ])
        if not image_files:
            print(f"No images found in {img_path}")
            return

        if args.tracking:
            # Tracking 模式：使用 SAM2 video predictor 进行帧间 tracking
            process_folder_with_tracking(
                image_files, sam2_predictor, grounding_model, args, device)
        else:
            # 原有逐帧独立检测模式
            if args.mask_dir:
                mask_dir = Path(args.mask_dir)
            else:
                mask_dir = img_path.parent / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {len(image_files)} images from {img_path}")
            print(f"Saving masks to {mask_dir}")
            print(f"Saving segmentation visualizations to {output_dir}")
            print(f"Text prompt: '{args.text_prompt}'")

            for i, image_file in enumerate(image_files):
                combined_mask, masks, confidences, labels = process_single_image(
                    str(image_file), sam2_predictor, grounding_model, args, device)

                mask_path = mask_dir / image_file.name
                cv2.imwrite(str(mask_path), combined_mask)

                # 保存可视化分割结果到 output_dir
                img = cv2.imread(str(image_file))
                if masks is not None and len(masks) > 0:
                    class_ids = np.array(list(range(len(labels))))
                    labels_display = [
                        f"{name} {conf:.2f}" for name, conf in zip(labels, confidences)
                    ]
                    boxes_for_vis = []
                    for m in masks:
                        ys, xs = np.where(m)
                        if len(ys) > 0:
                            boxes_for_vis.append([xs.min(), ys.min(), xs.max(), ys.max()])
                        else:
                            boxes_for_vis.append([0, 0, 0, 0])
                    input_boxes = np.array(boxes_for_vis)

                    detections = sv.Detections(
                        xyxy=input_boxes,
                        mask=np.array(masks).astype(bool),
                        class_id=class_ids,
                    )

                    box_annotator = sv.BoxAnnotator()
                    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                    label_annotator = sv.LabelAnnotator()
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels_display)
                    mask_annotator = sv.MaskAnnotator()
                    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                    cv2.imwrite(str(output_dir / image_file.name), annotated_frame)
                else:
                    cv2.imwrite(str(output_dir / image_file.name), img)

                if (i + 1) % 10 == 0 or i == 0:
                    n_fg = np.count_nonzero(combined_mask)
                    total = combined_mask.shape[0] * combined_mask.shape[1]
                    print(f"  [{i+1}/{len(image_files)}] {image_file.name} "
                          f"-> fg {n_fg}/{total} ({100*n_fg/total:.1f}%)")

            print(f"Done. {len(image_files)} masks saved to {mask_dir}")
            print(f"Done. {len(image_files)} visualizations saved to {output_dir}")

    else:
        # 单张图片模式
        combined_mask, masks, confidences, labels = process_single_image(
            str(img_path), sam2_predictor, grounding_model, args, device)

        # 保存 mask
        if args.mask_dir:
            mask_dir = Path(args.mask_dir)
            mask_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_dir / img_path.name), combined_mask)
            print(f"Saved mask to {mask_dir / img_path.name}")

        # 可视化
        img = cv2.imread(str(img_path))
        if masks is not None and len(masks) > 0:
            class_ids = np.array(list(range(len(labels))))
            labels_display = [
                f"{name} {conf:.2f}" for name, conf in zip(labels, confidences)
            ]
            image_source, _ = load_image(str(img_path))
            h, w, _ = image_source.shape
            boxes_for_vis = []
            for m in masks:
                ys, xs = np.where(m)
                if len(ys) > 0:
                    boxes_for_vis.append([xs.min(), ys.min(), xs.max(), ys.max()])
                else:
                    boxes_for_vis.append([0, 0, 0, 0])
            input_boxes = np.array(boxes_for_vis)

            detections = sv.Detections(
                xyxy=input_boxes,
                mask=np.array(masks).astype(bool),
                class_id=class_ids,
            )

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels_display)
            cv2.imwrite(str(output_dir / "groundingdino_annotated_image.jpg"), annotated_frame)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(str(output_dir / "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        cv2.imwrite(str(output_dir / "mask.png"), combined_mask)
        print(f"Saved visualization to {output_dir}")

        # dump json
        if dump_json and masks is not None and len(masks) > 0:
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            image_source, _ = load_image(str(img_path))
            h, w, _ = image_source.shape
            results = {
                "image_path": str(img_path),
                "annotations": [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score
                    in zip(labels, input_boxes.tolist(), mask_rles,
                           [[s] if not isinstance(s, list) else s for s in [0.0]*len(masks)])
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }

            json_path = output_dir / "grounded_sam2_local_image_demo_results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved JSON results to {json_path}")


if __name__ == "__main__":
    main()
