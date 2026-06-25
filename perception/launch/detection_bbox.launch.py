"""
Launch file for GDINO + SAM2 bbox detection node (optional FoundationStereo depth).

Publishes tag-labeled 2D and 3D axis-aligned bounding boxes.

Usage:
    # 默认：产线工装检测（电机/螺丝刀/螺丝/钳子等 10 类 prompt）+ 时序跟踪 + RViz
    ros2 launch robot_perception detection_bbox.launch.py launch_rviz:=true

    # 桌面乐高积木:
    ros2 launch robot_perception detection_bbox.launch.py \
        auto_industry_scene_prompts:=false auto_lego_scene_prompts:=true

    # 指定乐高 tag（与抓取链路一致，更精确）:
    ros2 launch robot_perception detection_bbox.launch.py \
        auto_industry_scene_prompts:=false auto_lego_scene_prompts:=false \
        tags:="['2_4_blue_lego','2_10_red_bridge_lego']"

    # 自定义 prompt:
    ros2 launch robot_perception detection_bbox.launch.py \
        auto_industry_scene_prompts:=false auto_lego_scene_prompts:=false \
        text_prompts:="['blue lego block.', 'red lego block.']"

    # 恢复 32 类开放场景:
    ros2 launch robot_perception detection_bbox.launch.py \
        auto_industry_scene_prompts:=false auto_lego_scene_prompts:=false \
        auto_open_scene_prompts:=true
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_perception.utils.paths import (
    STEREO_PLAN_PATH,
    MINICPM_WEIGHT_DIR,
    PIXEL3D_MODEL_PATH,
)


def _as_bool(value):
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def _as_float(value):
    return float(value)


def _as_int(value):
    return int(value)


def _yaml_string_list(context, name):
    raw = LaunchConfiguration(name).perform(context).strip()
    if not raw:
        return []
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed if x is not None and str(x)]
    return [str(parsed)]


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('robot_perception')
    default_rviz_config = os.path.join(pkg_share, 'rviz', 'detection_bbox.rviz')

    cam_ns = _yaml_string_list(context, 'cam_ns') or ['/camera_head']
    tags = _yaml_string_list(context, 'tags')
    text_prompts = _yaml_string_list(context, 'text_prompts')

    rviz_config = LaunchConfiguration('rviz_config').perform(context)
    if not rviz_config:
        rviz_config = default_rviz_config

    default_calib = os.path.join(
        pkg_share, 'config', 'calib_results', 'full_calibration_result.txt')
    calib_file = LaunchConfiguration('calib_file').perform(context).strip()
    if not calib_file:
        calib_file = default_calib

    params = {
        'cam_ns': cam_ns,
        'allow_freeform_prompts': _as_bool(
            LaunchConfiguration('allow_freeform_prompts').perform(context)),
        'auto_open_scene_prompts': _as_bool(
            LaunchConfiguration('auto_open_scene_prompts').perform(context)),
        'auto_industry_scene_prompts': _as_bool(
            LaunchConfiguration('auto_industry_scene_prompts').perform(context)),
        'auto_lego_scene_prompts': _as_bool(
            LaunchConfiguration('auto_lego_scene_prompts').perform(context)),
        'accept_unmatched_detections': _as_bool(
            LaunchConfiguration('accept_unmatched_detections').perform(context)),
        'always_publish_annotated': _as_bool(
            LaunchConfiguration('always_publish_annotated').perform(context)),
        'fps': _as_float(LaunchConfiguration('fps').perform(context)),
        'box_threshold': _as_float(LaunchConfiguration('box_threshold').perform(context)),
        'text_threshold': _as_float(LaunchConfiguration('text_threshold').perform(context)),
        'min_detection_score': _as_float(
            LaunchConfiguration('min_detection_score').perform(context)),
        'use_temporal_tracking': _as_bool(
            LaunchConfiguration('use_temporal_tracking').perform(context)),
        'use_cutie_tracking': _as_bool(
            LaunchConfiguration('use_cutie_tracking').perform(context)),
        'cutie_seg_threshold': _as_float(
            LaunchConfiguration('cutie_seg_threshold').perform(context)),
        'track_mask_iou_min': _as_float(
            LaunchConfiguration('track_mask_iou_min').perform(context)),
        'track_area_ratio_min': _as_float(
            LaunchConfiguration('track_area_ratio_min').perform(context)),
        'track_area_ratio_max': _as_float(
            LaunchConfiguration('track_area_ratio_max').perform(context)),
        'track_global_detect_interval': _as_int(
            LaunchConfiguration('track_global_detect_interval').perform(context)),
        'track_stable_discovery_interval': _as_int(
            LaunchConfiguration('track_stable_discovery_interval').perform(context)),
        'track_empty_discovery_interval': _as_int(
            LaunchConfiguration('track_empty_discovery_interval').perform(context)),
        'track_skip_discovery_when_stable': _as_bool(
            LaunchConfiguration('track_skip_discovery_when_stable').perform(context)),
        'track_stable_min_age': _as_int(
            LaunchConfiguration('track_stable_min_age').perform(context)),
        'track_discovery_skip_reinit_iou': _as_float(
            LaunchConfiguration('track_discovery_skip_reinit_iou').perform(context)),
        'track_discovery_update_iou_min': _as_float(
            LaunchConfiguration('track_discovery_update_iou_min').perform(context)),
        'track_discovery_batch_size': _as_int(
            LaunchConfiguration('track_discovery_batch_size').perform(context)),
        'track_discovery_max_batches_per_frame': _as_int(
            LaunchConfiguration('track_discovery_max_batches_per_frame').perform(context)),
        'track_discovery_use_batches': _as_bool(
            LaunchConfiguration('track_discovery_use_batches').perform(context)),
        'track_discovery_gap_fill': _as_bool(
            LaunchConfiguration('track_discovery_gap_fill').perform(context)),
        'track_discovery_gap_fill_only': _as_bool(
            LaunchConfiguration('track_discovery_gap_fill_only').perform(context)),
        'track_gap_fill_discovery_interval': _as_int(
            LaunchConfiguration('track_gap_fill_discovery_interval').perform(context)),
        'track_spatial_blacklist_frames': _as_int(
            LaunchConfiguration('track_spatial_blacklist_frames').perform(context)),
        'track_spatial_blacklist_iou': _as_float(
            LaunchConfiguration('track_spatial_blacklist_iou').perform(context)),
        'track_lost_max_frames': _as_int(
            LaunchConfiguration('track_lost_max_frames').perform(context)),
        'track_depth_std_min': _as_float(
            LaunchConfiguration('track_depth_std_min').perform(context)),
        'track_assoc_iou_min': _as_float(
            LaunchConfiguration('track_assoc_iou_min').perform(context)),
        'track_assoc_require_label_match': _as_bool(
            LaunchConfiguration('track_assoc_require_label_match').perform(context)),
        'track_label_lock': _as_bool(
            LaunchConfiguration('track_label_lock').perform(context)),
        'track_score_ema': _as_float(
            LaunchConfiguration('track_score_ema').perform(context)),
        'use_layer2_roi_refine': _as_bool(
            LaunchConfiguration('use_layer2_roi_refine').perform(context)),
        'track_roi_expand_ratio': _as_float(
            LaunchConfiguration('track_roi_expand_ratio').perform(context)),
        'track_refine_box_threshold': _as_float(
            LaunchConfiguration('track_refine_box_threshold').perform(context)),
        'track_refine_score_min': _as_float(
            LaunchConfiguration('track_refine_score_min').perform(context)),
        'track_refine_iou_min': _as_float(
            LaunchConfiguration('track_refine_iou_min').perform(context)),
        'track_refine_iou_hard_min': _as_float(
            LaunchConfiguration('track_refine_iou_hard_min').perform(context)),
        'track_label_vote_window': _as_int(
            LaunchConfiguration('track_label_vote_window').perform(context)),
        'sam2_score_min': _as_float(
            LaunchConfiguration('sam2_score_min').perform(context)),
        'vlm_nms_iou': _as_float(
            LaunchConfiguration('vlm_nms_iou').perform(context)),
        'vlm_max_area_ratio': _as_float(
            LaunchConfiguration('vlm_max_area_ratio').perform(context)),
        'publish_track_debug': _as_bool(
            LaunchConfiguration('publish_track_debug').perform(context)),
        'slop': _as_float(LaunchConfiguration('slop').perform(context)),
        'device': LaunchConfiguration('device').perform(context),
        'use_stereo_depth': _as_bool(LaunchConfiguration('use_stereo_depth').perform(context)),
        'stereo_plan_path': LaunchConfiguration('stereo_plan_path').perform(context),
        'stereo_baseline': _as_float(LaunchConfiguration('stereo_baseline').perform(context)),
        'stereo_max_age_sec': _as_float(
            LaunchConfiguration('stereo_max_age_sec').perform(context)),
        'ir_optical_frame': LaunchConfiguration('ir_optical_frame').perform(context),
        'color_optical_frame': LaunchConfiguration('color_optical_frame').perform(context),
        'right_ir_optical_frame': LaunchConfiguration('right_ir_optical_frame').perform(context),
        'calib_file': calib_file,
        'workbench_frame_id': LaunchConfiguration('workbench_frame_id').perform(context),
        'publish_static_tf': _as_bool(
            LaunchConfiguration('publish_static_tf').perform(context)),
        'enable_world_roi_filter': _as_bool(
            LaunchConfiguration('enable_world_roi_filter').perform(context)),
        'world_roi_mode': LaunchConfiguration('world_roi_mode').perform(context),
        'world_forward_max_m': _as_float(
            LaunchConfiguration('world_forward_max_m').perform(context)),
        'workbench_z': _as_float(LaunchConfiguration('workbench_z').perform(context)),
        'workbench_surface_tol_m': _as_float(
            LaunchConfiguration('workbench_surface_tol_m').perform(context)),
        'workbench_max_height_m': _as_float(
            LaunchConfiguration('workbench_max_height_m').perform(context)),
        'enable_ransac_workbench_plane': _as_bool(
            LaunchConfiguration('enable_ransac_workbench_plane').perform(context)),
        'workbench_z_prior_tol_m': _as_float(
            LaunchConfiguration('workbench_z_prior_tol_m').perform(context)),
        'workbench_plane_ransac_iters': _as_int(
            LaunchConfiguration('workbench_plane_ransac_iters').perform(context)),
        'workbench_plane_inlier_thresh_m': _as_float(
            LaunchConfiguration('workbench_plane_inlier_thresh_m').perform(context)),
        'workbench_plane_min_inliers': _as_int(
            LaunchConfiguration('workbench_plane_min_inliers').perform(context)),
        'workbench_plane_sample_stride': _as_int(
            LaunchConfiguration('workbench_plane_sample_stride').perform(context)),
        'workbench_plane_roi_v_frac': _as_float(
            LaunchConfiguration('workbench_plane_roi_v_frac').perform(context)),
        'workbench_plane_normal_max_tilt_deg': _as_float(
            LaunchConfiguration('workbench_plane_normal_max_tilt_deg').perform(context)),
        'workbench_plane_ema': _as_float(
            LaunchConfiguration('workbench_plane_ema').perform(context)),
        'workbench_plane_update_interval': _as_int(
            LaunchConfiguration('workbench_plane_update_interval').perform(context)),
        'workbench_plane_exclude_objects': _as_bool(
            LaunchConfiguration('workbench_plane_exclude_objects').perform(context)),
        'workbench_plane_exclude_dilate_px': _as_int(
            LaunchConfiguration('workbench_plane_exclude_dilate_px').perform(context)),
        'publish_workbench_plane_marker': _as_bool(
            LaunchConfiguration('publish_workbench_plane_marker').perform(context)),
        'workbench_plane_marker_size_m': _as_float(
            LaunchConfiguration('workbench_plane_marker_size_m').perform(context)),
        'workbench_plane_marker_thickness_m': _as_float(
            LaunchConfiguration('workbench_plane_marker_thickness_m').perform(context)),
        'enable_hybrid_surface_mesh': _as_bool(
            LaunchConfiguration('enable_hybrid_surface_mesh').perform(context)),
        'publish_hybrid_surface_marker': _as_bool(
            LaunchConfiguration('publish_hybrid_surface_marker').perform(context)),
        'hybrid_surface_min_points': _as_int(
            LaunchConfiguration('hybrid_surface_min_points').perform(context)),
        'hybrid_surface_top_frac': _as_float(
            LaunchConfiguration('hybrid_surface_top_frac').perform(context)),
        'hybrid_surface_max_top_points': _as_int(
            LaunchConfiguration('hybrid_surface_max_top_points').perform(context)),
        'hybrid_surface_max_triangle_edge_m': _as_float(
            LaunchConfiguration('hybrid_surface_max_triangle_edge_m').perform(context)),
        'hybrid_surface_min_triangle_normal_z': _as_float(
            LaunchConfiguration('hybrid_surface_min_triangle_normal_z').perform(context)),
        'hybrid_surface_marker_alpha': _as_float(
            LaunchConfiguration('hybrid_surface_marker_alpha').perform(context)),
        'hybrid_surface_rim_search_m': _as_float(
            LaunchConfiguration('hybrid_surface_rim_search_m').perform(context)),
        'hybrid_surface_aabb_bottom': _as_bool(
            LaunchConfiguration('hybrid_surface_aabb_bottom').perform(context)),
        'hybrid_surface_outlier_filter': _as_bool(
            LaunchConfiguration('hybrid_surface_outlier_filter').perform(context)),
        'hybrid_surface_iqr_k': _as_float(
            LaunchConfiguration('hybrid_surface_iqr_k').perform(context)),
        'hybrid_surface_sor_k': _as_int(
            LaunchConfiguration('hybrid_surface_sor_k').perform(context)),
        'hybrid_surface_sor_std': _as_float(
            LaunchConfiguration('hybrid_surface_sor_std').perform(context)),
        'publish_camera_bbox': _as_bool(
            LaunchConfiguration('publish_camera_bbox').perform(context)),
        'publish_workbench_bbox': _as_bool(
            LaunchConfiguration('publish_workbench_bbox').perform(context)),
        'publish_rviz_markers': _as_bool(
            LaunchConfiguration('publish_rviz_markers').perform(context)),
        'rviz_show_fill': _as_bool(LaunchConfiguration('rviz_show_fill').perform(context)),
        'rviz_fill_alpha': _as_float(LaunchConfiguration('rviz_fill_alpha').perform(context)),
        'resize_width': _as_int(LaunchConfiguration('resize_width').perform(context)),
        'resize_height': _as_int(LaunchConfiguration('resize_height').perform(context)),
        'min_mask_pixels': _as_int(LaunchConfiguration('min_mask_pixels').perform(context)),
        'min_depth_points': _as_int(LaunchConfiguration('min_depth_points').perform(context)),
        'depth_min_m': _as_float(LaunchConfiguration('depth_min_m').perform(context)),
        'depth_max_m': _as_float(LaunchConfiguration('depth_max_m').perform(context)),
        'subscribe_detection_prompts': _as_bool(
            LaunchConfiguration('subscribe_detection_prompts').perform(context)),
        'detection_prompts_topic': LaunchConfiguration(
            'detection_prompts_topic').perform(context),
        'use_vlm_detect': _as_bool(LaunchConfiguration('use_vlm_detect').perform(context)),
        'vlm_base_url': LaunchConfiguration('vlm_base_url').perform(context),
        'vlm_api_key': LaunchConfiguration('vlm_api_key').perform(context),
        'vlm_model': LaunchConfiguration('vlm_model').perform(context),
        'vlm_debug_dir': LaunchConfiguration('vlm_debug_dir').perform(context),
        'use_scene_understand': _as_bool(
            LaunchConfiguration('use_scene_understand').perform(context)),
        'scene_prompt_config': LaunchConfiguration(
            'scene_prompt_config').perform(context),
        'scene_change_check_interval': _as_int(
            LaunchConfiguration('scene_change_check_interval').perform(context)),
        'scene_change_threshold': _as_float(
            LaunchConfiguration('scene_change_threshold').perform(context)),
        'scene_change_pixel_threshold': _as_int(
            LaunchConfiguration('scene_change_pixel_threshold').perform(context)),
        'scene_max_objects': _as_int(
            LaunchConfiguration('scene_max_objects').perform(context)),
        'scene_understand_cooldown': _as_float(
            LaunchConfiguration('scene_understand_cooldown').perform(context)),
        'scene_use_prior_hints': _as_bool(
            LaunchConfiguration('scene_use_prior_hints').perform(context)),
        'log_scene_prompts': _as_bool(
            LaunchConfiguration('log_scene_prompts').perform(context)),
        'scene_first_run_retry_sec': _as_float(
            LaunchConfiguration('scene_first_run_retry_sec').perform(context)),
        'scene_prompt_stale_sec': _as_float(
            LaunchConfiguration('scene_prompt_stale_sec').perform(context)),
        'scene_prompt_never_detected_grace_sec': _as_float(
            LaunchConfiguration('scene_prompt_never_detected_grace_sec').perform(context)),
        'scene_preserve_gdino_prompts': _as_bool(
            LaunchConfiguration('scene_preserve_gdino_prompts').perform(context)),
        'scene_force_refresh_empty_frames': _as_int(
            LaunchConfiguration('scene_force_refresh_empty_frames').perform(context)),
        'log_frame_timing': _as_bool(
            LaunchConfiguration('log_frame_timing').perform(context)),
        'publish_object_pointcloud': _as_bool(
            LaunchConfiguration('publish_object_pointcloud').perform(context)),
        'publish_mask_image': _as_bool(
            LaunchConfiguration('publish_mask_image').perform(context)),
        'publish_surface_mesh_topic': _as_bool(
            LaunchConfiguration('publish_surface_mesh_topic').perform(context)),
        'use_pixel3d': _as_bool(
            LaunchConfiguration('use_pixel3d').perform(context)),
        'pixel3d_model_path': LaunchConfiguration('pixel3d_model_path').perform(context),
        'pixel3d_low_vram': _as_bool(
            LaunchConfiguration('pixel3d_low_vram').perform(context)),
        'pixel3d_trigger_age': _as_int(
            LaunchConfiguration('pixel3d_trigger_age').perform(context)),
        'pixel3d_max_concurrent': _as_int(
            LaunchConfiguration('pixel3d_max_concurrent').perform(context)),
        'pixel3d_cache_dir': LaunchConfiguration('pixel3d_cache_dir').perform(context),
        'pixel3d_use_label_cache': _as_bool(
            LaunchConfiguration('pixel3d_use_label_cache').perform(context)),
        'pixel3d_offline_config': LaunchConfiguration(
            'pixel3d_offline_config').perform(context),
    }
    # ROS launch rejects empty string arrays — omit and use node defaults instead.
    if tags:
        params['tags'] = tags
    if text_prompts:
        params['text_prompts'] = text_prompts

    log_level = LaunchConfiguration('log_level').perform(context)

    return [
        Node(
            package='robot_perception',
            executable='detection_bbox',
            name='detection_bbox',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[params],
            additional_env={
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            },
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='detection_bbox_rviz',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(LaunchConfiguration('launch_rviz')),
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('robot_perception')
    default_rviz_config = os.path.join(pkg_share, 'rviz', 'detection_bbox.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('cam_ns', default_value="['/camera_head']",
                             description='Camera namespace(s) as YAML list'),
        DeclareLaunchArgument('tags', default_value="[]",
                             description='Object tags for labeled detection (empty = open detection by default)'),
        DeclareLaunchArgument('text_prompts', default_value="[]",
                             description='Arbitrary GDINO text prompts (overrides tags if set)'),
        DeclareLaunchArgument('allow_freeform_prompts', default_value='true',
                             description='Treat unknown strings on detection_prompts as text prompts'),
        DeclareLaunchArgument('auto_open_scene_prompts', default_value='false',
                             description='Use 32-class open-scene prompts when no tags/text_prompts'),
        DeclareLaunchArgument('auto_industry_scene_prompts', default_value='true',
                             description='Use built-in production-line prompts (motor/screwdriver/etc.)'),
        DeclareLaunchArgument('auto_lego_scene_prompts', default_value='false',
                             description='Use built-in lego color prompts for desk Lego'),
        DeclareLaunchArgument('accept_unmatched_detections', default_value='true',
                             description='Publish GDINO phrases even if not exactly matching a prompt'),
        DeclareLaunchArgument('always_publish_annotated', default_value='true',
                             description='Always publish annotated image with 2D/3D overlays'),
        DeclareLaunchArgument('fps', default_value='5.0',
                             description='Max detection frame rate'),
        DeclareLaunchArgument('box_threshold', default_value='0.35',
                             description='Grounding DINO box threshold'),
        DeclareLaunchArgument('text_threshold', default_value='0.25',
                             description='Grounding DINO text threshold'),
        DeclareLaunchArgument('min_detection_score', default_value='0.45',
                             description='Drop detections below this GDINO confidence'),
        DeclareLaunchArgument('use_temporal_tracking', default_value='true',
                             description='Enable Cutie+GDINO temporal tracking for stable IDs/labels'),
        DeclareLaunchArgument('use_cutie_tracking', default_value='true',
                             description='Use Cutie mask propagation between global detections'),
        DeclareLaunchArgument('cutie_seg_threshold', default_value='0.1',
                             description='Cutie segmentation threshold'),
        DeclareLaunchArgument('track_mask_iou_min', default_value='0.5',
                             description='Min mask IoU vs previous frame for fast track'),
        DeclareLaunchArgument('track_area_ratio_min', default_value='0.3',
                             description='Min mask area ratio vs previous frame'),
        DeclareLaunchArgument('track_area_ratio_max', default_value='3.0',
                             description='Max mask area ratio vs previous frame'),
        DeclareLaunchArgument('track_global_detect_interval', default_value='10',
                             description='Global GDINO interval (frames) when tracks are unstable/lost'),
        DeclareLaunchArgument('track_stable_discovery_interval', default_value='15',
                             description='Global GDINO interval (frames) when all tracks are stable; 0=use skip flag only'),
        DeclareLaunchArgument('track_empty_discovery_interval', default_value='5',
                             description='Global GDINO interval (frames) when no tracks exist'),
        DeclareLaunchArgument('track_skip_discovery_when_stable', default_value='true',
                             description='Skip periodic discovery when stable AND track_stable_discovery_interval=0'),
        DeclareLaunchArgument('track_stable_min_age', default_value='3',
                             description='Min track age (frames) to count as stable'),
        DeclareLaunchArgument('track_discovery_skip_reinit_iou', default_value='0.65',
                             description='Skip discovery mask update when IoU vs current mask exceeds this'),
        DeclareLaunchArgument('track_discovery_update_iou_min', default_value='0.2',
                             description='Reject discovery update when mask IoU vs track is below this'),
        DeclareLaunchArgument('track_discovery_batch_size', default_value='2',
                             description='GDINO prompts per discovery batch (shorter caption)'),
        DeclareLaunchArgument('track_discovery_max_batches_per_frame', default_value='0',
                             description='Max discovery batches per frame (0=all batches, rotate if less)'),
        DeclareLaunchArgument('track_discovery_use_batches', default_value='true',
                             description='Split scene prompts into batches for global discovery'),
        DeclareLaunchArgument('track_discovery_gap_fill', default_value='true',
                             description='When scene objects lack tracks, run focused GDINO discovery on undetected prompts'),
        DeclareLaunchArgument('track_discovery_gap_fill_only', default_value='true',
                             description='Gap-fill discovery searches only undetected scene prompts, not the full list'),
        DeclareLaunchArgument('track_gap_fill_discovery_interval', default_value='5',
                             description='Discovery interval (frames) while scene prompts remain untracked'),
        DeclareLaunchArgument('track_spatial_blacklist_frames', default_value='100',
                             description='Suppress redetections near rejected surface-like bboxes (frames)'),
        DeclareLaunchArgument('track_spatial_blacklist_iou', default_value='0.3',
                             description='IoU threshold for spatial redetection blacklist'),
        DeclareLaunchArgument('track_lost_max_frames', default_value='5',
                             description='Remove track after N consecutive failures'),
        DeclareLaunchArgument('track_depth_std_min', default_value='0.003',
                             description='Min depth std to distinguish object from flat surface'),
        DeclareLaunchArgument('track_assoc_iou_min', default_value='0.3',
                             description='Min bbox IoU to associate detection to track'),
        DeclareLaunchArgument('track_assoc_require_label_match', default_value='true',
                             description='Mature tracks only associate detections with same label'),
        DeclareLaunchArgument('track_label_lock', default_value='true',
                             description='Keep label fixed after track registration'),
        DeclareLaunchArgument('track_score_ema', default_value='0.7',
                             description='EMA weight for detection score smoothing'),
        DeclareLaunchArgument('use_layer2_roi_refine', default_value='true',
                             description='Layer2: ROI-local GDINO+SAM before global fallback'),
        DeclareLaunchArgument('track_roi_expand_ratio', default_value='2.0',
                             description='Expand track bbox for Layer2 ROI crop'),
        DeclareLaunchArgument('track_refine_box_threshold', default_value='0.25',
                             description='GDINO box threshold inside Layer2 ROI'),
        DeclareLaunchArgument('track_refine_score_min', default_value='0.35',
                             description='Min GDINO score to accept Layer2 refine'),
        DeclareLaunchArgument('track_refine_iou_min', default_value='0.5',
                             description='Mask IoU for refined mode in Layer2'),
        DeclareLaunchArgument('track_refine_iou_hard_min', default_value='0.25',
                             description='Min mask IoU to accept corrected Layer2 result'),
        DeclareLaunchArgument('track_label_vote_window', default_value='5',
                             description='Sliding window for label majority vote'),
        DeclareLaunchArgument('publish_track_debug', default_value='false',
                             description='Publish JSON track state on .../track_debug'),
        DeclareLaunchArgument('sam2_score_min', default_value='0.7',
                             description='Min SAM2 IoU prediction score to accept a mask'),
        DeclareLaunchArgument('vlm_nms_iou', default_value='0.5',
                             description='NMS IoU threshold for VLM multi-prompt dedup'),
        DeclareLaunchArgument('vlm_max_area_ratio', default_value='0.25',
                             description='Reject VLM boxes larger than this fraction of image'),
        DeclareLaunchArgument('slop', default_value='0.05',
                             description='Stereo time sync slop (seconds)'),
        DeclareLaunchArgument('device', default_value='',
                             description='Compute device (empty = auto)'),
        DeclareLaunchArgument('use_stereo_depth', default_value='false',
                             description='Use FoundationStereo TRT depth (requires stereo engine + IR topics)'),
        DeclareLaunchArgument('stereo_plan_path',
                             default_value=STEREO_PLAN_PATH,
                             description='FoundationStereo TRT engine path'),
        DeclareLaunchArgument('stereo_baseline', default_value='0.095',
                             description='Stereo baseline in meters (overridden by TF)'),
        DeclareLaunchArgument('stereo_max_age_sec', default_value='0.25',
                             description='Max age (s) of stereo depth vs color before fallback'),
        DeclareLaunchArgument('ir_optical_frame', default_value='',
                             description='Left IR optical frame for TF'),
        DeclareLaunchArgument('color_optical_frame', default_value='',
                             description='Color optical frame for TF'),
        DeclareLaunchArgument('right_ir_optical_frame', default_value='',
                             description='Right IR optical frame for TF'),
        DeclareLaunchArgument('calib_file', default_value='',
                             description='T_world_cam calib (.txt/.npz); empty = package config/calib_results'),
        DeclareLaunchArgument('workbench_frame_id', default_value='world',
                             description='World / workbench frame id for 3D bbox and RViz'),
        DeclareLaunchArgument('publish_static_tf', default_value='true',
                             description='Publish static TF world->camera from calib_file for RViz'),
        DeclareLaunchArgument('enable_world_roi_filter', default_value='true',
                             description='Filter published 3D bboxes by world ROI rules'),
        DeclareLaunchArgument('world_roi_mode', default_value='and',
                             description='ROI mode: and | or | surface_only'),
        DeclareLaunchArgument('world_forward_max_m', default_value='1.0',
                             description='Max distance (m) along camera forward axis in world'),
        DeclareLaunchArgument('workbench_z', default_value='-1.0',
                             description='Prior workbench height in world (m); XY plane ~0, table ~-1m'),
        DeclareLaunchArgument('workbench_surface_tol_m', default_value='0.04',
                             description='Tolerance (m) for object bottom on workbench plane'),
        DeclareLaunchArgument('workbench_max_height_m', default_value='0.35',
                             description='Max object height (m) above workbench to count as on-surface'),
        DeclareLaunchArgument('enable_ransac_workbench_plane', default_value='true',
                             description='RANSAC fit workbench plane from depth for dynamic ROI'),
        DeclareLaunchArgument('workbench_z_prior_tol_m', default_value='0.15',
                             description='RANSAC plane z must stay within this of workbench_z prior (m)'),
        DeclareLaunchArgument('workbench_plane_ransac_iters', default_value='200',
                             description='RANSAC iterations for workbench plane'),
        DeclareLaunchArgument('workbench_plane_inlier_thresh_m', default_value='0.015',
                             description='Inlier distance threshold for plane RANSAC (m)'),
        DeclareLaunchArgument('workbench_plane_min_inliers', default_value='500',
                             description='Minimum inlier points to accept a plane'),
        DeclareLaunchArgument('workbench_plane_sample_stride', default_value='4',
                             description='Depth pixel stride for plane sampling'),
        DeclareLaunchArgument('workbench_plane_roi_v_frac', default_value='0.55',
                             description='Use bottom fraction of image for plane fit (0-1)'),
        DeclareLaunchArgument('workbench_plane_normal_max_tilt_deg', default_value='12.0',
                             description='Max tilt from world +Z for accepted plane (deg)'),
        DeclareLaunchArgument('workbench_plane_ema', default_value='0.85',
                             description='EMA smoothing for plane normal/offset'),
        DeclareLaunchArgument('workbench_plane_update_interval', default_value='3',
                             description='Re-run RANSAC every N processed frames'),
        DeclareLaunchArgument('workbench_plane_exclude_objects', default_value='true',
                             description='Exclude last-frame object masks from plane RANSAC'),
        DeclareLaunchArgument('workbench_plane_exclude_dilate_px', default_value='8',
                             description='Dilate excluded object mask (px) for plane sampling'),
        DeclareLaunchArgument('publish_workbench_plane_marker', default_value='true',
                             description='Publish RViz semi-transparent workbench plane marker'),
        DeclareLaunchArgument('workbench_plane_marker_size_m', default_value='0.8',
                             description='RViz plane marker side length (m)'),
        DeclareLaunchArgument('workbench_plane_marker_thickness_m', default_value='0.004',
                             description='RViz plane marker thickness (m)'),
        DeclareLaunchArgument('enable_hybrid_surface_mesh', default_value='true',
                             description='Build hybrid surface: observed top + AABB box bottom'),
        DeclareLaunchArgument('publish_hybrid_surface_marker', default_value='true',
                             description='Publish RViz TRIANGLE_LIST hybrid surface markers'),
        DeclareLaunchArgument('hybrid_surface_min_points', default_value='0',
                             description='Min depth points for surface mesh (0 = use min_depth_points)'),
        DeclareLaunchArgument('hybrid_surface_top_frac', default_value='0.45',
                             description='Upper height fraction used for observed top Delaunay'),
        DeclareLaunchArgument('hybrid_surface_max_top_points', default_value='600',
                             description='Max points for top-surface triangulation'),
        DeclareLaunchArgument('hybrid_surface_max_triangle_edge_m', default_value='0.04',
                             description='Drop Delaunay triangles with longer edges (m)'),
        DeclareLaunchArgument('hybrid_surface_min_triangle_normal_z', default_value='0.15',
                             description='Keep top triangles with normal.z above this'),
        DeclareLaunchArgument('hybrid_surface_marker_alpha', default_value='0.55',
                             description='RViz hybrid surface marker alpha'),
        DeclareLaunchArgument('hybrid_surface_rim_search_m', default_value='0.012',
                             description='Radius (m) to estimate rim height from nearby depth points'),
        DeclareLaunchArgument('hybrid_surface_aabb_bottom', default_value='false',
                             description='Use AABB rectangle bottom cap; false = tight hull footprint'),
        DeclareLaunchArgument('hybrid_surface_outlier_filter', default_value='true',
                             description='Enable IQR+SOR outlier removal before mesh construction'),
        DeclareLaunchArgument('hybrid_surface_iqr_k', default_value='1.5',
                             description='IQR multiplier for axis outlier removal'),
        DeclareLaunchArgument('hybrid_surface_sor_k', default_value='8',
                             description='SOR k-nearest neighbors count'),
        DeclareLaunchArgument('hybrid_surface_sor_std', default_value='1.5',
                             description='SOR std_ratio threshold (lower = more aggressive)'),
        DeclareLaunchArgument('publish_camera_bbox', default_value='false',
                             description='Publish 3D AABB in camera frame'),
        DeclareLaunchArgument('publish_workbench_bbox', default_value='true',
                             description='Publish 3D AABB in world/workbench frame'),
        DeclareLaunchArgument('publish_rviz_markers', default_value='true',
                             description='Publish MarkerArray for RViz'),
        DeclareLaunchArgument('rviz_show_fill', default_value='false',
                             description='Draw semi-transparent AABB cube fill (off when using hybrid surface)'),
        DeclareLaunchArgument('rviz_fill_alpha', default_value='0.15',
                             description='RViz cube fill alpha'),
        DeclareLaunchArgument('resize_width', default_value='0',
                             description='Resize width (0 = original)'),
        DeclareLaunchArgument('resize_height', default_value='0',
                             description='Resize height (0 = original)'),
        DeclareLaunchArgument('min_mask_pixels', default_value='100',
                             description='Minimum SAM mask pixels'),
        DeclareLaunchArgument('min_depth_points', default_value='50',
                             description='Minimum depth points for 3D bbox'),
        DeclareLaunchArgument('depth_min_m', default_value='0.01',
                             description='Minimum valid depth (m)'),
        DeclareLaunchArgument('depth_max_m', default_value='1.5',
                             description='Maximum valid depth (m)'),
        DeclareLaunchArgument('subscribe_detection_prompts', default_value='true',
                             description='Subscribe to detection_prompts_topic for dynamic targets'),
        DeclareLaunchArgument('detection_prompts_topic',
                             default_value='/robot_perception/detection_prompts',
                             description='Topic for dynamic detection target prompts'),
        DeclareLaunchArgument('use_vlm_detect', default_value='false',
                             description='Use local VLM for bbox discovery instead of GDINO'),
        DeclareLaunchArgument('vlm_base_url', default_value='http://127.0.0.1:8000/v1',
                             description='Local MiniCPM transformers serve URL'),
        DeclareLaunchArgument('vlm_api_key', default_value='local',
                             description='VLM API key (local serve accepts any non-empty string)'),
        DeclareLaunchArgument('vlm_model',
                             default_value=MINICPM_WEIGHT_DIR,
                             description='VLM model id for local transformers serve (weight dir path)'),
        DeclareLaunchArgument('vlm_debug_dir', default_value='',
                             description='Directory for VLM debug outputs (reserved)'),
        DeclareLaunchArgument('use_scene_understand', default_value='false',
                             description='Enable VLM scene understanding (requires use_vlm_detect)'),
        DeclareLaunchArgument('scene_prompt_config', default_value='',
                             description='YAML for VLM scene prompt templates; empty=package config/scene_understand_prompts.yaml'),
        DeclareLaunchArgument('scene_change_check_interval', default_value='10',
                             description='Frames between scene change detection checks'),
        DeclareLaunchArgument('scene_change_threshold', default_value='0.15',
                             description='Fraction of untracked pixels changed to trigger re-understanding'),
        DeclareLaunchArgument('scene_change_pixel_threshold', default_value='30',
                             description='Grayscale diff threshold to count a pixel as changed'),
        DeclareLaunchArgument('scene_max_objects', default_value='10',
                             description='Max objects returned by scene understanding'),
        DeclareLaunchArgument('scene_understand_cooldown', default_value='10.0',
                             description='Minimum seconds between scene understanding triggers'),
        DeclareLaunchArgument('scene_use_prior_hints', default_value='true',
                             description='Pass last scene objects to VLM on re-runs for stability'),
        DeclareLaunchArgument('log_scene_prompts', default_value='true',
                             description='Log VLM describe_scene prompt/response on each scene understanding run'),
        DeclareLaunchArgument('scene_first_run_retry_sec', default_value='2.0',
                             description='Seconds between retries when first scene VLM parse fails'),
        DeclareLaunchArgument('scene_prompt_stale_sec', default_value='45.0',
                             description='Drop scene prompt if no matching detection for this many seconds (0=off)'),
        DeclareLaunchArgument('scene_prompt_never_detected_grace_sec', default_value='120.0',
                             description='Keep VLM prompts with zero GDINO hits for this long before prune'),
        DeclareLaunchArgument('scene_preserve_gdino_prompts', default_value='true',
                             description='Keep full VLM phrase (with color) as GDINO prompt for scene objects'),
        DeclareLaunchArgument('scene_force_refresh_empty_frames', default_value='20',
                             description='Force VLM scene refresh after this many empty frames (0 tracks, 0 dets)'),
        DeclareLaunchArgument('log_frame_timing', default_value='true',
                             description='Log per-frame pipeline timing (scene/track/publish)'),
        DeclareLaunchArgument('publish_object_pointcloud', default_value='false',
                             description='Publish per-object PointCloud2 with instance_id field'),
        DeclareLaunchArgument('publish_mask_image', default_value='false',
                             description='Publish instance-indexed mask image (mono16)'),
        DeclareLaunchArgument('publish_surface_mesh_topic', default_value='true',
                             description='Publish SurfaceMeshArray topic with per-object mesh data'),
        DeclareLaunchArgument('use_pixel3d', default_value='false',
                             description='Enable Pixel3D full mesh completion for tracked objects'),
        DeclareLaunchArgument('pixel3d_model_path',
                             default_value=PIXEL3D_MODEL_PATH,
                             description='Path to Pixel3D model weights'),
        DeclareLaunchArgument('pixel3d_low_vram', default_value='true',
                             description='Pixel3D low-VRAM mode (models stay on CPU, load to GPU per stage)'),
        DeclareLaunchArgument('pixel3d_trigger_age', default_value='5',
                             description='Track stable frames before triggering Pixel3D'),
        DeclareLaunchArgument('pixel3d_max_concurrent', default_value='1',
                             description='Max concurrent Pixel3D inference tasks'),
        DeclareLaunchArgument('pixel3d_cache_dir', default_value='',
                             description='Disk cache dir for Pixel3D meshes (empty=disabled)'),
        DeclareLaunchArgument('pixel3d_use_label_cache', default_value='true',
                             description='Reuse cached mesh for same-label objects'),
        DeclareLaunchArgument('pixel3d_offline_config', default_value='',
                             description='Pixel3D offline manifest JSON (empty=package default)'),
        DeclareLaunchArgument('launch_rviz', default_value='false',
                             description='Launch RViz2 with bundled config'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz_config,
                             description='Path to RViz2 config file'),
        DeclareLaunchArgument('log_level', default_value='info',
                             description='ROS2 log level'),
        OpaqueFunction(function=_launch_setup),
    ])
