from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Common detection table:
#   det[i] = [x1, y1, x2, y2, conf, cls] in ORIGINAL image pixel coords
Detections = np.ndarray


@dataclass(frozen=True)
class LayoutCfg:
    """How outputs are named/written within a bucket."""
    layout: str  # preserve | flat | batch
    layout_batch_size: int
    pad_file: int
    pad_batch: int


@dataclass
class InferItem:
    im_path: Path
    rel_in: Path
    bucket: str
    bucket_key: str
    n_all: int
    n_kept: int
    sort_n: int
    im0_shape_hw: Tuple[int, int]  # (h,w)
    det_all: Optional[Detections]
    det_kept: Optional[Detections]


@dataclass(frozen=True)
class VizConfig:
    color_map: str = ""
    line_thickness: int = 2
    hide_labels: bool = False
    hide_conf: bool = False
    min_font_scale: float = 0.35
    max_font_scale: float = 0.9
    label_max_width_ratio: float = 0.70
    label_pad: int = 2
    label_bg_alpha: float = 0.55
    touch_tol: int = 0
    label_gap_ratio_w: float = 0.05
    label_gap_ratio_h: float = 0.90


@dataclass(frozen=True)
class RunConfig:
    # Model/inference
    model_onnx: str
    imgsz: int = 640
    conf_thres: float = 0.40
    iou_thres: float = 0.40
    max_det: int = 1000
    classes: Optional[List[int]] = None
    agnostic_nms: bool = False
    providers: Optional[List[str]] = None  # ONNXRuntime providers

    # Inference batching (ORT forward pass)
    infer_batch_size: int = 1

    # Postprocess: deduplicate near-identical boxes (after NMS)
    dedup: bool = False
    dedup_iou: float = 0.90
    dedup_merge: bool = False
    dedup_class_agnostic: bool = False

    # Outputs (labels default ON)
    save_txt: bool = True
    save_conf: bool = True
    save_vis_img: bool = False
    save_vis_mode: str = "all_except_no_objects"  # all_except_no_objects | all | kept_only
    copy_images: bool = False
    copy_images_mode: str = "all"  # all | kept_only

    # Crops
    save_crop: bool = False
    crop_layout_mode: str = "class"            # class (default) | image
    crop_include_ignored: bool = False  # when using keep-classes

    # Ranked crop ordering (class mode)
    crop_batch_size: int = 100  # batch size within each class after ranking
    crop_score_mode: str = "area_conf"  # area_conf | area | conf
    crop_score_area_exp: float = 1.0
    crop_score_conf_exp: float = 1.0
    crop_max_cache_images: int = 8  # number of source images to keep in-memory while writing crops


    # Crop region expansion (both supported)
    crop_pad_px: int = 0               # fixed pixels each side
    crop_pad_pct: float = 0.0          # percentage of box size each side (e.g. 0.10 = 10%)
    # Crop expansion mode
    crop_expand_mode: str = "margin"   # margin | context
    crop_context_gap_px: int = 0       # used only for context mode

    # Crop output format
    crop_ext: str = ".png"             # default .png
    crop_jpg_quality: int = 95
    crop_jpg_subsampling: str = "444"  # 444 = no chroma subsampling

    # Filtering/splitting
    split_by_detections: bool = False
    keep_classes_tokens: Optional[List[str]] = None
    keep_apply: str = "both"  # both | vis
    backup_original_labels: bool = False
    backup_labels_dirname: str = "labels_full"

    # Layout/sorting
    layout: str = "preserve"  # preserve | flat | batch
    layout_batch_size: int = 100

    sort_by_objects: bool = False
    # sorting controls (used when sort_by_objects=True AND layout in {flat,batch})
    sort_mode: str = "quality"  # objects | area_sum | area_max | conf_mean | conf_max | quality
    # composite score weights (sort_mode=quality)
    sort_w_count: float = 1.0
    sort_w_area_sum: float = 4.0
    sort_w_area_max: float = 1.0
    sort_w_conf_mean: float = 2.0
    sort_w_conf_max: float = 0.5
    sort_w_nclasses: float = 0.5

    meta_file: Optional[str] = None

    # Names (id -> label)
    names: Optional[str] = None  # path to JSON or YAML mapping (optional; sidecar auto-load required)

    # Viz
    viz: VizConfig = VizConfig()