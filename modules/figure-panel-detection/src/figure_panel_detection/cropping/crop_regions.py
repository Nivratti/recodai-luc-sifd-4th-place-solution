from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import cv2
import numpy as np

import hashlib
import math
import os
import shutil
from collections import OrderedDict


def _sanitize_component(s: str) -> str:
    s = str(s).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "unknown"


@dataclass(frozen=True)
class CropConfig:
    pad_px: int = 0
    pad_pct: float = 0.0
    expand_mode: str = "margin"      # margin | context
    context_gap_px: int = 0
    ext: str = ".png"               # default PNG
    jpg_quality: int = 95
    jpg_subsampling: str = "444"    # 444=no chroma subsampling


def _imwrite_params_for_crop(cfg: CropConfig) -> List[int]:
    ext = cfg.ext.lower()
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpg_quality)]

        # Try to enforce chroma sampling if OpenCV supports it
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
            params.append(int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR))

            if cfg.jpg_subsampling == "444" and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
                params.append(int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444))
            elif cfg.jpg_subsampling == "422" and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_422"):
                params.append(int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422))
            elif cfg.jpg_subsampling == "420" and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_420"):
                params.append(int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420))
            else:
                # constants missing; drop sampling option
                params = params[:-1]
        return params

    return []


def _axis_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _compute_context_limits(
    x1: float, y1: float, x2: float, y2: float,
    obstacles_xyxy: np.ndarray,
    gap: float,
) -> Tuple[float, float, float, float]:
    """
    Compute max allowed expansion limits based on nearby obstacle boxes:
      - left_limit: max obstacle.x2 among boxes left of us with vertical overlap
      - right_limit: min obstacle.x1 among boxes right of us with vertical overlap
      - top_limit: max obstacle.y2 among boxes above us with horizontal overlap
      - bottom_limit: min obstacle.y1 among boxes below us with horizontal overlap
    """
    left_limit = 0.0
    right_limit = float("inf")
    top_limit = 0.0
    bottom_limit = float("inf")

    if obstacles_xyxy.size == 0:
        return left_limit, right_limit, top_limit, bottom_limit

    ox1 = obstacles_xyxy[:, 0]
    oy1 = obstacles_xyxy[:, 1]
    ox2 = obstacles_xyxy[:, 2]
    oy2 = obstacles_xyxy[:, 3]

    # vertical overlap with our original box
    v_ov = np.maximum(0.0, np.minimum(y2, oy2) - np.maximum(y1, oy1))
    # horizontal overlap with our original box
    h_ov = np.maximum(0.0, np.minimum(x2, ox2) - np.maximum(x1, ox1))

    # left neighbors: obstacle is to the left (its right edge <= our left edge)
    left_mask = (v_ov > 0.0) & (ox2 <= x1)
    if np.any(left_mask):
        left_limit = float(np.max(ox2[left_mask] + gap))

    # right neighbors: obstacle is to the right (its left edge >= our right edge)
    right_mask = (v_ov > 0.0) & (ox1 >= x2)
    if np.any(right_mask):
        right_limit = float(np.min(ox1[right_mask] - gap))

    # top neighbors: obstacle above (its bottom <= our top)
    top_mask = (h_ov > 0.0) & (oy2 <= y1)
    if np.any(top_mask):
        top_limit = float(np.max(oy2[top_mask] + gap))

    # bottom neighbors: obstacle below (its top >= our bottom)
    bot_mask = (h_ov > 0.0) & (oy1 >= y2)
    if np.any(bot_mask):
        bottom_limit = float(np.min(oy1[bot_mask] - gap))

    return left_limit, right_limit, top_limit, bottom_limit


def _clip_box_xyxy(
    x1: float, y1: float, x2: float, y2: float,
    w: int, h: int,
    pad_px: int, pad_pct: float,
    expand_mode: str,
    context_gap_px: int,
    obstacles_xyxy: Optional[np.ndarray],
) -> Optional[Tuple[int, int, int, int]]:
    bw = float(max(1.0, x2 - x1))
    bh = float(max(1.0, y2 - y1))

    px = float(max(0.0, pad_pct)) * bw + float(max(0, pad_px))
    py = float(max(0.0, pad_pct)) * bh + float(max(0, pad_px))

    # desired expanded box (float)
    tx1 = x1 - px
    ty1 = y1 - py
    tx2 = x2 + px
    ty2 = y2 + py

    # context: limit expansion against neighbor boxes
    if expand_mode == "context" and obstacles_xyxy is not None and obstacles_xyxy.size > 0:
        obs = np.asarray(obstacles_xyxy, dtype=np.float32).reshape(-1, 4)

        # exclude obstacles that are (almost) identical to current box
        cur = np.array([x1, y1, x2, y2], dtype=np.float32)
        same = np.all(np.abs(obs - cur[None, :]) < 1e-3, axis=1)
        obs = obs[~same]

        gap = float(max(0, int(context_gap_px)))
        left_limit, right_limit, top_limit, bottom_limit = _compute_context_limits(x1, y1, x2, y2, obs, gap=gap)

        # don't shrink: limits must not cut into the original box
        left_limit = min(left_limit, x1)
        right_limit = max(right_limit, x2)
        top_limit = min(top_limit, y1)
        bottom_limit = max(bottom_limit, y2)

        tx1 = max(tx1, left_limit)
        tx2 = min(tx2, right_limit)
        ty1 = max(ty1, top_limit)
        ty2 = min(ty2, bottom_limit)

    # integer crop box (x2/y2 are exclusive)
    x1i = int(np.floor(tx1))
    y1i = int(np.floor(ty1))
    x2i = int(np.ceil(tx2))
    y2i = int(np.ceil(ty2))

    x1i = max(0, min(x1i, w - 1))
    y1i = max(0, min(y1i, h - 1))
    x2i = max(0, min(x2i, w))
    y2i = max(0, min(y2i, h))

    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i



# -----------------------------
# Ranked crop saving (class mode)
# -----------------------------

def _score_crop(area_px: float, conf: float, mode: str, area_exp: float = 1.0, conf_exp: float = 1.0) -> float:
    """Compute ranking score. Defaults: score=(area_px**area_exp)*(conf**conf_exp)."""
    a = float(max(0.0, area_px))
    c = float(max(0.0, conf))
    mode = str(mode or "").strip().lower().replace("-", "_")
    if mode in ("area", "area_only"):
        return a ** float(area_exp)
    if mode in ("conf", "confidence", "conf_only"):
        return c ** float(conf_exp)
    # default: area * conf (with optional exponents)
    return (a ** float(area_exp)) * (c ** float(conf_exp))


def _digits_for(n: int, min_digits: int = 1) -> int:
    if n <= 1:
        return int(min_digits)
    return int(max(min_digits, len(str(n - 1))))


def _class_dir_name(class_id: int, names: Dict[int, str]) -> str:
    return _sanitize_component(names.get(int(class_id), str(class_id)))


def _stable_rel_hash(rel_in: Path) -> str:
    h = hashlib.sha1(str(rel_in).encode("utf-8", errors="ignore")).hexdigest()
    return h[:8]


def save_ranked_class_crops(
    *,
    temp_root: Path,
    final_root: Path,
    detections: List[Dict[str, Any]],
    image_ctx: Dict[str, Dict[str, Any]],
    names: Dict[int, str],
    cfg: CropConfig,
    batch_size: int,
    score_mode: str = "area_conf",
    score_area_exp: float = 1.0,
    score_conf_exp: float = 1.0,
    max_cache_images: int = 8,
) -> List[Dict[str, Any]]:
    """Save crops in ranked order (class -> batch -> filename).

    Inputs:
      - detections: list of dicts with keys: img_key, class_id, conf, box_xyxy, area_px, det_index
      - image_ctx: per-image dict with keys: im_path (Path), rel_in (Path), out_img_rel (Path),
                   bucket (str), label_ref/image_ref/source_ref (str|None), obstacles_xyxy (np.ndarray|None)
    Writes into `temp_root` and returns mapping rows that reference `final_root`.
    """
    if not detections:
        return []

    # Fresh temp root
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    # Group by class
    by_cls: Dict[int, List[Dict[str, Any]]] = {}
    for d in detections:
        cid = int(d["class_id"])
        by_cls.setdefault(cid, []).append(d)

    # Image cache (LRU)
    cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _get_image(img_key: str) -> Optional[np.ndarray]:
        ctx = image_ctx.get(img_key)
        if not ctx:
            return None
        p = str(ctx.get("im_path"))
        if p in cache:
            cache.move_to_end(p)
            return cache[p]
        im = cv2.imread(p)
        if im is None:
            return None
        cache[p] = im
        cache.move_to_end(p)
        while len(cache) > int(max_cache_images):
            cache.popitem(last=False)
        return im

    mapping: List[Dict[str, Any]] = []

    for cls_id, dets in by_cls.items():
        # compute scores
        for d in dets:
            d["_score"] = _score_crop(
                area_px=float(d.get("area_px", 0.0)),
                conf=float(d.get("conf", 0.0)),
                mode=score_mode,
                area_exp=score_area_exp,
                conf_exp=score_conf_exp,
            )

        # sort: score desc, area desc, conf desc, then stable by (img_key, det_index)
        dets.sort(
            key=lambda x: (
                -float(x.get("_score", 0.0)),
                -float(x.get("area_px", 0.0)),
                -float(x.get("conf", 0.0)),
                str(x.get("img_key", "")),
                int(x.get("det_index", 0)),
            )
        )

        n = len(dets)
        if n == 0:
            continue

        pad_rank = _digits_for(n, min_digits=6)
        nb = int(math.ceil(n / max(1, int(batch_size))))
        pad_batch = _digits_for(nb, min_digits=3)

        class_dir = _class_dir_name(cls_id, names)
        class_root_tmp = temp_root / class_dir

        for rank, d in enumerate(dets):
            img_key = str(d.get("img_key", ""))
            ctx = image_ctx.get(img_key)
            if not ctx:
                continue

            im = _get_image(img_key)
            if im is None:
                continue
            h, w = im.shape[:2]

            x1, y1, x2, y2 = map(float, d.get("box_xyxy", [0, 0, 0, 0]))
            area_px = float(d.get("area_px", 0.0))
            conf = float(d.get("conf", 0.0))
            det_index = int(d.get("det_index", 0))

            obstacles_xyxy = ctx.get("obstacles_xyxy", None)
            if obstacles_xyxy is not None and not isinstance(obstacles_xyxy, np.ndarray):
                try:
                    obstacles_xyxy = np.asarray(obstacles_xyxy, dtype=np.float32)
                except Exception:
                    obstacles_xyxy = None

            clipped = _clip_box_xyxy(
                x1, y1, x2, y2,
                w=w, h=h,
                pad_px=int(cfg.pad_px),
                pad_pct=float(cfg.pad_pct),
                expand_mode=str(cfg.expand_mode),
                context_gap_px=int(cfg.context_gap_px),
                obstacles_xyxy=obstacles_xyxy,
            )
            if clipped is None:
                continue
            cx1, cy1, cx2, cy2 = clipped
            crop = im[cy1:cy2, cx1:cx2]
            if crop is None or crop.size == 0:
                continue

            b = int(rank // max(1, int(batch_size)))
            bname = f"batch_{b:0{pad_batch}d}"
            dest_dir = class_root_tmp / bname
            dest_dir.mkdir(parents=True, exist_ok=True)

            # stable, unique filename (rank prefix keeps directory listing in-ranked)
            rel_in = ctx.get("rel_in", Path(str(ctx.get("im_path"))).name)
            if isinstance(rel_in, str):
                rel_in = Path(rel_in)
            src_stem = Path(str(ctx.get("im_path"))).stem
            h8 = _stable_rel_hash(rel_in)

            crop_name = f"{rank:0{pad_rank}d}__{src_stem}__{h8}__d{det_index:03d}{cfg.ext}"
            tmp_path = dest_dir / crop_name

            imwrite_params = _imwrite_params_for_crop(cfg)
            ok = cv2.imwrite(str(tmp_path), crop, imwrite_params)
            if not ok:
                continue

            # Build mapping row referencing FINAL path (after swap/rename)
            rel_crop = Path(class_dir) / bname / crop_name
            final_path = final_root / rel_crop

            mapping.append(
                {
                    "rank": int(rank),
                    "score": float(d.get("_score", 0.0)),
                    "area_px": float(area_px),
                    "conf": float(conf),
                    "class_id": int(cls_id),
                    "class_name": names.get(int(cls_id), str(cls_id)),
                    "image_w": int(w),
                    "image_h": int(h),
                    "box_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "crop_box_xyxy": [int(cx1), int(cy1), int(cx2), int(cy2)],
                    "crop_rel": str(rel_crop),
                    "crop": str(final_path),
                    "source_image": str(ctx.get("im_path")),
                    "input_rel": str(ctx.get("rel_in")),
                    "out_img_rel": str(ctx.get("out_img_rel")),
                    "bucket": str(ctx.get("bucket")),
                    "det_index": int(det_index),
                    "label_ref": ctx.get("label_ref"),
                    "image_ref": ctx.get("image_ref"),
                    "source_ref": ctx.get("source_ref"),
                    "crop_ext": cfg.ext,
                    "crop_pad_px": int(cfg.pad_px),
                    "crop_pad_pct": float(cfg.pad_pct),
                    "crop_expand_mode": str(cfg.expand_mode),
                    "crop_context_gap_px": int(cfg.context_gap_px),
                    "score_mode": str(score_mode),
                    "score_area_exp": float(score_area_exp),
                    "score_conf_exp": float(score_conf_exp),
                }
            )

    # Write mapping.json into temp_root (will be moved to final)
    mapping_path = temp_root / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mapping


def save_crops(
    im0_bgr: np.ndarray,
    det_xyxy_conf_cls: np.ndarray,
    crop_root: Path,
    layout_mode: str,
    names: Dict[int, str],
    rel_in: Path,
    out_img_rel: Path,
    source_ref: str,
    label_ref: Optional[str],
    image_ref: str,
    bucket: str,
    cfg: CropConfig,
    obstacles_xyxy: Optional[np.ndarray] = None,  # Nx4, used for context mode
) -> List[Dict[str, Any]]:
    if det_xyxy_conf_cls is None or len(det_xyxy_conf_cls) == 0:
        return []

    crop_root.mkdir(parents=True, exist_ok=True)

    H, W = im0_bgr.shape[:2]
    det = np.asarray(det_xyxy_conf_cls)

    mapping: List[Dict[str, Any]] = []
    image_folder = _sanitize_component(rel_in.with_suffix("").as_posix().replace("/", "__"))
    imwrite_params = _imwrite_params_for_crop(cfg)

    for i, row in enumerate(det):
        x1, y1, x2, y2, conf, cls = row.tolist()
        cls_id = int(round(float(cls)))
        cls_name = names.get(cls_id, str(cls_id))

        clipped = _clip_box_xyxy(
            x1, y1, x2, y2,
            w=W, h=H,
            pad_px=int(cfg.pad_px),
            pad_pct=float(cfg.pad_pct),
            expand_mode=str(cfg.expand_mode),
            context_gap_px=int(cfg.context_gap_px),
            obstacles_xyxy=obstacles_xyxy,
        )
        if clipped is None:
            continue
        cx1, cy1, cx2, cy2 = clipped

        crop = im0_bgr[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            continue

        if layout_mode == "class":
            class_dir = _sanitize_component(cls_name)
            dest_dir = crop_root / class_dir / out_img_rel.parent
            stem = out_img_rel.stem
            crop_name = f"{stem}_{i:03d}{cfg.ext}"

        elif layout_mode == "image":
            # preserve batch folder like image outputs: crops/batch_000/<image_folder>/000.png
            dest_dir = crop_root / out_img_rel.parent / image_folder
            crop_name = f"{i:03d}{cfg.ext}"

        else:
            raise ValueError(f"Unknown crop layout mode: {layout_mode}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        crop_path = dest_dir / crop_name

        ok = cv2.imwrite(str(crop_path), crop, imwrite_params)
        if not ok:
            continue

        mapping.append(
            {
                "crop": str(crop_path),
                "bucket": bucket,
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": float(conf),
                "box_xyxy": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "source_ref": source_ref,
                "label_ref": label_ref,
                "image_ref": image_ref,
                "input_rel": str(rel_in),
                "out_img_rel": str(out_img_rel),
                "det_index": int(i),
                "crop_ext": cfg.ext,
                "crop_pad_px": int(cfg.pad_px),
                "crop_pad_pct": float(cfg.pad_pct),
                "crop_expand_mode": str(cfg.expand_mode),
                "crop_context_gap_px": int(cfg.context_gap_px),
            }
        )

    return mapping


def extract_crops(
    im0_bgr: np.ndarray,
    det_xyxy_conf_cls: np.ndarray,
    names: Dict[int, str],
    cfg: CropConfig,
    obstacles_xyxy: Optional[np.ndarray] = None,  # Nx4, used for context mode
) -> List[Dict[str, Any]]:
    """
    In-memory crop extractor.

    Parameters
    ----------
    im0_bgr:
        Original image in BGR uint8 format.
    det_xyxy_conf_cls:
        Nx6 detections in ORIGINAL image coords: [x1,y1,x2,y2,conf,cls]
    names:
        id->name mapping for class_name
    cfg:
        CropConfig controlling padding/expansion
    obstacles_xyxy:
        Optional Nx4 obstacles used for expand_mode="context".
        If None and expand_mode="context", you typically want to pass det[:,0:4].

    Returns
    -------
    List[dict] where each dict contains:
      - crop_bgr: np.ndarray (Hc,Wc,3) uint8
      - box_xyxy: [x1,y1,x2,y2] crop box in original coords
      - class_id, class_name, conf, det_index
    """
    if det_xyxy_conf_cls is None or len(det_xyxy_conf_cls) == 0:
        return []

    if im0_bgr is None:
        raise ValueError("im0_bgr is None")
    if im0_bgr.ndim != 3 or im0_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image (H,W,3), got {im0_bgr.shape}")

    H, W = im0_bgr.shape[:2]
    det = np.asarray(det_xyxy_conf_cls)

    out: List[Dict[str, Any]] = []

    for i, row in enumerate(det):
        x1, y1, x2, y2, conf, cls = row.tolist()
        cls_id = int(round(float(cls)))
        cls_name = names.get(cls_id, str(cls_id))

        clipped = _clip_box_xyxy(
            x1, y1, x2, y2,
            w=W, h=H,
            pad_px=int(cfg.pad_px),
            pad_pct=float(cfg.pad_pct),
            expand_mode=str(cfg.expand_mode),
            context_gap_px=int(cfg.context_gap_px),
            obstacles_xyxy=obstacles_xyxy,
        )
        if clipped is None:
            continue
        cx1, cy1, cx2, cy2 = clipped

        crop = im0_bgr[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            continue

        out.append(
            {
                "crop_bgr": crop,
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": float(conf),
                "box_xyxy": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "det_index": int(i),
            }
        )

    return out
