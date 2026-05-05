#!/usr/bin/env python3
"""
mask_viz_analyze.py

Visualize + analyze instance masks stored in .npy files paired with images.

Purpose:
    Visualize supplemental compound figure data provided by competition host.

Key features:
- Pair images and masks by stem (filename without extension).
- Strict validation:
  - mask normalization to (N,H,W)
  - image/mask size must match (no resizing)
  - per-instance mask values must be binary {0,1} (strict; no thresholding)
  - guardrails for suspicious instance axis / max instances
- Per-instance connected-components region counting and region stats:
  area, percent coverage, bbox, centroid, perimeter estimate (boundary pixels), compactness proxy
- Outputs (flat by default):
  - colored combined instance mask PNG
  - overlay (instances)
  - overlay (boundaries)
- Logging via loguru with different console/file levels, full stacktraces
- Report: JSONL (nested structure) by default; optional CSV (flattened)
- tqdm progress bar; --workers for multithreaded processing

Dependencies:
  pip install numpy pillow loguru tqdm
Optional (recommended for speed/robustness):
  pip install opencv-python
Optional fallback:
  pip install scipy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

# Optional deps
try:
    import cv2  # type: ignore

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    from scipy import ndimage  # type: ignore

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -------------------------
# Exceptions (clear failures)
# -------------------------
class DatasetError(RuntimeError):
    pass


class ValidationError(DatasetError):
    pass


class ReadError(DatasetError):
    pass


# -------------------------
# Config / CLI
# -------------------------
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    stem: str
    image_path: Optional[Path]
    mask_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize & analyze .npy instance masks paired with images by stem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--images-dir", type=Path, required=True, help="Directory containing images.")
    p.add_argument("--masks-dir", type=Path, required=True, help="Directory containing .npy masks.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (flat output).")

    p.add_argument("--on-missing", choices=["error", "skip"], default="error",
                   help="If mask missing for an image.")
    p.add_argument("--on-orphan-mask", choices=["warn", "ignore", "error"], default="warn",
                   help="If mask exists but image is missing.")
    p.add_argument("--on-read-error", choices=["error", "warn-skip"], default="error",
                   help="Unreadable image/mask behavior.")
    p.add_argument("--on-mismatch", choices=["error", "skip"], default="error",
                   help="Image/mask HxW mismatch behavior.")

    p.add_argument("--max-instances", type=int, default=256,
                   help="Safety limit for max instance count N.")
    p.add_argument("--connectivity", choices=["4", "8"], default="8",
                   help="Connected components connectivity.")
    p.add_argument("--boundary-width", type=int, default=4,
                   help="Boundary thickness in pixels (>=1).")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Overlay alpha for instance overlay (0..1).")

    p.add_argument("--mask-prefix", type=str, default="mask_",
                   help="Prefix for mask outputs.")
    p.add_argument("--viz-prefix", type=str, default="viz_",
                   help="Prefix for visualization outputs.")

    p.add_argument("--no-overlay", action="store_true", help="Disable instance overlay output.")
    p.add_argument("--no-boundary", action="store_true", help="Disable boundary overlay output.")
    p.add_argument("--no-colored-mask", action="store_true", help="Disable colored combined mask output.")

    p.add_argument("--report-format", choices=["jsonl", "csv"], default="jsonl",
                   help="Report format: JSONL (nested) or CSV (flattened per-region).")
    p.add_argument("--report-name", type=str, default="report",
                   help="Base name for report file (without extension).")

    p.add_argument("--dry-run", action="store_true",
                   help="Do all validations + stats, but do NOT write image outputs (report/log still written).")

    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker threads (1 = serial).")

    p.add_argument("--log-file-level",
                   choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default="DEBUG")
    p.add_argument("--log-console-level",
                   choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default="INFO")

    p.add_argument("--combined-subdir", type=str, default="combined",
                help="Subfolder under out-dir for combined outputs.")
    p.add_argument("--per-instance-subdir", type=str, default="per_instance",
                help="Subfolder under out-dir for per-instance outputs.")
    p.add_argument("--no-per-instance", action="store_true",
                help="Disable saving per-instance (per-channel) outputs.")

    return p.parse_args()


def setup_logging(out_dir: Path, console_level: str, file_level: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level=console_level,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    logger.add(
        str(log_path),
        level=file_level,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    return log_path


# -------------------------
# Pairing / discovery
# -------------------------
def list_images(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.exists():
        raise DatasetError(f"Images dir does not exist: {images_dir}")
    out: Dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            out[p.stem] = p
    return out


def list_masks(masks_dir: Path) -> Dict[str, Path]:
    if not masks_dir.exists():
        raise DatasetError(f"Masks dir does not exist: {masks_dir}")
    out: Dict[str, Path] = {}
    for p in masks_dir.rglob("*.npy"):
        if p.is_file():
            out[p.stem] = p
    return out


def build_pairs(images: Dict[str, Path], masks: Dict[str, Path]) -> Tuple[List[Pair], List[Pair]]:
    """Returns (paired_or_image_only, orphan_masks)."""
    stems = sorted(set(images.keys()) | set(masks.keys()))
    pairs: List[Pair] = []
    orphans: List[Pair] = []
    for s in stems:
        ip = images.get(s)
        mp = masks.get(s)
        pr = Pair(stem=s, image_path=ip, mask_path=mp)
        if ip is None and mp is not None:
            orphans.append(pr)
        else:
            pairs.append(pr)
    return pairs, orphans


# -------------------------
# I/O
# -------------------------
def read_image(path: Path) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        raise ReadError(f"Failed to read image: {path}") from e


def load_npy(path: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)
    except Exception as e:
        raise ReadError(f"Failed to load npy: {path}") from e


# -------------------------
# Mask normalization & validation
# -------------------------
def normalize_mask_to_nhw(mask: np.ndarray, max_instances: int) -> np.ndarray:
    """
    Normalize mask to shape (N, H, W).
    Accepts:
      (N,H,W), (H,W,N), (H,W)
    """
    if mask.ndim == 2:
        h, w = mask.shape
        out = mask.reshape(1, h, w)
        return out

    if mask.ndim != 3:
        raise ValidationError(f"Mask must be 2D or 3D, got shape={mask.shape}")

    a, b, c = mask.shape  # ambiguous

    # Candidate 1: assume (N,H,W)
    cand1 = (a, b, c)
    # Candidate 2: assume (H,W,N) -> transpose to (N,H,W)
    cand2 = (c, a, b)

    # Heuristic: choose candidate where N is small (<= max_instances) and H,W are "large-ish"
    def score(shape_nhw: Tuple[int, int, int]) -> Tuple[int, int]:
        n, h, w = shape_nhw
        # lower is better
        penalty = 0
        if n > max_instances:
            penalty += 10_000 + (n - max_instances)
        if h < 8 or w < 8:
            penalty += 1000
        return (penalty, n)

    s1 = score(cand1)
    s2 = score(cand2)

    if s2 < s1:
        nhw = np.transpose(mask, (2, 0, 1))
    else:
        nhw = mask

    n, h, w = nhw.shape

    # Guardrail: suspicious axis (classic bug: width/height treated as N)
    if (n == h or n == w) and n > 64:
        raise ValidationError(
            f"Suspicious instance axis: normalized N={n} equals H={h} or W={w}. "
            f"Raw shape={mask.shape}, norm shape={nhw.shape}"
        )

    if n > max_instances:
        raise ValidationError(
            f"Too many instances: N={n} exceeds --max-instances={max_instances}. "
            f"Raw shape={mask.shape}, norm shape={nhw.shape}"
        )

    return nhw


def unique_as_python_list(u: np.ndarray) -> List[Any]:
    # Convert numpy scalars to python scalars for JSON
    return [x.item() if hasattr(x, "item") else x for x in u.tolist()]


def check_instance_binary_strict(inst: np.ndarray) -> List[Any]:
    """
    Returns unique values (python scalars).
    Enforces strict binary values {0,1} (with dtype-appropriate exactness).
    """
    u = np.unique(inst)
    u_list = unique_as_python_list(u)

    # Strict exact checks:
    # - bool: {False, True} -> ok
    # - int:  {0, 1} -> ok
    # - float: values must be exactly 0.0 or 1.0
    if inst.dtype == np.bool_:
        mapped = set(int(v) for v in u.astype(np.uint8).tolist())
        if not mapped.issubset({0, 1}):
            raise ValidationError(f"Non-binary bool instance values: {u_list}")
        return u_list

    if np.issubdtype(inst.dtype, np.integer):
        mapped = set(int(v) for v in u.tolist())
        if not mapped.issubset({0, 1}):
            raise ValidationError(f"Non-binary integer instance values: {u_list}")
        return u_list

    if np.issubdtype(inst.dtype, np.floating):
        for v in u.tolist():
            if not (v == 0.0 or v == 1.0):
                raise ValidationError(f"Non-binary float instance values: {u_list}")
        return u_list

    # Other dtypes are not supported for strict binary masks
    raise ValidationError(f"Unsupported mask dtype for strict binary check: {inst.dtype}, uniques={u_list}")


# -------------------------
# Connected components + region stats
# -------------------------
def erode_mask(mask01: np.ndarray, connectivity: int) -> np.ndarray:
    """
    mask01: uint8 {0,1}
    Returns eroded mask {0,1}.
    """
    if HAVE_CV2:
        kernel = np.ones((3, 3), dtype=np.uint8)
        # cv2 doesn't use connectivity for erosion; kernel handles it approximately
        er = cv2.erode(mask01, kernel, iterations=1)
        return (er > 0).astype(np.uint8)

    if HAVE_SCIPY:
        structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=bool
        )
        er = ndimage.binary_erosion(mask01.astype(bool), structure=structure, iterations=1)
        return er.astype(np.uint8)

    # Pure numpy fallback (slow, crude): 3x3 erosion requiring all neighbors
    m = mask01.astype(bool)
    pad = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.ones_like(m, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if connectivity == 4 and abs(dy) + abs(dx) == 2:
                continue
            out &= pad[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]]
    return out.astype(np.uint8)


def dilate_mask(mask01: np.ndarray, width: int, connectivity: int) -> np.ndarray:
    if width <= 1:
        return mask01

    iters = max(1, width - 1)

    if HAVE_CV2:
        kernel = np.ones((3, 3), dtype=np.uint8)
        d = cv2.dilate(mask01, kernel, iterations=iters)
        return (d > 0).astype(np.uint8)

    if HAVE_SCIPY:
        structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=bool
        )
        d = ndimage.binary_dilation(mask01.astype(bool), structure=structure, iterations=iters)
        return d.astype(np.uint8)

    # Pure fallback: repeated 3x3 dilation
    m = mask01.astype(bool)
    for _ in range(iters):
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        out = np.zeros_like(m, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if connectivity == 4 and abs(dy) + abs(dx) == 2:
                    continue
                out |= pad[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]]
        m = out
    return m.astype(np.uint8)


def boundary_pixels(mask01: np.ndarray, connectivity: int) -> np.ndarray:
    """Boundary as mask & ~erode(mask)."""
    er = erode_mask(mask01, connectivity)
    b = (mask01 > 0) & (er == 0)
    return b.astype(np.uint8)


def connected_components_regions(mask_bool: np.ndarray, connectivity: int) -> List[Dict[str, Any]]:
    """
    Compute connected components and return list of region dicts.
    Coordinates:
      bbox: (x1,y1,x2,y2) inclusive
      centroid: (cx,cy) float in image coordinates
    """
    mask01 = mask_bool.astype(np.uint8)

    regions: List[Dict[str, Any]] = []
    if mask01.sum() == 0:
        return regions

    if HAVE_CV2:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask01, connectivity=connectivity)
        # label 0=background
        for lab in range(1, num_labels):
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            w = int(stats[lab, cv2.CC_STAT_WIDTH])
            h = int(stats[lab, cv2.CC_STAT_HEIGHT])
            area = int(stats[lab, cv2.CC_STAT_AREA])
            cx = float(centroids[lab][0])
            cy = float(centroids[lab][1])

            comp = (labels == lab)
            bnd = boundary_pixels(comp.astype(np.uint8), connectivity)
            perim = int(bnd.sum())
            compact = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else None

            regions.append({
                "region_id": lab,
                "area_px": area,
                "bbox_xyxy": [x, y, x + w - 1, y + h - 1],
                "centroid_xy": [cx, cy],
                "perimeter_px_est": perim,
                "compactness_4piA_over_P2": compact,
            })
        return regions

    if HAVE_SCIPY:
        structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=bool
        )
        labels, num = ndimage.label(mask_bool, structure=structure)
        if num == 0:
            return regions

        objs = ndimage.find_objects(labels)
        for i, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            sub = labels[slc] == i
            area = int(sub.sum())
            if area == 0:
                continue
            y1, y2 = slc[0].start, slc[0].stop
            x1, x2 = slc[1].start, slc[1].stop
            # centroid in sub coords then shift
            cy, cx = ndimage.center_of_mass(sub)
            cx_abs = float(cx + x1)
            cy_abs = float(cy + y1)

            bnd = boundary_pixels(sub.astype(np.uint8), connectivity)
            perim = int(bnd.sum())
            compact = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else None

            regions.append({
                "region_id": i,
                "area_px": area,
                "bbox_xyxy": [int(x1), int(y1), int(x2 - 1), int(y2 - 1)],
                "centroid_xy": [cx_abs, cy_abs],
                "perimeter_px_est": perim,
                "compactness_4piA_over_P2": compact,
            })
        return regions

    raise DatasetError("Need opencv-python or scipy for connected components. Install opencv-python (recommended).")


# -------------------------
# Visualization helpers
# -------------------------
def hsv_to_rgb255(h: float, s: float, v: float) -> Tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    if n <= 0:
        return []
    cols = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        cols.append(hsv_to_rgb255(h, 0.9, 0.95))
    return cols


def alpha_blend(base: np.ndarray, mask_bool: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Blend color onto base where mask_bool is True."""
    out = base.astype(np.float32).copy()
    idx = mask_bool
    if idx.any():
        c = np.array(color, dtype=np.float32)
        out[idx] = out[idx] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def paint_pixels(base: np.ndarray, mask_bool: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Hard paint color where mask_bool is True."""
    out = base.copy()
    out[mask_bool] = np.array(color, dtype=np.uint8)
    return out

def alpha_paint(base: np.ndarray, mask_bool: np.ndarray, color, alpha: float = 0.9) -> np.ndarray:
    out = base.astype(np.float32).copy()
    idx = mask_bool
    if idx.any():
        c = np.array(color, dtype=np.float32)
        out[idx] = out[idx] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

# -------------------------
# Reporting
# -------------------------
def open_report_paths(out_dir: Path, base_name: str, fmt: str) -> Tuple[Path, Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir
    report_path = report_dir / f"{base_name}.{fmt}"
    csv_flat_path = None
    if fmt == "csv":
        # additionally write a per-region flat CSV
        csv_flat_path = report_dir / f"{base_name}_regions_flat.csv"
    return report_path, csv_flat_path


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_flat_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        # still create empty file with headers if possible
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Main per-sample processing
# -------------------------
def process_one(
    pair: Pair,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      (jsonl_record or None), flat_region_rows_for_csv
    """
    stem = pair.stem
    ip = pair.image_path
    mp = pair.mask_path

    if ip is None and mp is not None:
        msg = f"Orphan mask (no image): stem={stem} mask={mp}"
        if args.on_orphan_mask == "ignore":
            logger.debug(msg)
            return None, []
        if args.on_orphan_mask == "warn":
            logger.warning(msg)
            return None, []
        raise DatasetError(msg)

    if ip is not None and mp is None:
        msg = f"Missing mask for image: stem={stem} image={ip}"
        if args.on_missing == "skip":
            logger.warning(msg)
            return None, []
        raise DatasetError(msg)

    if ip is None or mp is None:
        # nothing to do
        return None, []

    # Read
    try:
        img = read_image(ip)
        mask_raw = load_npy(mp)
    except Exception:
        if args.on_read_error == "warn-skip":
            logger.exception(f"Read error (skipping): stem={stem}")
            return None, []
        raise  # full trace via loguru in outer wrapper

    h_img, w_img = img.shape[0], img.shape[1]

    # Normalize mask
    mask_norm = normalize_mask_to_nhw(mask_raw, max_instances=args.max_instances)
    n, h_m, w_m = mask_norm.shape

    # Size match
    if (h_m, w_m) != (h_img, w_img):
        msg = (f"Size mismatch for stem={stem}: image(H,W)=({h_img},{w_img}) "
               f"mask(H,W)=({h_m},{w_m}) raw_shape={tuple(mask_raw.shape)} norm_shape={tuple(mask_norm.shape)}")
        if args.on_mismatch == "skip":
            logger.warning(msg)
            return None, []
        raise ValidationError(msg)

    # Strict binary checks per instance + stack unique
    stack_u = np.unique(mask_norm)
    stack_u_list = unique_as_python_list(stack_u)

    inst_summaries: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    total_pixels = float(h_img * w_img)
    colors = distinct_colors(n)

    # Prepare visual artifacts
    combined_mask_rgb = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    overlay_instances = img.copy()
    overlay_boundaries = img.copy()
    boundary_mask_rgb = np.zeros((h_img, w_img, 3), dtype=np.uint8)  # boundary-only (black bg)

    connectivity = int(args.connectivity)
    boundary_width = max(1, int(args.boundary_width))

    combined_dir = args.out_dir / args.combined_subdir
    per_root = args.out_dir / args.per_instance_subdir
    stem_dir = per_root / stem

    if not args.dry_run:
        combined_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_per_instance:
            stem_dir.mkdir(parents=True, exist_ok=True)
            
    # Process each instance
    for i in range(n):
        inst = mask_norm[i]

        # Unique + strict binary
        u_list = check_instance_binary_strict(inst)

        # Convert to bool foreground
        if inst.dtype == np.bool_:
            inst_fg = inst
        elif np.issubdtype(inst.dtype, np.integer) or np.issubdtype(inst.dtype, np.floating):
            inst_fg = (inst == 1)
        else:
            # Shouldn't happen due to strict check
            raise ValidationError(f"Unsupported dtype after strict check: {inst.dtype}")

        fg_px = int(inst_fg.sum())
        fg_pct = float((fg_px / total_pixels) * 100.0)

        # Regions
        regions = connected_components_regions(inst_fg, connectivity=connectivity)
        # Add percent coverage to each region
        for r in regions:
            r["percent_of_image"] = float((r["area_px"] / total_pixels) * 100.0)

        inst_summary = {
            "instance_index": i,
            "unique_values": u_list,
            "fg_pixels": fg_px,
            "fg_percent_of_image": fg_pct,
            "num_regions": len(regions),
            "regions": regions,
        }
        inst_summaries.append(inst_summary)

        # Flat CSV rows (per region)
        for r in regions:
            flat_rows.append({
                "stem": stem,
                "image_path": str(ip),
                "mask_path": str(mp),
                "instance_index": i,
                "region_id": r["region_id"],
                "area_px": r["area_px"],
                "percent_of_image": r["percent_of_image"],
                "bbox_x1": r["bbox_xyxy"][0],
                "bbox_y1": r["bbox_xyxy"][1],
                "bbox_x2": r["bbox_xyxy"][2],
                "bbox_y2": r["bbox_xyxy"][3],
                "centroid_x": r["centroid_xy"][0],
                "centroid_y": r["centroid_xy"][1],
                "perimeter_px_est": r["perimeter_px_est"],
                "compactness_4piA_over_P2": r["compactness_4piA_over_P2"],
            })

        # Visuals (unless dry-run)
        col = colors[i]

        if not args.no_colored_mask:
            combined_mask_rgb[inst_fg] = np.array(col, dtype=np.uint8)

        if not args.no_overlay:
            overlay_instances = alpha_blend(overlay_instances, inst_fg, col, alpha=float(args.alpha))

        if not args.no_boundary and fg_px > 0:
            # Boundary per instance (union of region boundaries)
            inst01 = inst_fg.astype(np.uint8)
            bnd = boundary_pixels(inst01, connectivity=connectivity)
            bnd = dilate_mask(bnd, width=boundary_width, connectivity=connectivity).astype(bool)
            # overlay_boundaries = paint_pixels(overlay_boundaries, bnd, col)
            # boundary overlay (existing behavior)
            overlay_boundaries = alpha_paint(overlay_boundaries, bnd, col, alpha=0.95)

            # boundary-only mask (black bg, colored boundaries)
            boundary_mask_rgb[bnd] = np.array(col, dtype=np.uint8)

        # ---- per-instance outputs ----
        if (not args.dry_run) and (not args.no_per_instance):
            digits = max(3, len(str(n - 1)))
            tag = f"inst_{i:0{digits}d}"

            # 1) instance colored mask on black bg
            inst_mask_rgb = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            inst_mask_rgb[inst_fg] = np.array(col, dtype=np.uint8)

            # 2) overlay of this instance on image
            inst_overlay = alpha_blend(img.copy(), inst_fg, col, alpha=float(args.alpha))

            # 3) boundary-only for this instance
            inst_bnd_rgb = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            inst_bnd_rgb[bnd] = np.array(col, dtype=np.uint8)

            Image.fromarray(inst_mask_rgb).save(stem_dir / f"{tag}_mask.png")
            Image.fromarray(inst_overlay).save(stem_dir / f"{tag}_overlay.png")
            Image.fromarray(inst_bnd_rgb).save(stem_dir / f"{tag}_boundary.png")

        logger.debug(
            f"[{stem}] instance {i+1}/{n}: uniques={u_list} fg_px={fg_px} regions={len(regions)}"
        )

    record = {
        "stem": stem,
        "image_path": str(ip),
        "mask_path": str(mp),
        "image_hw": [h_img, w_img],
        "mask_hw": [h_m, w_m],
        "mask_raw_shape": list(mask_raw.shape),
        "mask_norm_shape": [n, h_m, w_m],
        "n_instances": n,
        "stack_unique_values": stack_u_list,
        "instances": inst_summaries,
    }

    if not args.dry_run:
        # Colored combined instance mask
        if not args.no_colored_mask:
            out_mask = combined_dir / f"{stem}_{args.mask_prefix}instances_color.png"
            Image.fromarray(combined_mask_rgb).save(out_mask)

        # Instance overlay
        if not args.no_overlay:
            out_viz = combined_dir / f"{stem}_{args.viz_prefix}overlay_instances.png"
            Image.fromarray(overlay_instances).save(out_viz)

        # Boundary overlay (existing)
        if not args.no_boundary:
            out_bnd = combined_dir / f"{stem}_{args.viz_prefix}overlay_boundaries.png"
            Image.fromarray(overlay_boundaries).save(out_bnd)

        # Boundary mask (new, boundary-only)
        out_bnd_mask = combined_dir / f"{stem}_{args.viz_prefix}boundary_mask.png"
        Image.fromarray(boundary_mask_rgb).save(out_bnd_mask)

    if (not args.dry_run) and (not args.no_per_instance):
        # ensure n_instances appears first in file (dict insertion order)
        meta = {
            "n_instances": n,
            "stem": stem,
            "image_path": str(ip),
            "mask_path": str(mp),
            "image_hw": [h_img, w_img],
            "mask_raw_shape": list(mask_raw.shape),
            "mask_norm_shape": [n, h_m, w_m],
            "colors_rgb_by_index": {str(i): list(map(int, colors[i])) for i in range(n)},
            "stack_unique_values": stack_u_list,
        }
        with (stem_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return record, flat_rows


# -------------------------
# Main
# -------------------------
def main() -> int:
    args = parse_args()
    log_path = setup_logging(args.out_dir, args.log_console_level, args.log_file_level)

    logger.info("Starting mask visualization + analysis")
    logger.info(f"images_dir={args.images_dir}")
    logger.info(f"masks_dir={args.masks_dir}")
    logger.info(f"out_dir={args.out_dir}")
    logger.info(f"log_file={log_path}")
    logger.info(f"dry_run={args.dry_run} workers={args.workers}")

    try:
        images = list_images(args.images_dir)
        masks = list_masks(args.masks_dir)
        pairs, orphans = build_pairs(images, masks)

        # Handle orphan masks according to policy (warn/ignore/error)
        for o in orphans:
            msg = f"Orphan mask found: stem={o.stem} mask={o.mask_path}"
            if args.on_orphan_mask == "ignore":
                logger.debug(msg)
            elif args.on_orphan_mask == "warn":
                logger.warning(msg)
            else:
                raise DatasetError(msg)

        # Report paths
        report_path, csv_flat_path = open_report_paths(args.out_dir, args.report_name, args.report_format)

        jsonl_fp = None
        flat_rows_all: List[Dict[str, Any]] = []
        if args.report_format == "jsonl":
            jsonl_fp = report_path.open("w", encoding="utf-8")

        processed = 0
        skipped = 0
        errors = 0

        # Threaded execution (safe logger)
        if args.workers < 1:
            raise ValueError("--workers must be >= 1")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pr in pairs:
                futures.append(ex.submit(process_one, pr, args))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
                try:
                    rec, flat_rows = fut.result()
                    if rec is None:
                        skipped += 1
                        continue
                    processed += 1

                    # Write report
                    if args.report_format == "jsonl":
                        assert jsonl_fp is not None
                        write_jsonl_line(jsonl_fp, rec)
                    else:
                        flat_rows_all.extend(flat_rows)

                except Exception:
                    errors += 1
                    logger.exception("Error during processing")
                    if args.on_read_error == "warn-skip" or args.on_mismatch == "skip" or args.on_missing == "skip":
                        # Allow continuing only if user chose skip-ish modes
                        continue
                    # Default behavior: stop (full trace already logged)
                    raise

        if jsonl_fp is not None:
            jsonl_fp.close()

        if args.report_format == "csv":
            # write flattened region CSV
            assert csv_flat_path is not None
            write_flat_csv(flat_rows_all, csv_flat_path)
            logger.info(f"Wrote CSV report: {csv_flat_path}")

        if args.report_format == "jsonl":
            logger.info(f"Wrote JSONL report: {report_path}")

        logger.info(f"Done. processed={processed} skipped={skipped} errors={errors}")
        return 0

    except Exception:
        logger.exception("Fatal error")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
