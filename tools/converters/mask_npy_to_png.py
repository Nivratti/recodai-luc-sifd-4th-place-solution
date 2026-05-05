#!/usr/bin/env python3
"""
Mask .npy/.npz -> PNGs (multi + combined) with strict size validation
=====================================================================

- Recursively scans a MASKS ROOT for .npy/.npz files.
- Extracts ALL 2D masks (no cap):
    * SQUEEZE FIRST: (1,H,W)/(H,W,1) -> (H,W)
    * If 2D -> binary mask (nonzero -> foreground). No label splitting.
    * If 3D -> detect channel axis; split along that axis (K planes).
      - If forged image size (H_img,W_img) known, prefer axis whose slice matches it.
- STRICT size check vs paired forged image by default (mismatch → SKIP saving).
  Use --allow-size-mismatch to save anyway (WARN).
- Saves **both** outputs in one run under --output-root:
    <output-root>/
      multi/<mirrored path>/<stem>_1.png, <stem>_2.png, ...
      combined/<mirrored path>/<stem>.png  (logical OR of masks that match expected size)
- Logs to console AND <output-root>/logs/mask_png_export.log (Loguru).
- Writes CSV summary at <output-root>/summaries/mask_export_summary.csv

Pairing rule:
- Match mask stem to a forged image with the same stem under --images-root.

Example (Windows):
    python scripts\\data_preparation\\converters\\mask_npy_to_png.py ^
      --images-root "D:\\datasets\\recodai-luc-scientific-image-forgery-detection\\raw\\train_images\\forged" ^
      --masks-root  "D:\\datasets\\recodai-luc-scientific-image-forgery-detection\\raw\\train_masks" ^
      --output-root "D:\\datasets\\recodai-luc-scientific-image-forgery-detection\\derived\\masks_from_npy" ^
      --skip-existing

To allow saving despite size mismatch:
      --allow-size-mismatch
"""

from __future__ import annotations
import argparse
import sys
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

# --------------------------- Logging ---------------------------

def setup_logging(out_root: Path) -> None:
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mask_png_export.log"
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | <cyan>{message}</cyan>")
    logger.add(log_file, level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
               rotation="10 MB", retention=10, compression="zip")
    logger.debug(f"Logging to: {log_file}")

# --------------------------- Discovery ---------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MASK_EXTS = {".npy", ".npz"}

def iter_mask_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            yield p

def index_images(images_root: Path) -> Dict[str, Path]:
    """Build a stem->path index for pairing forged images quickly."""
    idx: Dict[str, Path] = {}
    for p in images_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        if stem in idx and idx[stem] != p:
            logger.warning(f"[IMG-INDEX] duplicate stem '{stem}': {idx[stem]} AND {p} (keeping first)")
            continue
        idx[stem] = p
    logger.info(f"[IMG-INDEX] indexed {len(idx)} images under: {images_root}")
    return idx

# --------------------------- Loading ---------------------------

def load_mask_container(path: Path) -> Any:
    try:
        if path.suffix.lower() == ".npz":
            with np.load(str(path), allow_pickle=True) as z:
                return {k: z[k] for k in z.files}
        else:
            try:
                return np.load(str(path), allow_pickle=False)
            except Exception:
                return np.load(str(path), allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Load failed: {e}") from e

def summarize_keys(obj: Any, suffix: str) -> Optional[str]:
    if suffix == ".npz" and isinstance(obj, dict):
        ks = list(obj.keys())
        if not ks: return "npz: (no entries)"
        return "npz keys: " + ", ".join(ks[:10]) + (f" (+{len(ks)-10} more)" if len(ks)>10 else "")
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return summarize_keys(obj.item(), suffix)
    if isinstance(obj, dict):
        ks = list(map(str, obj.keys()))
        if ks: return "dict keys: " + ", ".join(ks[:15]) + (f" (+{len(ks)-15} more)" if len(ks)>15 else "")
        return "dict: (no keys)"
    if isinstance(obj, (list, tuple)) and obj:
        u = set()
        for it in obj[:5]:
            if isinstance(it, dict): u.update(map(str, it.keys()))
        if u:
            ks = sorted(list(u))
            return "list-of-dicts union keys: " + ", ".join(ks[:15]) + (f" (+{len(ks)-15} more)" if len(ks)>15 else "")
    return None

# --------------------------- Extraction helpers ---------------------------

MASK_KEYS = {"mask","masks","m","seg","gt_mask","target","instances","mask_list"}

def squeeze_all(a: np.ndarray) -> np.ndarray:
    return np.squeeze(a)

def ensure_bool(a: np.ndarray, float_thresh: float = 0.5) -> np.ndarray:
    if np.issubdtype(a.dtype, np.floating): return a > float_thresh
    return a != 0

def detect_channel_axis(a: np.ndarray,
                        expected_hw: Optional[Tuple[int,int]]) -> Optional[int]:
    """
    Decide which axis is channels.
      1) If expected_hw=(H,W) known, prefer axis whose slice matches (H,W).
      2) Else choose smallest axis if slicing yields 2D.
      3) None if nothing reasonable is found.
    """
    if a.ndim != 3:
        return None
    if expected_hw is not None:
        H_exp, W_exp = expected_hw
        for axis in (0, 2, 1):  # prefer channel-first or last
            try:
                sl = np.squeeze(np.take(a, 0, axis=axis))
                if sl.ndim == 2 and sl.shape == (H_exp, W_exp):
                    return axis
            except Exception:
                pass
    # fallback: smallest axis that yields 2D slices
    sizes = list(a.shape)
    axis = min(range(3), key=lambda i: sizes[i])
    try:
        sl = np.squeeze(np.take(a, 0, axis=axis))
        if sl.ndim == 2:
            return axis
    except Exception:
        return None
    return axis

def extract_from_obj(obj: Any,
                     expected_hw: Optional[Tuple[int,int]],
                     decisions: List[str]) -> List[np.ndarray]:
    """Return ALL 2D boolean masks (no cap). Nonzero -> foreground. No label splitting."""
    # unwrap object scalar
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return extract_from_obj(obj.item(), expected_hw, decisions)

    # ndarray
    if isinstance(obj, np.ndarray):
        orig = obj.shape
        if obj.ndim == 0:
            raise ValueError("unsupported 0-D array")
        a = squeeze_all(obj)              # <--- critical: squeeze first
        sq = a.shape

        if a.ndim == 2:
            decisions.append(f"ndarray {orig} -> squeezed {sq}: 2D binary(nonzero)")
            return [ensure_bool(a)]

        if a.ndim == 3:
            ch_axis = detect_channel_axis(a, expected_hw=expected_hw)
            if ch_axis is None:
                decisions.append(f"ndarray {orig} -> squeezed {sq}: 3D no channel axis; using first slice of smallest axis")
                axis = min(range(3), key=lambda i: a.shape[i])
                plane = np.squeeze(np.take(a, 0, axis=axis))
                return [ensure_bool(plane)] if plane.ndim == 2 else []
            K = a.shape[ch_axis]
            decisions.append(f"ndarray {orig} -> squeezed {sq}: 3D split axis={ch_axis} (K={K})")
            planes = [np.squeeze(np.take(a, i, axis=ch_axis)) for i in range(K)]
            return [ensure_bool(p) for p in planes if p.ndim == 2]

        raise ValueError(f"Unsupported ndarray shape after squeeze: {sq}")

    # dict
    if isinstance(obj, dict):
        for k in ["masks","mask_list","instances"]:
            if k in obj:
                decisions.append(f"dict contains '{k}' -> recurse")
                return extract_from_obj(obj[k], expected_hw, decisions)
        for k in MASK_KEYS:
            if k in obj:
                decisions.append(f"dict contains '{k}' -> recurse")
                return extract_from_obj(obj[k], expected_hw, decisions)
        for v in obj.values():  # best-effort
            try:
                ms = extract_from_obj(v, expected_hw, decisions)
                if ms:
                    return ms
            except Exception:
                pass
        raise ValueError("No recognizable mask keys in dict")

    # list/tuple
    if isinstance(obj, (list, tuple)):
        out: List[np.ndarray] = []
        for it in obj:
            out.extend(extract_from_obj(it, expected_hw, decisions))
        if out:
            decisions.append("list/tuple -> concatenated child masks")
            return out
        raise ValueError("Empty or unrecognized list/tuple content")

    raise ValueError(f"Unrecognized container: {type(obj)}")

# --------------------------- Saving & pairing ---------------------------

def save_multi_pngs(masks: List[np.ndarray], out_base: Path) -> List[Path]:
    """Save multiple masks as <stem>_1.png, <stem>_2.png, ..."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    if len(masks) == 1:
        p = out_base.with_suffix(".png")
        Image.fromarray((masks[0].astype(np.uint8) * 255), mode="L").save(p)
        saved.append(p)
        return saved
    for i, m in enumerate(masks, 1):
        p = out_base.parent / f"{out_base.stem}_{i}.png"
        Image.fromarray((m.astype(np.uint8) * 255), mode="L").save(p)
        saved.append(p)
    return saved

def save_combined_png(masks: List[np.ndarray], out_png: Path) -> Optional[Path]:
    """Logical-OR combine masks (assumes all same shape) and save as <stem>.png"""
    if not masks:
        return None
    H, W = masks[0].shape
    combo = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m.shape != (H, W):
            logger.error(f"[COMBINED] shape mismatch within file for combined: {(H,W)} vs {m.shape}")
            return None
        combo |= m
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((combo.astype(np.uint8) * 255), mode="L").save(out_png)
    return out_png

def load_image_hw(img_path: Path) -> Tuple[int,int]:
    with Image.open(img_path) as im:
        w, h = im.size
    return (h, w)

# --------------------------- Pipeline ---------------------------

def process_one_file(mask_path: Path,
                     masks_root: Path,
                     out_root: Path,
                     img_index: Dict[str, Path],
                     skip_existing: bool,
                     allow_size_mismatch: bool) -> Dict[str, Any]:
    """Extract ALL masks; save multi + combined; strict size check by default."""
    rel = mask_path.relative_to(masks_root)
    # two out bases
    multi_base = (out_root / "multi" / rel).with_suffix("")
    comb_png  =  (out_root / "combined" / rel).with_suffix(".png")

    # Pair image by stem
    stem = mask_path.stem
    img_path = img_index.get(stem)
    H_exp: Optional[int] = None
    W_exp: Optional[int] = None
    img_status = "no_image"
    if img_path:
        try:
            H_exp, W_exp = load_image_hw(img_path)
            img_status = "found"
        except Exception as e:
            img_status = f"image_load_error: {e}"

    # Skip existing (only if BOTH outputs exist)
    if skip_existing:
        multi_exists = multi_base.with_suffix(".png").exists() or any(multi_base.parent.glob(f"{multi_base.stem}_*.png"))
        comb_exists  = comb_png.exists()
        if multi_exists and comb_exists:
            logger.info(f"[SKIP] {mask_path} -> multi & combined outputs already exist")
            return dict(mask_file=str(mask_path), image_file=str(img_path) if img_path else "",
                        status="skipped_existing", num_masks=0, saved_multi=0, saved_combined=0,
                        shape="", expected_hw=(H_exp, W_exp), image_status=img_status,
                        size_status="", decision="")

    # Load mask container + keys
    obj = load_mask_container(mask_path)
    key_info = summarize_keys(obj, mask_path.suffix.lower())
    if key_info:
        logger.info(f"[KEYS] {mask_path.name}: {key_info}")

    # Extract ALL masks
    decisions: List[str] = []
    masks = extract_from_obj(obj, expected_hw=(H_exp, W_exp) if (H_exp and W_exp) else None,
                             decisions=decisions)

    # Validate shapes vs expected image size (strict by default)
    size_status = "unknown"
    first_shape = ""
    saved_multi = 0
    saved_combined = 0

    if not masks:
        logger.warning(f"[EMPTY] {mask_path.name}: no usable 2D masks extracted")
        return dict(mask_file=str(mask_path), image_file=str(img_path) if img_path else "",
                    status="no_masks", num_masks=0, saved_multi=0, saved_combined=0,
                    shape="", expected_hw=(H_exp, W_exp), image_status=img_status,
                    size_status="n/a", decision=" | ".join(decisions[:12]))

    h0, w0 = masks[0].shape
    first_shape = f"{h0}x{w0}"

    # If we know expected size, enforce (strict) unless allow_size_mismatch
    in_shape_masks: List[np.ndarray] = masks
    if H_exp and W_exp:
        mismatches = [(m.shape[0] != H_exp or m.shape[1] != W_exp) for m in masks]
        if any(mismatches):
            size_status = f"size_mismatch_expected={H_exp}x{W_exp}"
            msg = f"[MISMATCH] {mask_path.name}: expected {H_exp}x{W_exp}, got {[m.shape for m in masks]}"
            if allow_size_mismatch:
                logger.warning(msg + " -> saving anyway (--allow-size-mismatch)")
                # For combined: prefer only masks that match expected size
                in_shape_masks = [m for m in masks if m.shape == (H_exp, W_exp)]
                if not in_shape_masks:
                    logger.warning("[COMBINED] no in-shape masks to combine; skipping combined output")
            else:
                logger.error(msg + " -> SKIP (strict)")
                return dict(mask_file=str(mask_path), image_file=str(img_path) if img_path else "",
                            status="size_mismatch_skipped", num_masks=len(masks),
                            saved_multi=0, saved_combined=0, shape=first_shape,
                            expected_hw=(H_exp, W_exp), image_status=img_status,
                            size_status=size_status, decision=" | ".join(decisions[:12]))
        else:
            size_status = "size_ok"
            in_shape_masks = masks
    else:
        size_status = "no_image_size"
        # For combined, use equality to first mask’s shape
        shp = masks[0].shape
        in_shape_masks = [m for m in masks if m.shape == shp]

    # Save MULTI
    saved_multi_paths = save_multi_pngs(masks, multi_base)
    saved_multi = len(saved_multi_paths)

    # Save COMBINED (only if we have at least one in-shape mask)
    comb_saved_path = None
    if in_shape_masks:
        comb_saved_path = save_combined_png(in_shape_masks, comb_png)
        saved_combined = 1 if comb_saved_path else 0
    else:
        saved_combined = 0

    logger.info(f"[OK] {mask_path.name}: masks={len(masks)} "
                f"saved_multi={saved_multi} saved_combined={saved_combined} "
                f"img_status={img_status} size_status={size_status}")

    return dict(mask_file=str(mask_path),
                image_file=str(img_path) if img_path else "",
                status="ok",
                num_masks=len(masks),
                saved_multi=saved_multi,
                saved_combined=saved_combined,
                shape=first_shape,
                expected_hw=(H_exp, W_exp),
                image_status=img_status,
                size_status=size_status,
                decision=" | ".join(decisions[:12]))

def main():
    ap = argparse.ArgumentParser(description="Export ALL masks (multi + combined) from .npy/.npz with strict size validation.")
    ap.add_argument("--images-root", type=Path, required=True, help="Root of forged images (for size validation)")
    ap.add_argument("--masks-root",  type=Path, required=True, help="Root of .npy/.npz masks")
    ap.add_argument("--output-root", type=Path, required=True, help="Output root up to 'masks_from_npy' (script writes multi/ and combined/ subfolders)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if both outputs already exist")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="Save even when mask size != paired image size (default is strict skip)")
    args = ap.parse_args()

    if not args.images_root.exists() or not args.images_root.is_dir():
        print(f"ERROR: --images-root not a directory: {args.images_root}", file=sys.stderr)
        sys.exit(2)
    if not args.masks_root.exists() or not args.masks_root.is_dir():
        print(f"ERROR: --masks-root not a directory: {args.masks_root}", file=sys.stderr)
        sys.exit(2)

    # Prepare out folders
    (args.output_root / "multi").mkdir(parents=True, exist_ok=True)
    (args.output_root / "combined").mkdir(parents=True, exist_ok=True)
    (args.output_root / "summaries").mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_root)

    # Build image index once
    img_index = index_images(args.images_root)

    # Process
    rows: List[Dict[str, Any]] = []
    total_files = total_masks = total_saved_multi = total_saved_comb = failures = 0

    for mask_path in iter_mask_files(args.masks_root):
        try:
            res = process_one_file(mask_path, args.masks_root, args.output_root, img_index,
                                   skip_existing=args.skip_existing,
                                   allow_size_mismatch=args.allow_size_mismatch)
            rows.append(res)
            total_files += 1
            total_masks += int(res.get("num_masks", 0))
            total_saved_multi += int(res.get("saved_multi", 0))
            total_saved_comb  += int(res.get("saved_combined", 0))
        except Exception as e:
            failures += 1
            logger.exception(f"[FAIL] {mask_path} -> {e}")

    # CSV summary
    csv_path = args.output_root / "summaries" / "mask_export_summary.csv"
    fieldnames = ["mask_file","image_file","status","num_masks","saved_multi","saved_combined",
                  "shape","expected_hw","image_status","size_status","decision"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    logger.info(f"Done. files={total_files} masks={total_masks} "
                f"saved_multi={total_saved_multi} saved_combined={total_saved_comb} failures={failures}")
    logger.info(f"Summary CSV: {csv_path}")

if __name__ == "__main__":
    main()
