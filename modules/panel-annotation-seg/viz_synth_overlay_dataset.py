#!/usr/bin/env python3
"""
viz_synth_overlay_dataset.py

Visual QA for synthetic overlay datasets (e.g. *_ann masks).

Outputs (default):
- Triptych: image | mask | overlay
- Contours overlay: image with mask contours
- Ghost view: animated GIF blinking image <-> overlay (fallback to strip PNG if Pillow not available)
- Zoom-on-mask patches: montage of zoomed crops around mask components

Also writes JSON logs by default:
- <output>/_viz_logs/run_args.json
- <output>/_viz_logs/missing.json  (relative paths)

Example:
  python viz_synth_overlay_dataset.py \
    --input  ~/Desktop/projects/panel-annotation-seg/out/synth_ann/v2 \
    --output ~/Desktop/projects/panel-annotation-seg/out/synth_ann/v2_viz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Pair:
    image_path: Path
    mask_path: Optional[Path]
    reason: str


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def safe_imread(p: Path, flags=cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Robust imread (handles non-ascii paths on some OSes)."""
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return cv2.imread(str(p), flags)


def ensure_3ch_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def binarize_mask(mask: np.ndarray, thresh: int = 127) -> np.ndarray:
    """
    Convert mask image (possibly grayscale/RGB/RGBA) to binary uint8 {0,255}.
    - If RGBA and alpha has signal, uses alpha as mask.
    """
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            alpha = mask[:, :, 3]
            if alpha.max() > 0:
                gray = alpha
            else:
                gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    _, bw = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    return bw.astype(np.uint8)


def overlay_mask_on_image(
    img_bgr: np.ndarray,
    mask_bw: np.ndarray,
    alpha: float = 0.45,
    overlay_bgr: Tuple[int, int, int] = (0, 0, 255),  # red
) -> np.ndarray:
    img = img_bgr.copy()
    if mask_bw.shape[:2] != img.shape[:2]:
        mask_bw = cv2.resize(mask_bw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:] = overlay_bgr

    m = mask_bw > 0
    img[m] = (img[m].astype(np.float32) * (1.0 - alpha) + colored[m].astype(np.float32) * alpha).astype(np.uint8)
    return img


def draw_contours(
    img_bgr: np.ndarray,
    mask_bw: np.ndarray,
    thickness: int = 2,
    contour_bgr: Tuple[int, int, int] = (0, 255, 0),  # green
) -> np.ndarray:
    img = img_bgr.copy()
    if mask_bw.shape[:2] != img.shape[:2]:
        mask_bw = cv2.resize(mask_bw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, contour_bgr, int(thickness), lineType=cv2.LINE_AA)
    return img


def make_triptych(img_bgr: np.ndarray, mask_bw: np.ndarray, overlay_bgr: np.ndarray, mask_pct: float) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    if mask_bw.shape[:2] != (H, W):
        mask_bw = cv2.resize(mask_bw, (W, H), interpolation=cv2.INTER_NEAREST)

    mask_vis = cv2.cvtColor(mask_bw, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = img_bgr
    canvas[:, W:2 * W] = mask_vis
    canvas[:, 2 * W:3 * W] = overlay_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.0, W / 900.0))
    th = max(1, int(2 * scale))
    y = int(30 * scale)

    cv2.putText(canvas, "image", (10, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(canvas, "mask", (W + 10, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(canvas, "overlay", (2 * W + 10, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"mask%: {mask_pct*100:.2f}",
        (2 * W + 10, y + int(30 * scale)),
        font,
        scale,
        (255, 255, 255),
        th,
        cv2.LINE_AA,
    )
    return canvas


def write_image(out_path: Path, img_bgr: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext not in IMG_EXTS:
        out_path = out_path.with_suffix(".png")
        ext = ".png"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode: {out_path}")
    buf.tofile(str(out_path))


def endswith_any(stem: str, suffixes: Sequence[str]) -> Optional[str]:
    """If stem ends with any suffix (exact match at end), return matched suffix else None."""
    for s in suffixes:
        if stem.endswith(s):
            return s
    return None


def collect_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_image_file(p):
                files.append(p)
    return files


def try_write_ghost_gif(out_path: Path, img_bgr: np.ndarray, overlay_bgr: np.ndarray, fps: float = 2.0) -> bool:
    """
    Try saving an animated GIF blinking image <-> overlay using Pillow.
    Returns True if GIF written, else False.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ov_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    frames = [Image.fromarray(img_rgb), Image.fromarray(ov_rgb)]
    duration_ms = int(max(50, 1000.0 / max(0.1, fps)))

    # loop=0 => infinite loop
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return True


def make_ghost_strip(img_bgr: np.ndarray, overlay_bgr: np.ndarray) -> np.ndarray:
    """Fallback static ghost strip: image | overlay | image | overlay."""
    H, W = img_bgr.shape[:2]
    canvas = np.zeros((H, W * 4, 3), dtype=np.uint8)
    canvas[:, 0:W] = img_bgr
    canvas[:, W:2 * W] = overlay_bgr
    canvas[:, 2 * W:3 * W] = img_bgr
    canvas[:, 3 * W:4 * W] = overlay_bgr
    return canvas


def pad_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize keeping aspect so that max side == size, then pad to (size,size).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8) if img.ndim == 3 else np.zeros((size, size), dtype=np.uint8)

    scale = size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left

    if resized.ndim == 2:
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def make_patch_montage(
    img_bgr: np.ndarray,
    mask_bw: np.ndarray,
    overlay_bgr: np.ndarray,
    patch_size: int = 256,
    max_patches: int = 4,
    pad_frac: float = 0.20,
    min_area: int = 30,
) -> Optional[np.ndarray]:
    """
    Build a montage of zoomed-in crops around largest connected components in mask.
    Each row is: crop_image | crop_mask | crop_overlay (each square patch_size).
    Returns montage image or None if mask is empty.
    """
    if mask_bw.shape[:2] != img_bgr.shape[:2]:
        mask_bw = cv2.resize(mask_bw, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    if (mask_bw > 0).sum() == 0:
        return None

    # Connected components on binary mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_bw > 0).astype(np.uint8), connectivity=8)
    # stats: [label, x, y, w, h, area] but label 0 is background
    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            comps.append((area, x, y, w, h))
    if not comps:
        return None

    comps.sort(reverse=True, key=lambda t: t[0])
    comps = comps[:max_patches]

    rows = []
    H, W = img_bgr.shape[:2]
    for area, x, y, w, h in comps:
        pad = int(round(max(w, h) * pad_frac))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        crop_img = img_bgr[y0:y1, x0:x1]
        crop_mask = mask_bw[y0:y1, x0:x1]
        crop_ov = overlay_bgr[y0:y1, x0:x1]

        crop_mask_vis = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)

        a = pad_to_square(crop_img, patch_size)
        b = pad_to_square(crop_mask_vis, patch_size)
        c = pad_to_square(crop_ov, patch_size)

        row = np.concatenate([a, b, c], axis=1)  # (patch_size, patch_size*3, 3)

        # annotate area%
        area_pct = float(area) / float(H * W)
        cv2.putText(
            row,
            f"comp area%: {area_pct*100:.3f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(row)

    montage = np.concatenate(rows, axis=0)
    return montage


def rel_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Input root folder")
    ap.add_argument("--output", required=True, type=str, help="Output root folder")

    ap.add_argument("--mode", choices=["triptych", "contours", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")

    ap.add_argument("--mask-thresh", type=int, default=127)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--contour-thickness", type=int, default=2)

    ap.add_argument("--suffix-triptych", type=str, default="__triptych.png")
    ap.add_argument("--suffix-contours", type=str, default="__contours.png")

    # Mask pairing
    ap.add_argument(
        "--mask-suffixes",
        type=str,
        default="_ann,_mask,_anno,_annotation,_annotations,_label,_labels,_gt",
        help="Comma-separated suffixes that identify mask files by filename stem ending.",
    )

    # Default ON extras
    ap.add_argument("--ghost", dest="ghost", action="store_true", help="Enable ghost/blink outputs (default on)")
    ap.add_argument("--no-ghost", dest="ghost", action="store_false", help="Disable ghost/blink outputs")
    ap.set_defaults(ghost=True)
    ap.add_argument("--ghost-fps", type=float, default=2.0, help="Blink FPS for ghost GIF")

    ap.add_argument("--patches", dest="patches", action="store_true", help="Enable zoom-on-mask patches (default on)")
    ap.add_argument("--no-patches", dest="patches", action="store_false", help="Disable zoom-on-mask patches")
    ap.set_defaults(patches=True)
    ap.add_argument("--patch-size", type=int, default=256, help="Square patch size for montage tiles")
    ap.add_argument("--max-patches", type=int, default=4, help="Max connected components to show per image")
    ap.add_argument("--patch-pad-frac", type=float, default=0.20, help="BBox expansion as fraction of max(w,h)")
    ap.add_argument("--patch-min-area", type=int, default=30, help="Min component area (px) to keep")

    # Output suffixes for new artifacts
    ap.add_argument("--suffix-ghost-gif", type=str, default="__ghost.gif")
    ap.add_argument("--suffix-ghost-strip", type=str, default="__ghost_strip.png")
    ap.add_argument("--suffix-patches", type=str, default="__patches.png")

    # Default JSON logs
    ap.add_argument("--log-json", dest="log_json", action="store_true", help="Write JSON logs (default on)")
    ap.add_argument("--no-log-json", dest="log_json", action="store_false", help="Disable JSON logs")
    ap.set_defaults(log_json=True)

    args = ap.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.exists():
        raise FileNotFoundError(f"Input root not found: {in_root}")

    mask_suffixes = [s.strip() for s in args.mask_suffixes.split(",") if s.strip()]
    all_imgs = collect_images(in_root)

    # Split into masks vs base images by suffix rule (perfect for *_ann.png style)
    mask_files: List[Path] = []
    image_files: List[Path] = []
    for p in all_imgs:
        m = endswith_any(p.stem.lower(), mask_suffixes)
        if m is not None:
            mask_files.append(p)
        else:
            image_files.append(p)

    # base_stem -> mask_path
    mask_by_base: Dict[str, Path] = {}
    for mp in mask_files:
        stem = mp.stem
        matched = endswith_any(stem.lower(), mask_suffixes)
        if matched is None:
            continue
        base = stem[: -len(matched)]
        mask_by_base.setdefault(base, mp)

    pairs: List[Pair] = []
    for ip in image_files:
        mp = mask_by_base.get(ip.stem)
        if mp is not None:
            pairs.append(Pair(ip, mp, "stem+suffix_pair"))
        else:
            found = None
            for sfx in mask_suffixes:
                cand = ip.parent / f"{ip.stem}{sfx}{ip.suffix}"
                if cand.exists():
                    found = cand
                    break
            pairs.append(Pair(ip, found, "fallback_same_dir"))

    if args.max_images and args.max_images > 0:
        pairs = pairs[: args.max_images]

    print(f"[INFO] Scanned image-like files: {len(all_imgs)}")
    print(f"[INFO] Base images: {len(image_files)} | Mask candidates: {len(mask_files)}")
    print(f"[INFO] Pairs to process: {len(pairs)}")
    print(f"[INFO] ghost={args.ghost} patches={args.patches} log_json={args.log_json}")

    missing: List[dict] = []
    processed = 0
    ok_trip = ok_cont = ok_ghost = ok_patches = 0

    for pair in tqdm(pairs, desc="visualizing", unit="img"):
        processed += 1
        rel_img = pair.image_path.relative_to(in_root)
        base_out = out_root / rel_img.parent / pair.image_path.stem

        out_trip = Path(str(base_out) + args.suffix_triptych)
        out_cont = Path(str(base_out) + args.suffix_contours)
        out_ghost_gif = Path(str(base_out) + args.suffix_ghost_gif)
        out_ghost_strip = Path(str(base_out) + args.suffix_ghost_strip)
        out_patches = Path(str(base_out) + args.suffix_patches)

        # overwrite logic per artifact
        if not args.overwrite:
            # If user wants only triptych/contours, we can skip those individually.
            pass

        img = safe_imread(pair.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            missing.append({
                "type": "BAD_IMAGE",
                "image": rel_posix(pair.image_path, in_root),
                "mask": None,
                "reason": pair.reason,
            })
            continue
        img_bgr = ensure_3ch_bgr(img)

        if pair.mask_path is None or not pair.mask_path.exists():
            missing.append({
                "type": "NO_MASK",
                "image": rel_posix(pair.image_path, in_root),
                "mask": None,
                "reason": pair.reason,
            })
            continue

        mask_raw = safe_imread(pair.mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            missing.append({
                "type": "BAD_MASK",
                "image": rel_posix(pair.image_path, in_root),
                "mask": rel_posix(pair.mask_path, in_root),
                "reason": pair.reason,
            })
            continue

        try:
            mask_bw = binarize_mask(mask_raw, thresh=int(args.mask_thresh))
        except Exception as e:
            missing.append({
                "type": "MASK_FAIL",
                "image": rel_posix(pair.image_path, in_root),
                "mask": rel_posix(pair.mask_path, in_root),
                "reason": f"{pair.reason}; err={str(e)}",
            })
            continue

        mask_pct = float((mask_bw > 0).mean())
        overlay = overlay_mask_on_image(img_bgr, mask_bw, alpha=float(args.overlay_alpha))
        cont = draw_contours(img_bgr, mask_bw, thickness=int(args.contour_thickness))

        # Triptych / contours
        if args.mode in ("triptych", "both"):
            if args.overwrite or (not out_trip.exists()):
                trip = make_triptych(img_bgr, mask_bw, overlay, mask_pct)
                write_image(out_trip, trip)
            ok_trip += 1

        if args.mode in ("contours", "both"):
            if args.overwrite or (not out_cont.exists()):
                write_image(out_cont, cont)
            ok_cont += 1

        # Ghost (default on)
        if args.ghost:
            if args.overwrite or ((not out_ghost_gif.exists()) and (not out_ghost_strip.exists())):
                wrote_gif = try_write_ghost_gif(out_ghost_gif, img_bgr, overlay, fps=float(args.ghost_fps))
                if not wrote_gif:
                    # fallback strip
                    strip = make_ghost_strip(img_bgr, overlay)
                    write_image(out_ghost_strip, strip)
            ok_ghost += 1

        # Zoom-on-mask patches (default on)
        if args.patches:
            if args.overwrite or (not out_patches.exists()):
                montage = make_patch_montage(
                    img_bgr=img_bgr,
                    mask_bw=mask_bw,
                    overlay_bgr=overlay,
                    patch_size=int(args.patch_size),
                    max_patches=int(args.max_patches),
                    pad_frac=float(args.patch_pad_frac),
                    min_area=int(args.patch_min_area),
                )
                if montage is not None:
                    write_image(out_patches, montage)
            ok_patches += 1

    # JSON logs (default on)
    if args.log_json:
        logs_dir = out_root / "_viz_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        run_info = {
            "timestamp_local": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "input_root": str(in_root),
            "output_root": str(out_root),
            "args": vars(args),
            "counts": {
                "scanned_image_like_files": len(all_imgs),
                "base_images": len(image_files),
                "mask_candidates": len(mask_files),
                "pairs": len(pairs),
                "processed": processed,
                "ok_triptych": ok_trip,
                "ok_contours": ok_cont,
                "ok_ghost": ok_ghost,
                "ok_patches": ok_patches,
                "missing_count": len(missing),
            },
        }
        (logs_dir / "run_args.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        (logs_dir / "missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    print(f"[DONE] Input:  {in_root}")
    print(f"[DONE] Output: {out_root}")
    print(f"[DONE] processed={processed} missing={len(missing)}")
    if args.log_json:
        print(f"[DONE] logs: {out_root / '_viz_logs' / 'run_args.json'}")
        print(f"[DONE] logs: {out_root / '_viz_logs' / 'missing.json'}")

if __name__ == "__main__":
    main()
