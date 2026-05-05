#!/usr/bin/env python
"""
Batch extract mask PNGs from .npy mask files and create overlays.

Inputs
------
- image_dir: folder with forged images (e.g., .png / .jpg)
- mask_dir : folder with .npy mask files (same stem as images)
- out_dir  : output root folder

For each mask file "<stem>.npy" it creates:

out_dir/<stem>/
    <stem>_image.png               # RGB copy of forged image
    <stem>_overlay.png             # filled color overlay
    <stem>_overlay_boundary.png    # boundary-only overlay
    <stem>_mask_01.png             # binary mask for channel 1
    <stem>_mask_02.png             # binary mask for channel 2
    ...

Notes
-----
- .npy mask can be:
    (H, W)          single mask
    (C, H, W)       multi-channel
    (H, W, C)       multi-channel
- Any non-zero value is treated as 1 (binary).
- Masks are resized to match image size (nearest neighbor).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
import cv2  # pip install opencv-python


# ---------------------------
# I/O helpers
# ---------------------------

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_image_rgb(path: Path) -> Image.Image:
    """Read image as RGB PIL Image."""
    img = Image.open(path).convert("RGB")
    return img


def load_npy_mask(path: Path) -> np.ndarray:
    """Load .npy mask as numpy array."""
    arr = np.load(path, allow_pickle=True)
    return np.array(arr)


def to_chw(mask: np.ndarray) -> np.ndarray:
    """
    Normalize mask array to (C, H, W).

    Accepted shapes:
        (H, W)
        (C, H, W)
        (H, W, C)
    Any extra singleton dimensions are squeezed.
    """
    a = np.array(mask)
    # Remove trivial singleton dims
    while a.ndim > 2 and 1 in a.shape:
        a = a.squeeze()

    if a.ndim == 2:
        return a[None, ...]  # (1, H, W)

    if a.ndim == 3:
        # Heuristic: if first dim is small and the other two are "image-like",
        # treat as (C, H, W); otherwise assume (H, W, C).
        if a.shape[0] <= 4 and a.shape[1] > 8 and a.shape[2] > 8:
            return a  # (C, H, W)
        else:
            return a.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Fallback: squeeze and retry
    a = a.squeeze()
    if a.ndim == 2:
        return a[None, ...]
    if a.ndim == 3:
        return a
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def resize_binary(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a binary mask (0/1 or 0/255) to (width, height) using NEAREST.
    Returns uint8 mask with values {0,1}.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    pil = Image.fromarray(mask_u8)
    if pil.size != size:
        pil = pil.resize(size, resample=Image.NEAREST)
    out = (np.array(pil) > 127).astype(np.uint8)
    return out


def find_matching_image(mask_path: Path, image_dir: Path) -> Path | None:
    """Find the image file in `image_dir` with same stem as mask."""
    stem = mask_path.stem
    for ext in VALID_EXTS:
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


# ---------------------------
# Overlay and boundary drawing
# ---------------------------

def generate_colors(n: int) -> List[np.ndarray]:
    """Generate a list of distinct RGB colors."""
    base_colors = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 200, 0),
        (255, 200, 0),
        (180, 0, 200),
        (255, 105, 180),
        (0, 255, 255),
    ]
    colors = [np.array(c, dtype=np.uint8) for c in base_colors]
    # If more masks than base colors, just cycle
    if n <= len(colors):
        return colors[:n]
    else:
        extra = n - len(colors)
        rng = np.random.default_rng(42)
        for _ in range(extra):
            colors.append(rng.integers(0, 256, size=3, dtype=np.uint8))
        return colors


def overlay_masks(
    image: Image.Image,
    masks: Iterable[np.ndarray],
    alpha_fill: float = 0.35,
    alpha_edge: float = 0.95,
) -> Tuple[Image.Image, Image.Image]:
    """
    Create filled overlay and boundary-only overlay.

    Parameters
    ----------
    image : PIL.Image
        RGB image.
    masks : iterable of np.ndarray
        Each mask is (H, W) uint8, values {0,1}.
    alpha_fill : float
        Transparency for filled overlay.
    alpha_edge : float
        Transparency for boundary overlay.

    Returns
    -------
    overlay_filled : PIL.Image
    overlay_edge   : PIL.Image
    """
    base = np.array(image).astype(np.uint8)
    overlay = base.copy()
    edge_overlay = base.copy()

    masks = list(masks)
    colors = generate_colors(len(masks))

    kernel = np.ones((3, 3), np.uint8)

    for m, color in zip(masks, colors):
        if m is None:
            continue
        mask_bool = m.astype(bool)
        if not mask_bool.any():
            continue

        # --- filled overlay ---
        idx = mask_bool
        overlay[idx] = (
            alpha_fill * color + (1.0 - alpha_fill) * overlay[idx]
        ).astype(np.uint8)

        # --- boundary overlay ---
        m_u8 = (m > 0).astype(np.uint8) * 255
        dil = cv2.dilate(m_u8, kernel, iterations=1)
        ero = cv2.erode(m_u8, kernel, iterations=1)
        edge = (dil - ero) > 0
        eidx = edge.astype(bool)
        edge_overlay[eidx] = (
            alpha_edge * color + (1.0 - alpha_edge) * edge_overlay[eidx]
        ).astype(np.uint8)

    return Image.fromarray(overlay), Image.fromarray(edge_overlay)


# ---------------------------
# Main processing
# ---------------------------

def process_one(mask_path: Path, image_dir: Path, out_dir: Path) -> None:
    """Process a single .npy mask file."""
    img_path = find_matching_image(mask_path, image_dir)
    if img_path is None:
        print(f"[WARN] No matching image for {mask_path.name}, skipping.")
        return

    img = read_image_rgb(img_path)
    W, H = img.size

    mask_raw = load_npy_mask(mask_path)
    chw = to_chw(mask_raw)

    # Resize each channel to image size and binarize
    masks: List[np.ndarray] = [
        resize_binary(ch, (W, H)) for ch in chw
    ]

    stem = mask_path.stem
    out_subdir = out_dir / stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Save original forged image copy
    img.save(out_subdir / f"{stem}_image.png")

    # Save individual masks
    for i, m in enumerate(masks, start=1):
        out_mask = out_subdir / f"{stem}_mask_{i:02d}.png"
        Image.fromarray(m * 255).save(out_mask)

    # Create overlays
    overlay, edge_overlay = overlay_masks(img, masks)
    overlay.save(out_subdir / f"{stem}_overlay.png")
    edge_overlay.save(out_subdir / f"{stem}_overlay_boundary.png")

    print(
        f"[OK] {mask_path.name} -> {len(masks)} mask(s) "
        f"saved in {out_subdir}"
    )


def run(image_dir: Path, mask_dir: Path, out_dir: Path) -> None:
    mask_paths = sorted(mask_dir.glob("*.npy"))
    if not mask_paths:
        print(f"[WARN] No .npy files found in {mask_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for mp in mask_paths:
        process_one(mp, image_dir=image_dir, out_dir=out_dir)


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract PNG masks and overlays from .npy mask files."
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory with forged images.",
    )
    p.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory with .npy mask files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root directory.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image_dir, args.mask_dir, args.out_dir)
