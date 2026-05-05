#!/usr/bin/env python3
"""
make_cbir_aug_positives.py

Generate augmentation-positive variants for CBIR evaluation.

For each real panel image I:
  - create N variants I' using configurable transforms:
    * random crop + resize back
    * random resample (downscale+upscale)
    * rotation in {0,90,180,270}
    * flips (H/V)
    * gaussian blur
    * JPEG compression artifacts (in-memory)
    * gamma + contrast
    * gaussian noise
  - save variants under one folder per image
  - write per-image metadata JSON (what transforms were applied + params)
  - write global manifest JSONL for easy later evaluation

Output layout:
  out_root/
    manifest.jsonl
    summary.json
    <stem>__<hash8>/
      orig.png
      meta.json
      v001__rot90__flipH__jpegq70.png
      v002__crop0.85__gamma0.92__noise3.0.png
      ...

Example:
  python make_cbir_aug_positives.py \
    --in_dir /data/panels \
    --out_dir out/aug_pos \
    --n_variants 8 \
    --resize_mode letterpad \
    --img_size 512 \
    --p_crop 0.65 --crop_min 0.70 --crop_max 0.95 \
    --p_rot 0.35 --rot_choices 0,90,180,270 \
    --p_flip 0.35 --flip_modes h,v \
    --p_jpeg 0.35 --jpeg_qmin 35 --jpeg_qmax 85 \
    --p_gamma 0.35 --gamma_min 0.80 --gamma_max 1.25 \
    --p_contrast 0.35 --contrast_min 0.80 --contrast_max 1.25 \
    --p_blur 0.25 --blur_min 0.4 --blur_max 1.6 \
    --p_noise 0.25 --noise_min 1.0 --noise_max 6.0 \
    --seed 0

Notes:
- This script is ONLY for generating augmented data; it does NOT run CBIR.
- Uses Loguru for logging.
- Final outputs are PNG by default (to avoid adding extra compression),
  but JPEG artifacts are injected in-memory when enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from PIL import Image, ImageEnhance, ImageFilter
import re

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


IMG_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Config
# -----------------------------
@dataclass
class AugConfig:
    # output / sizing
    out_ext: str = ".png"                 # final saved file extension
    grayscale: bool = False               # convert to L then back to RGB (3-ch)
    resize_mode: str = "none"             # "none" | "resize" | "letterpad"
    img_size: Optional[int] = None        # required if resize_mode != "none"
    letterpad_fill: int = 0               # padding fill value 0..255

    # variant generation
    n_variants: int = 8
    ensure_nontrivial: bool = True        # force at least 1 transform if none sampled

    # transform probabilities
    p_crop: float = 0.60
    p_resample: float = 0.35
    p_rot: float = 0.30
    p_flip: float = 0.30
    p_blur: float = 0.20
    p_jpeg: float = 0.30
    p_gamma: float = 0.30
    p_contrast: float = 0.30
    p_noise: float = 0.25

    # crop params (crop fraction relative to original, keep aspect ratio)
    crop_min: float = 0.70
    crop_max: float = 0.95

    # resample params (downscale+upscale factor range)
    resample_min: float = 0.50
    resample_max: float = 0.90
    resample_methods: Tuple[str, ...] = ("bilinear", "bicubic", "lanczos", "nearest")

    # rotation params
    rot_choices: Tuple[int, ...] = (0, 90, 180, 270)

    # flip params: subset of {"h","v"}
    flip_modes: Tuple[str, ...] = ("h", "v")

    # blur params (Gaussian blur radius)
    blur_min: float = 0.3
    blur_max: float = 1.8

    # JPEG compression quality range
    jpeg_qmin: int = 35
    jpeg_qmax: int = 85

    # gamma params
    gamma_min: float = 0.80
    gamma_max: float = 1.25

    # contrast params
    contrast_min: float = 0.80
    contrast_max: float = 1.25

    # noise params (Gaussian noise sigma in pixel space 0..255)
    noise_min: float = 1.0
    noise_max: float = 6.0


# -----------------------------
# Utilities
# -----------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_short(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def discover_images(in_dir: Path, recursive: bool, exts: Sequence[str]) -> List[Path]:
    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    it = in_dir.rglob("*") if recursive else in_dir.glob("*")
    out: List[Path] = []
    for p in it:
        if p.is_file() and p.suffix.lower() in exts_set:
            out.append(p)
    out.sort()
    return out


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.copy()
    except Exception as e:
        logger.warning(f"Failed to read: {path} | {type(e).__name__}: {e}")
        return None


def to_rgb(im: Image.Image, grayscale: bool) -> Image.Image:
    if grayscale:
        return im.convert("L").convert("RGB")
    return im.convert("RGB")


def letterpad_square(im: Image.Image, size: int, fill: int = 0) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), (fill, fill, fill))
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = im.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def direct_resize(im: Image.Image, size: int) -> Image.Image:
    return im.resize((size, size), resample=Image.BILINEAR)


def apply_final_resize_policy(im: Image.Image, cfg: AugConfig) -> Image.Image:
    if cfg.resize_mode == "none":
        return im
    if cfg.img_size is None:
        raise ValueError("img_size is required when resize_mode != 'none'")
    if cfg.resize_mode == "resize":
        return direct_resize(im, cfg.img_size)
    if cfg.resize_mode == "letterpad":
        return letterpad_square(im, cfg.img_size, fill=int(cfg.letterpad_fill))
    raise ValueError(f"Unknown resize_mode: {cfg.resize_mode}")


def pil_to_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im).astype(np.float32)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def sanitize_name(s: str) -> str:
    # keep it filesystem friendly
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:160] if len(s) > 160 else s

def make_output_dir_for_image(src_path: Path, in_dir: Path, out_dir: Path) -> tuple[Path, str]:
    """
    Returns (img_out_dir, source_rel_str).

    Preserves input folder structure:
      in_dir/a/b/panel.png -> out_dir/a/b/panel__png/

    Adds suffix only if the folder already exists (collision).
    """
    try:
        rel = src_path.relative_to(in_dir)
        rel_str = rel.as_posix()
        rel_parent = rel.parent
    except Exception:
        # if src_path is outside in_dir, fall back to a flat "external" area
        rel_str = str(src_path)
        rel_parent = Path("_external")

    # sanitize each path component
    safe_parent = Path(*[sanitize_name(p) for p in rel_parent.parts if p not in ("", ".")])

    stem = sanitize_name(src_path.stem)
    ext = sanitize_name(src_path.suffix.lower().lstrip(".")) or "img"

    base = out_dir / safe_parent / f"{stem}__{ext}"

    # collision handling (only if needed)
    cand = base
    if cand.exists():
        for k in range(1, 200):
            suf = sha1_short(f"{rel_str}|{k}", n=6)
            cand2 = Path(str(base) + f"__{suf}")
            if not cand2.exists():
                cand = cand2
                break

    ensure_dir(cand)
    return cand, rel_str

# -----------------------------
# Transforms
# -----------------------------
def t_random_crop_keep_aspect(
    im: Image.Image, rng: random.Random, frac_min: float, frac_max: float
) -> Tuple[Image.Image, Dict[str, Any], str]:
    w, h = im.size
    frac = rng.uniform(frac_min, frac_max)
    # keep aspect ratio
    crop_w = max(1, int(round(w * frac)))
    crop_h = max(1, int(round(h * frac)))
    if crop_w >= w and crop_h >= h:
        # no-op crop
        return im, {"frac": round(frac, 4), "noop": True}, f"crop{frac:.2f}"
    x0 = rng.randint(0, w - crop_w)
    y0 = rng.randint(0, h - crop_h)
    cropped = im.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    # resize back to original size (introduces artifacts like typical reuse edits)
    resized = cropped.resize((w, h), resample=Image.BILINEAR)
    meta = {"frac": round(frac, 4), "x0": x0, "y0": y0, "w": crop_w, "h": crop_h, "resample": "bilinear"}
    tag = f"crop{frac:.2f}"
    return resized, meta, tag


_RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def t_resample_down_up(
    im: Image.Image, rng: random.Random, smin: float, smax: float, methods: Sequence[str]
) -> Tuple[Image.Image, Dict[str, Any], str]:
    w, h = im.size
    scale = rng.uniform(smin, smax)
    m = rng.choice(list(methods))
    res_m = _RESAMPLE_MAP.get(m, Image.BILINEAR)
    dw = max(1, int(round(w * scale)))
    dh = max(1, int(round(h * scale)))
    down = im.resize((dw, dh), resample=res_m)
    up = down.resize((w, h), resample=res_m)
    meta = {"scale": round(scale, 4), "method": m, "dw": dw, "dh": dh}
    tag = f"rs{scale:.2f}{m[0]}"
    return up, meta, tag


def t_rotate(
    im: Image.Image, rng: random.Random, choices: Sequence[int]
) -> Tuple[Image.Image, Dict[str, Any], str]:
    deg = int(rng.choice(list(choices)))
    if deg == 0:
        return im, {"deg": 0, "noop": True}, "rot0"
    # expand=False keeps size; for 90-multiples it stays aligned
    out = im.rotate(deg, expand=False)
    meta = {"deg": deg}
    tag = f"rot{deg}"
    return out, meta, tag


def t_flip(
    im: Image.Image, rng: random.Random, modes: Sequence[str]
) -> Tuple[Image.Image, Dict[str, Any], str]:
    mode = rng.choice(list(modes))
    if mode == "h":
        return im.transpose(Image.FLIP_LEFT_RIGHT), {"mode": "h"}, "flipH"
    if mode == "v":
        return im.transpose(Image.FLIP_TOP_BOTTOM), {"mode": "v"}, "flipV"
    return im, {"mode": mode, "noop": True}, "flip?"


def t_blur(
    im: Image.Image, rng: random.Random, rmin: float, rmax: float
) -> Tuple[Image.Image, Dict[str, Any], str]:
    radius = float(rng.uniform(rmin, rmax))
    out = im.filter(ImageFilter.GaussianBlur(radius=radius))
    meta = {"radius": round(radius, 4)}
    tag = f"blur{radius:.2f}"
    return out, meta, tag


def t_jpeg_artifacts(
    im: Image.Image, rng: random.Random, qmin: int, qmax: int
) -> Tuple[Image.Image, Dict[str, Any], str]:
    import io

    q = int(rng.randint(int(qmin), int(qmax)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q, optimize=False, progressive=False)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    meta = {"quality": q}
    tag = f"jpegq{q}"
    return out, meta, tag


def t_gamma(
    im: Image.Image, rng: random.Random, gmin: float, gmax: float
) -> Tuple[Image.Image, Dict[str, Any], str]:
    gamma = float(rng.uniform(gmin, gmax))
    arr = pil_to_np(im)
    # gamma correction in 0..255 space
    x = arr / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), gamma)
    out = np_to_pil(y * 255.0)
    meta = {"gamma": round(gamma, 4)}
    tag = f"g{gamma:.2f}"
    return out, meta, tag


def t_contrast(
    im: Image.Image, rng: random.Random, cmin: float, cmax: float
) -> Tuple[Image.Image, Dict[str, Any], str]:
    c = float(rng.uniform(cmin, cmax))
    out = ImageEnhance.Contrast(im).enhance(c)
    meta = {"contrast": round(c, 4)}
    tag = f"c{c:.2f}"
    return out, meta, tag


def t_noise(
    im: Image.Image, rng: random.Random, nmin: float, nmax: float
) -> Tuple[Image.Image, Dict[str, Any], str]:
    sigma = float(rng.uniform(nmin, nmax))
    arr = pil_to_np(im)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np_to_pil(arr + noise)
    meta = {"sigma": round(sigma, 4)}
    tag = f"n{sigma:.1f}"
    return out, meta, tag


# -----------------------------
# Variant generation
# -----------------------------
def sample_and_apply_transforms(
    im: Image.Image, cfg: AugConfig, rng: random.Random
) -> Tuple[Image.Image, List[Dict[str, Any]], List[str]]:
    """
    Applies a random subset of transforms (each with its own probability).
    Returns transformed image, list of transform metadata, and tags.
    """
    transforms_meta: List[Dict[str, Any]] = []
    tags: List[str] = []

    # local helper
    def maybe(p: float) -> bool:
        return rng.random() < float(p)

    # IMPORTANT: order matters a bit; keep it stable
    applied_any = False

    if cfg.p_crop > 0 and maybe(cfg.p_crop):
        im, meta, tag = t_random_crop_keep_aspect(im, rng, cfg.crop_min, cfg.crop_max)
        transforms_meta.append({"name": "crop", "params": meta})
        tags.append(tag)
        applied_any = applied_any or not meta.get("noop", False)

    if cfg.p_resample > 0 and maybe(cfg.p_resample):
        im, meta, tag = t_resample_down_up(im, rng, cfg.resample_min, cfg.resample_max, cfg.resample_methods)
        transforms_meta.append({"name": "resample", "params": meta})
        tags.append(tag)
        applied_any = True

    if cfg.p_rot > 0 and maybe(cfg.p_rot):
        im, meta, tag = t_rotate(im, rng, cfg.rot_choices)
        transforms_meta.append({"name": "rotate", "params": meta})
        tags.append(tag)
        applied_any = applied_any or not meta.get("noop", False)

    if cfg.p_flip > 0 and maybe(cfg.p_flip) and len(cfg.flip_modes) > 0:
        im, meta, tag = t_flip(im, rng, cfg.flip_modes)
        transforms_meta.append({"name": "flip", "params": meta})
        tags.append(tag)
        applied_any = True

    if cfg.p_blur > 0 and maybe(cfg.p_blur):
        im, meta, tag = t_blur(im, rng, cfg.blur_min, cfg.blur_max)
        transforms_meta.append({"name": "blur", "params": meta})
        tags.append(tag)
        applied_any = True

    if cfg.p_jpeg > 0 and maybe(cfg.p_jpeg):
        im, meta, tag = t_jpeg_artifacts(im, rng, cfg.jpeg_qmin, cfg.jpeg_qmax)
        transforms_meta.append({"name": "jpeg", "params": meta})
        tags.append(tag)
        applied_any = True

    if cfg.p_gamma > 0 and maybe(cfg.p_gamma):
        im, meta, tag = t_gamma(im, rng, cfg.gamma_min, cfg.gamma_max)
        transforms_meta.append({"name": "gamma", "params": meta})
        tags.append(tag)
        applied_any = True

    if cfg.p_contrast > 0 and maybe(cfg.p_contrast):
        im, meta, tag = t_contrast(im, rng, cfg.contrast_min, cfg.contrast_max)
        transforms_meta.append({"name": "contrast", "params": meta})
        tags.append(tag)
        applied_any = True

    if cfg.p_noise > 0 and maybe(cfg.p_noise):
        # NOTE: noise uses np.random; tie it to rng for reproducibility
        # by seeding NumPy from rng each time we apply noise.
        np.random.seed(rng.randint(0, 2**31 - 1))
        im, meta, tag = t_noise(im, rng, cfg.noise_min, cfg.noise_max)
        transforms_meta.append({"name": "noise", "params": meta})
        tags.append(tag)
        applied_any = True

    # ensure not trivial
    if cfg.ensure_nontrivial and not applied_any:
        # Force one transform from the enabled set
        forced = []
        if cfg.p_crop > 0: forced.append("crop")
        if cfg.p_resample > 0: forced.append("resample")
        if cfg.p_rot > 0: forced.append("rotate")
        if cfg.p_flip > 0 and len(cfg.flip_modes) > 0: forced.append("flip")
        if cfg.p_blur > 0: forced.append("blur")
        if cfg.p_jpeg > 0: forced.append("jpeg")
        if cfg.p_gamma > 0: forced.append("gamma")
        if cfg.p_contrast > 0: forced.append("contrast")
        if cfg.p_noise > 0: forced.append("noise")

        if forced:
            name = rng.choice(forced)
            if name == "crop":
                im, meta, tag = t_random_crop_keep_aspect(im, rng, cfg.crop_min, cfg.crop_max)
            elif name == "resample":
                im, meta, tag = t_resample_down_up(im, rng, cfg.resample_min, cfg.resample_max, cfg.resample_methods)
            elif name == "rotate":
                im, meta, tag = t_rotate(im, rng, cfg.rot_choices)
            elif name == "flip":
                im, meta, tag = t_flip(im, rng, cfg.flip_modes)
            elif name == "blur":
                im, meta, tag = t_blur(im, rng, cfg.blur_min, cfg.blur_max)
            elif name == "jpeg":
                im, meta, tag = t_jpeg_artifacts(im, rng, cfg.jpeg_qmin, cfg.jpeg_qmax)
            elif name == "gamma":
                im, meta, tag = t_gamma(im, rng, cfg.gamma_min, cfg.gamma_max)
            elif name == "contrast":
                im, meta, tag = t_contrast(im, rng, cfg.contrast_min, cfg.contrast_max)
            else:  # noise
                np.random.seed(rng.randint(0, 2**31 - 1))
                im, meta, tag = t_noise(im, rng, cfg.noise_min, cfg.noise_max)

            transforms_meta.append({"name": f"{name}(forced)", "params": meta})
            tags.append(f"F{tag}")

    return im, transforms_meta, tags


def safe_tag_join(tags: List[str], max_len: int = 120) -> str:
    if not tags:
        return "none"
    s = "__".join(tags)
    # keep filename reasonable
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def save_image(im: Image.Image, out_path: Path, out_ext: str) -> None:
    out_ext = out_ext.lower()
    if out_ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        raise ValueError(f"Unsupported out_ext: {out_ext}")
    params = {}
    if out_ext in (".jpg", ".jpeg"):
        params = {"quality": 95, "subsampling": 0}
    out_path = out_path.with_suffix(out_ext)
    im.save(out_path, **params)


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="make-cbir-aug-positives",
        description="Generate augmentation-positive variants for CBIR evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in_dir", type=str, required=True, help="Input folder of real panel images.")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder root.")
    p.add_argument("--recursive", action="store_true", help="Recursively scan input folder.")
    p.add_argument("--exts", type=str, default=",".join(sorted(IMG_EXTS_DEFAULT)), help="Comma-separated extensions.")
    p.add_argument("--seed", type=int, default=0, help="Global seed. Per-image seed derives from this + path hash.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N images (for quick tests).")

    # output / sizing
    p.add_argument("--out_ext", type=str, default=".png", help="Final saved image extension (e.g. .png).")
    p.add_argument("--grayscale", action="store_true", help="Convert to grayscale then back to RGB.")
    p.add_argument("--resize_mode", type=str, default="none", choices=["none", "resize", "letterpad"])
    p.add_argument("--img_size", type=int, default=None, help="Target square size when resize_mode != none.")
    p.add_argument("--letterpad_fill", type=int, default=0, help="Padding fill value 0..255.")

    # variants
    p.add_argument("--n_variants", type=int, default=8)
    p.add_argument("--ensure_nontrivial", action="store_true", help="Force at least 1 transform per variant.")

    # probabilities
    p.add_argument("--p_crop", type=float, default=0.60)
    p.add_argument("--p_resample", type=float, default=0.35)
    p.add_argument("--p_rot", type=float, default=0.30)
    p.add_argument("--p_flip", type=float, default=0.30)
    p.add_argument("--p_blur", type=float, default=0.20)
    p.add_argument("--p_jpeg", type=float, default=0.30)
    p.add_argument("--p_gamma", type=float, default=0.30)
    p.add_argument("--p_contrast", type=float, default=0.30)
    p.add_argument("--p_noise", type=float, default=0.25)

    # params
    p.add_argument("--crop_min", type=float, default=0.70)
    p.add_argument("--crop_max", type=float, default=0.95)

    p.add_argument("--resample_min", type=float, default=0.50)
    p.add_argument("--resample_max", type=float, default=0.90)
    p.add_argument("--resample_methods", type=str, default="bilinear,bicubic,lanczos,nearest")

    p.add_argument("--rot_choices", type=str, default="0,90,180,270")

    p.add_argument("--flip_modes", type=str, default="h,v")

    p.add_argument("--blur_min", type=float, default=0.3)
    p.add_argument("--blur_max", type=float, default=1.8)

    p.add_argument("--jpeg_qmin", type=int, default=35)
    p.add_argument("--jpeg_qmax", type=int, default=85)

    p.add_argument("--gamma_min", type=float, default=0.80)
    p.add_argument("--gamma_max", type=float, default=1.25)

    p.add_argument("--contrast_min", type=float, default=0.80)
    p.add_argument("--contrast_max", type=float, default=1.25)

    p.add_argument("--noise_min", type=float, default=1.0)
    p.add_argument("--noise_max", type=float, default=6.0)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    cfg = AugConfig(
        out_ext=args.out_ext,
        grayscale=bool(args.grayscale),
        resize_mode=args.resize_mode,
        img_size=args.img_size,
        letterpad_fill=int(args.letterpad_fill),
        n_variants=int(args.n_variants),
        ensure_nontrivial=bool(args.ensure_nontrivial),
        p_crop=float(args.p_crop),
        p_resample=float(args.p_resample),
        p_rot=float(args.p_rot),
        p_flip=float(args.p_flip),
        p_blur=float(args.p_blur),
        p_jpeg=float(args.p_jpeg),
        p_gamma=float(args.p_gamma),
        p_contrast=float(args.p_contrast),
        p_noise=float(args.p_noise),
        crop_min=float(args.crop_min),
        crop_max=float(args.crop_max),
        resample_min=float(args.resample_min),
        resample_max=float(args.resample_max),
        resample_methods=tuple([m.strip() for m in args.resample_methods.split(",") if m.strip()]),
        rot_choices=tuple([int(x.strip()) for x in args.rot_choices.split(",") if x.strip()]),
        flip_modes=tuple([m.strip() for m in args.flip_modes.split(",") if m.strip()]),
        blur_min=float(args.blur_min),
        blur_max=float(args.blur_max),
        jpeg_qmin=int(args.jpeg_qmin),
        jpeg_qmax=int(args.jpeg_qmax),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
        contrast_min=float(args.contrast_min),
        contrast_max=float(args.contrast_max),
        noise_min=float(args.noise_min),
        noise_max=float(args.noise_max),
    )

    if cfg.resize_mode != "none" and cfg.img_size is None:
        logger.error("img_size is required when resize_mode is resize/letterpad.")
        return 2

    paths = discover_images(in_dir, recursive=bool(args.recursive), exts=exts)
    if args.limit and int(args.limit) > 0:
        paths = paths[: int(args.limit)]

    logger.info(f"Input dir: {in_dir}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Found {len(paths)} images (recursive={bool(args.recursive)})")
    logger.info(f"Aug config: {json.dumps(asdict(cfg), indent=2)}")

    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"

    n_ok = 0
    n_fail = 0
    n_variants_total = 0

    it = paths
    if tqdm is not None:
        it = tqdm(paths, desc="Generating", unit="img")

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for src_path in it:
            img_out_dir, rel = make_output_dir_for_image(src_path, in_dir, out_dir)

            im0 = safe_open_image(src_path)
            if im0 is None:
                n_fail += 1
                continue

            im0 = to_rgb(im0, grayscale=cfg.grayscale)
            orig_size = {"w": im0.size[0], "h": im0.size[1]}
            im_base = apply_final_resize_policy(im0, cfg)

            # save orig
            orig_file = img_out_dir / f"orig{cfg.out_ext}"
            save_image(im_base, orig_file, cfg.out_ext)

            # per-image deterministic seed derived from global seed + path string
            seed_mix = int(hashlib.md5((str(args.seed) + "|" + rel).encode("utf-8")).hexdigest()[:8], 16)
            base_rng = random.Random(seed_mix)

            variants_meta: List[Dict[str, Any]] = []

            for vi in range(1, cfg.n_variants + 1):
                # each variant uses its own seed derived from base_rng
                v_seed = base_rng.randint(0, 2**31 - 1)
                rng = random.Random(v_seed)

                im_v = im_base.copy()
                im_v, tmeta, tags = sample_and_apply_transforms(im_v, cfg, rng)

                # name
                tag_str = safe_tag_join(tags)
                vname = f"v{vi:03d}__{tag_str}{cfg.out_ext}"
                vpath = img_out_dir / vname
                save_image(im_v, vpath, cfg.out_ext)

                rec = {
                    "variant_id": vi,
                    "seed": v_seed,
                    "file": str(vpath.relative_to(out_dir)),
                    "transforms": tmeta,
                    "tags": tags,
                }
                variants_meta.append(rec)

                # global manifest line for later CBIR evaluation
                mf.write(
                    json.dumps(
                        {
                            "kind": "aug_positive",
                            "source_file": str(src_path),
                            "source_rel": rel,
                            "orig_file": str(orig_file.relative_to(out_dir)),
                            "variant_file": str(vpath.relative_to(out_dir)),
                            "variant_id": vi,
                            "seed": v_seed,
                            "transforms": tmeta,
                            "tags": tags,
                            "orig_size": orig_size,
                            "final_size": {"w": im_base.size[0], "h": im_base.size[1]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # per-image metadata
            meta = {
                "generated_at": now_tag(),
                "source_file": str(src_path),
                "source_rel": rel,
                "output_dir": str(img_out_dir.relative_to(out_dir)),
                "orig_size": orig_size,
                "final_size": {"w": im_base.size[0], "h": im_base.size[1]},
                "orig_file": str(orig_file.relative_to(out_dir)),
                "per_image_seed": seed_mix,
                "config_snapshot": asdict(cfg),
                "variants": variants_meta,
            }
            with open(img_out_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            n_ok += 1
            n_variants_total += cfg.n_variants

    summary = {
        "generated_at": now_tag(),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "images_found": len(paths),
        "images_processed_ok": n_ok,
        "images_failed": n_fail,
        "variants_per_image": cfg.n_variants,
        "variants_total": n_variants_total,
        "manifest": str(manifest_path),
        "config": asdict(cfg),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. ok={n_ok}, fail={n_fail}, variants_total={n_variants_total}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Summary : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
