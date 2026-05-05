#!/usr/bin/env python3
"""
Realistic synthetic overlay dataset generator for panel crops.

New features added:
1) Optional preprocess of input background:
   - If preprocess.long_side is set (e.g. 512) and image is larger, then choose one of:
     a) resize down to long_side (random interpolation)
     b) random region crop to long_side (keeps original aspect ratio)
     c) random resized crop: crop a larger region then resize to long_side (keeps aspect ratio)
2) Progress bar (tqdm, fallback to periodic prints)
3) Output layout options + batching:
   - output.layout = "split" (images/masks/meta roots) or "flat" (all files in one dir)
   - output.batch_size e.g. 100 -> part_0000, part_0001...
   - optional organize_by_panel_type

Deps:
  pip install pillow opencv-python numpy tqdm pyyaml
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageChops

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------
# Config dataclasses
# -------------------------

@dataclass
class IOConfig:
    microscopy_dir: str
    blot_dir: str
    out_dir: str
    format_image: str = "png"
    format_mask: str = "png"
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class SplitConfig:
    train_ratio: float = 0.9
    seed: int = 123


@dataclass
class SamplingConfig:
    n_train: int = 1000
    n_val: int = 200
    p_use_microscopy: float = 0.7

    # density
    p_none: float = 0.10
    p_light: float = 0.45
    p_medium: float = 0.35
    p_heavy: float = 0.10

    # type mix
    p_text: float = 0.50
    p_arrow: float = 0.25
    p_scalebar: float = 0.15
    p_callout: float = 0.07
    p_highlight: float = 0.03

    # count ranges
    light_range: Tuple[int, int] = (1, 2)
    medium_range: Tuple[int, int] = (2, 5)
    heavy_range: Tuple[int, int] = (5, 10)


@dataclass
class PreprocessConfig:
    # If None/0 => disabled (no size limit)
    long_side: Optional[int] = 512

    # Only applied if max(H,W) > long_side
    # Choose one action by probabilities (renormalized):
    p_resize: float = 0.60
    p_random_crop: float = 0.25
    p_random_resized_crop: float = 0.15

    # For random resized crop: choose crop_long = orig_long * u, then resize to long_side
    resized_crop_scale_range: Tuple[float, float] = (0.55, 1.00)

    # Resize interpolation choices (randomized)
    interp_methods: Tuple[str, ...] = ("area", "linear", "cubic", "lanczos")


@dataclass
class OutputConfig:
    # "split": images/<split>/..., masks/<split>/..., meta/<split>/...
    # "flat" : <split>/... (all files in same folder)
    layout: str = "split"  # "split" or "flat"

    # If True: add panel_type subfolder (microscopy/blot/unknown)
    organize_by_panel_type: bool = False

    # If >0: create batch folders like part_0000/, part_0001/ ...
    batch_size: int = 0
    batch_prefix: str = "part"
    batch_digits: int = 4


@dataclass
class TextConfig:
    fonts_dir: str
    text_choices: List[str] = dataclasses.field(default_factory=lambda: [
        "A", "B", "C", "D", "E",
        "1", "2", "3",
        "10 µm", "20 µm", "50 µm",
        "GAPDH", "Actin", "Tubulin",
        "Ctrl", "WT", "KO", "KD"
    ])
    font_size_px: Tuple[int, int] = (12, 42)
    p_outline: float = 0.30
    outline_px: Tuple[int, int] = (1, 2)
    p_box: float = 0.30
    box_pad_px: Tuple[int, int] = (2, 8)
    p_rotate: float = 0.20
    rotate_deg: Tuple[int, int] = (-35, 35)


@dataclass
class ArrowConfig:
    thickness_px: Tuple[int, int] = (1, 5)
    p_dashed: float = 0.20
    dash_len_px: Tuple[int, int] = (6, 14)
    gap_len_px: Tuple[int, int] = (4, 10)


@dataclass
class ScaleBarConfig:
    length_px: Tuple[int, int] = (40, 180)
    thickness_px: Tuple[int, int] = (3, 8)
    p_label: float = 0.80
    label_choices: List[str] = dataclasses.field(default_factory=lambda: ["5 µm", "10 µm", "20 µm", "50 µm"])


@dataclass
class ArtifactConfig:
    p_jpeg: float = 0.70
    jpeg_quality: Tuple[int, int] = (25, 70)

    p_resample: float = 0.70
    resample_scale: Tuple[float, float] = (0.55, 0.95)

    p_blur: float = 0.30
    blur_ksize: Tuple[int, int] = (1, 3)

    p_sharpen: float = 0.30
    sharpen_amount: Tuple[float, float] = (0.2, 0.8)

    p_noise: float = 0.25
    noise_sigma: Tuple[float, float] = (1.0, 6.0)


@dataclass
class MaskConfig:
    include_antialias: bool = True
    alpha_threshold: int = 1
    dilate_px: int = 1


@dataclass
class RuntimeConfig:
    workers: int = 0
    dry_run: bool = False
    limit: int = 0


@dataclass
class GeneratorConfig:
    io: IOConfig
    split: SplitConfig
    sampling: SamplingConfig
    preprocess: PreprocessConfig
    output: OutputConfig
    text: TextConfig
    arrow: ArrowConfig
    scalebar: ScaleBarConfig
    artifacts: ArtifactConfig
    mask: MaskConfig
    runtime: RuntimeConfig


# -------------------------
# Utility
# -------------------------

def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path, dry_run: bool) -> None:
    if dry_run:
        return
    p.mkdir(parents=True, exist_ok=True)


def _load_config(path: Path) -> Dict[str, Any]:
    data = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Install pyyaml or use JSON config.")
        return yaml.safe_load(data)
    return json.loads(data)


def _odd_ksize(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _rand_color_rgba(rng: np.random.Generator) -> Tuple[int, int, int, int]:
    palette = [
        (255, 255, 255, 255),
        (0, 0, 0, 255),
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 255, 255, 255),
        (255, 255, 0, 255),
        (255, 0, 255, 255),
    ]
    c = palette[int(rng.integers(0, len(palette)))]
    if rng.random() < 0.15:
        return (c[0], c[1], c[2], int(rng.integers(120, 220)))
    return c


def _rgba_to_bgra(c: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    r, g, b, a = c
    return (b, g, r, a)


def _bgra_to_rgba(arr_bgra: np.ndarray) -> np.ndarray:
    out = arr_bgra.copy()
    out[..., 0], out[..., 2] = out[..., 2].copy(), out[..., 0].copy()
    return out


def _alpha_composite_rgba(base_rgba: np.ndarray, over_rgba: np.ndarray) -> np.ndarray:
    base = base_rgba.astype(np.float32)
    over = over_rgba.astype(np.float32)

    a_o = over[..., 3:4] / 255.0
    a_b = base[..., 3:4] / 255.0
    rgb_o = over[..., :3]
    rgb_b = base[..., :3]

    out_a = a_o + a_b * (1.0 - a_o)
    out_rgb = np.where(
        out_a > 1e-6,
        (rgb_o * a_o + rgb_b * a_b * (1.0 - a_o)) / out_a,
        0.0
    )
    out = np.concatenate([out_rgb, out_a * 255.0], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _alpha_blend_bgr(bg_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    bg = bg_bgr.astype(np.float32)
    ov_rgb_bgr = overlay_rgba[..., :3].astype(np.float32)[:, :, ::-1]
    a = (overlay_rgba[..., 3].astype(np.float32) / 255.0)[..., None]
    out = bg * (1.0 - a) + ov_rgb_bgr * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _pick_font(fonts_dir: Path, rng: np.random.Generator) -> Path:
    fonts = [p for p in fonts_dir.rglob("*") if p.suffix.lower() in (".ttf", ".otf")]
    if not fonts:
        raise FileNotFoundError(f"No .ttf/.otf fonts found under: {fonts_dir}")
    return fonts[int(rng.integers(0, len(fonts)))]


def _safe_pad_for_image(H: int, W: int) -> int:
    return int(max(0, min(12, min(H, W) // 20)))


# -------------------------
# Preprocess (NEW)
# -------------------------

def _interp_code(name: str) -> int:
    name = name.lower().strip()
    if name == "area":
        return cv2.INTER_AREA
    if name == "linear":
        return cv2.INTER_LINEAR
    if name == "cubic":
        return cv2.INTER_CUBIC
    if name == "lanczos":
        return cv2.INTER_LANCZOS4
    return cv2.INTER_AREA


def preprocess_background(
    bg_bgr: np.ndarray,
    rng: np.random.Generator,
    pcfg: PreprocessConfig,
    op_trace: List[Dict[str, Any]],
) -> np.ndarray:
    """
    If pcfg.long_side is set and background is larger:
      - resize down OR random crop OR random resized crop
    Always keeps aspect ratio (crop keeps original aspect ratio).
    """
    L = pcfg.long_side
    if L is None or int(L) <= 0:
        return bg_bgr

    H, W = bg_bgr.shape[:2]
    orig_long = max(H, W)
    if orig_long <= int(L):
        return bg_bgr  # no upsample

    # Choose action
    probs = np.array([pcfg.p_resize, pcfg.p_random_crop, pcfg.p_random_resized_crop], dtype=np.float64)
    probs = probs / probs.sum()
    action = str(rng.choice(["resize", "random_crop", "random_resized_crop"], p=probs))

    interps = [ _interp_code(x) for x in pcfg.interp_methods ] if pcfg.interp_methods else [cv2.INTER_AREA]
    interp = int(rng.choice(interps))

    def resize_to_long(img: np.ndarray, long_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        s = float(long_side) / float(max(h, w))
        nh = max(2, int(round(h * s)))
        nw = max(2, int(round(w * s)))
        return cv2.resize(img, (nw, nh), interpolation=interp)

    if action == "resize":
        out = resize_to_long(bg_bgr, int(L))
        op_trace.append({"op": "preprocess_resize", "from_hw": [H, W], "to_hw": list(out.shape[:2]), "long_side": int(L), "interp": int(interp)})
        return out

    # Random crop that keeps original aspect ratio and ends at long_side size
    if action == "random_crop":
        # crop size equals target size with same aspect ratio as original
        s = float(int(L)) / float(orig_long)
        crop_h = max(2, int(round(H * s)))
        crop_w = max(2, int(round(W * s)))
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        y0 = int(rng.integers(0, max(1, H - crop_h + 1)))
        x0 = int(rng.integers(0, max(1, W - crop_w + 1)))
        out = bg_bgr[y0:y0 + crop_h, x0:x0 + crop_w].copy()
        op_trace.append({
            "op": "preprocess_random_crop",
            "from_hw": [H, W],
            "crop_xywh": [x0, y0, crop_w, crop_h],
            "to_hw": list(out.shape[:2]),
            "long_side": int(L),
        })
        return out

    # Random resized crop:
    # 1) choose crop_long in [max(L, orig_long*min), orig_long*max]
    # 2) crop a window of that size (keeping aspect ratio)
    # 3) resize down to long_side
    lo, hi = pcfg.resized_crop_scale_range
    crop_long = int(round(orig_long * float(rng.uniform(lo, hi))))
    crop_long = max(int(L), min(orig_long, crop_long))

    s = float(crop_long) / float(orig_long)
    crop_h = max(2, int(round(H * s)))
    crop_w = max(2, int(round(W * s)))
    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)

    y0 = int(rng.integers(0, max(1, H - crop_h + 1)))
    x0 = int(rng.integers(0, max(1, W - crop_w + 1)))
    crop = bg_bgr[y0:y0 + crop_h, x0:x0 + crop_w].copy()
    out = resize_to_long(crop, int(L))

    op_trace.append({
        "op": "preprocess_random_resized_crop",
        "from_hw": [H, W],
        "crop_xywh": [x0, y0, crop_w, crop_h],
        "crop_long": crop_long,
        "resized_to_long_side": int(L),
        "to_hw": list(out.shape[:2]),
        "interp": int(interp),
    })
    return out


# -------------------------
# Renderers (same as before, robust)
# -------------------------

def render_text_pil(H: int, W: int, rng: np.random.Generator, text_cfg: TextConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask = Image.new("L", (W, H), 0)
    draw_tmp = ImageDraw.Draw(overlay)

    if H < 16 or W < 16:
        return np.zeros((H, W, 4), np.uint8), np.zeros((H, W), np.uint8), {
            "type": "text", "skipped": True, "reason": "panel_too_small", "bbox_xyxy": [0, 0, 0, 0]
        }

    text = str(rng.choice(text_cfg.text_choices))
    font_path = _pick_font(Path(text_cfg.fonts_dir), rng)

    fs_lo, fs_hi = text_cfg.font_size_px
    fs_hi = min(fs_hi, max(10, min(H, W)))
    font_size = int(rng.integers(fs_lo, fs_hi + 1)) if fs_hi >= fs_lo else int(max(10, fs_hi))

    def measure(font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        bbox = draw_tmp.textbbox((0, 0), text, font=font)
        tw = max(1, int(bbox[2] - bbox[0]))
        th = max(1, int(bbox[3] - bbox[1]))
        return tw, th

    font = ImageFont.truetype(str(font_path), size=font_size)
    tw, th = measure(font)
    for _ in range(6):
        if tw <= W - 2 and th <= H - 2:
            break
        font_size = max(8, int(font_size * 0.8))
        font = ImageFont.truetype(str(font_path), size=font_size)
        tw, th = measure(font)

    pad = _safe_pad_for_image(H, W)
    mode = rng.choice(["corner", "edge", "any"], p=[0.45, 0.35, 0.20])

    def clamp_int(v: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, v)))

    x_min = 0 + pad
    y_min = 0 + pad
    x_max = max(x_min, (W - pad - tw))
    y_max = max(y_min, (H - pad - th))

    if mode == "corner":
        corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
        x0, y0 = corners[int(rng.integers(0, 4))]
    elif mode == "edge":
        if rng.random() < 0.5:
            y0 = y_min if rng.random() < 0.5 else y_max
            x0 = int(rng.integers(x_min, x_max + 1)) if x_max >= x_min else x_min
        else:
            x0 = x_min if rng.random() < 0.5 else x_max
            y0 = int(rng.integers(y_min, y_max + 1)) if y_max >= y_min else y_min
    else:
        x0 = int(rng.integers(x_min, x_max + 1)) if x_max >= x_min else x_min
        y0 = int(rng.integers(y_min, y_max + 1)) if y_max >= y_min else y_min

    color = _rand_color_rgba(rng)
    outline = (rng.random() < text_cfg.p_outline)
    outline_px = int(rng.integers(text_cfg.outline_px[0], text_cfg.outline_px[1] + 1))
    use_box = (rng.random() < text_cfg.p_box)
    box_pad = int(rng.integers(text_cfg.box_pad_px[0], text_cfg.box_pad_px[1] + 1))

    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp_mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(tmp)
    draw_m = ImageDraw.Draw(tmp_mask)

    if use_box:
        if color[0] > 200 and color[1] > 200 and color[2] > 200:
            box_color = (0, 0, 0, int(rng.integers(120, 220)))
        else:
            box_color = (255, 255, 255, int(rng.integers(120, 220)))

        bx1 = clamp_int(x0 - box_pad, 0, W - 1)
        by1 = clamp_int(y0 - box_pad, 0, H - 1)
        bx2 = clamp_int(x0 + tw + box_pad, 0, W - 1)
        by2 = clamp_int(y0 + th + box_pad, 0, H - 1)
        if bx2 >= bx1 and by2 >= by1:
            draw.rectangle([bx1, by1, bx2, by2], fill=box_color)
            draw_m.rectangle([bx1, by1, bx2, by2], fill=255)

    if outline:
        for dx in range(-outline_px, outline_px + 1):
            for dy in range(-outline_px, outline_px + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x0 + dx, y0 + dy), text, font=font, fill=(0, 0, 0, 255))
                draw_m.text((x0 + dx, y0 + dy), text, font=font, fill=255)

    draw.text((x0, y0), text, font=font, fill=color)
    draw_m.text((x0, y0), text, font=font, fill=255)

    rot = 0.0
    if rng.random() < text_cfg.p_rotate:
        rot = float(rng.integers(text_cfg.rotate_deg[0], text_cfg.rotate_deg[1] + 1))
        tmp = tmp.rotate(rot, resample=Image.BICUBIC, expand=False)
        tmp_mask = tmp_mask.rotate(rot, resample=Image.BICUBIC, expand=False)

    overlay = Image.alpha_composite(overlay, tmp)
    mask = ImageChops.lighter(mask, tmp_mask)

    overlay_rgba = np.array(overlay, dtype=np.uint8)
    mask_u8 = np.array(mask, dtype=np.uint8)
    mask_u8 = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)

    ys, xs = np.where(mask_u8 > 0)
    bbox_xyxy = [0, 0, 0, 0] if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    meta = {
        "type": "text",
        "text": text,
        "font_path": str(font_path),
        "font_size": int(font_size),
        "color_rgba": list(color),
        "outline": bool(outline),
        "outline_px": int(outline_px if outline else 0),
        "background_box": bool(use_box),
        "box_pad_px": int(box_pad if use_box else 0),
        "rotation_deg": float(rot),
        "bbox_xyxy": bbox_xyxy,
    }
    return overlay_rgba, mask_u8, meta


def render_arrow_cv(H: int, W: int, rng: np.random.Generator, arrow_cfg: ArrowConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if H < 8 or W < 8:
        return np.zeros((H, W, 4), np.uint8), np.zeros((H, W), np.uint8), {
            "type": "arrow", "skipped": True, "reason": "panel_too_small", "bbox_xyxy": [0, 0, 0, 0]
        }

    overlay_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    color_rgba = _rand_color_rgba(rng)
    color_bgra = _rgba_to_bgra(color_rgba)
    thickness = int(rng.integers(arrow_cfg.thickness_px[0], arrow_cfg.thickness_px[1] + 1))

    pad = max(2, _safe_pad_for_image(H, W))
    x1 = int(rng.integers(pad, max(pad + 1, W - pad)))
    y1 = int(rng.integers(pad, max(pad + 1, H - pad)))
    x2 = int(np.clip(x1 + rng.integers(-W // 3, W // 3 + 1), pad, W - pad))
    y2 = int(np.clip(y1 + rng.integers(-H // 3, H // 3 + 1), pad, H - pad))

    dashed = (rng.random() < arrow_cfg.p_dashed)
    dash_len = int(rng.integers(arrow_cfg.dash_len_px[0], arrow_cfg.dash_len_px[1] + 1))
    gap_len = int(rng.integers(arrow_cfg.gap_len_px[0], arrow_cfg.gap_len_px[1] + 1))

    if not dashed:
        cv2.arrowedLine(overlay_bgra, (x1, y1), (x2, y2), color=color_bgra,
                        thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        vx, vy = (x2 - x1), (y2 - y1)
        L = float(np.hypot(vx, vy) + 1e-6)
        ux, uy = vx / L, vy / L
        pos = 0.0
        while pos < L:
            seg_len = min(dash_len, L - pos)
            ax = int(round(x1 + ux * pos))
            ay = int(round(y1 + uy * pos))
            bx = int(round(x1 + ux * (pos + seg_len)))
            by = int(round(y1 + uy * (pos + seg_len)))

            if (pos + seg_len + gap_len) >= L:
                cv2.arrowedLine(overlay_bgra, (ax, ay), (x2, y2), color=color_bgra,
                                thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
                cv2.line(mask, (ax, ay), (x2, y2), color=255, thickness=thickness, lineType=cv2.LINE_AA)
                break
            else:
                cv2.line(overlay_bgra, (ax, ay), (bx, by), color=color_bgra,
                         thickness=thickness, lineType=cv2.LINE_AA)
                cv2.line(mask, (ax, ay), (bx, by), color=255,
                         thickness=thickness, lineType=cv2.LINE_AA)

            pos += (dash_len + gap_len)

    overlay_rgba = _bgra_to_rgba(overlay_bgra)
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)

    ys, xs = np.where(mask_u8 > 0)
    bbox_xyxy = [0, 0, 0, 0] if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    meta = {
        "type": "arrow",
        "color_rgba": list(color_rgba),
        "thickness_px": int(thickness),
        "dashed": bool(dashed),
        "dash_len_px": int(dash_len if dashed else 0),
        "gap_len_px": int(gap_len if dashed else 0),
        "p1": [int(x1), int(y1)],
        "p2": [int(x2), int(y2)],
        "bbox_xyxy": bbox_xyxy,
    }
    return overlay_rgba, mask_u8, meta


def render_scalebar(H: int, W: int, rng: np.random.Generator, sb_cfg: ScaleBarConfig, text_cfg: TextConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if H < 16 or W < 16:
        return np.zeros((H, W, 4), np.uint8), np.zeros((H, W), np.uint8), {
            "type": "scalebar", "skipped": True, "reason": "panel_too_small", "bbox_xyxy": [0, 0, 0, 0]
        }

    overlay_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    length = int(rng.integers(sb_cfg.length_px[0], sb_cfg.length_px[1] + 1))
    thickness = int(rng.integers(sb_cfg.thickness_px[0], sb_cfg.thickness_px[1] + 1))

    length = int(min(length, max(8, W - 8)))
    thickness = int(min(thickness, max(2, H - 8)))

    color_rgba = _rand_color_rgba(rng)
    color_bgra = _rgba_to_bgra(color_rgba)

    pad = max(4, _safe_pad_for_image(H, W))
    y = int(np.clip(H - pad - thickness, 0, H - 1))
    if rng.random() < 0.5:
        x = pad
        anchor = "bottom_left"
    else:
        x = int(np.clip(W - pad - length, 0, W - 1))
        anchor = "bottom_right"

    x2 = int(np.clip(x + length, 0, W - 1))
    y2 = int(np.clip(y + thickness, 0, H - 1))

    if x2 >= x and y2 >= y:
        cv2.rectangle(overlay_bgra, (x, y), (x2, y2), color=color_bgra, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(mask, (x, y), (x2, y2), color=255, thickness=-1, lineType=cv2.LINE_AA)

    label = None
    if rng.random() < sb_cfg.p_label:
        label = str(rng.choice(sb_cfg.label_choices))

        overlay_rgba = _bgra_to_rgba(overlay_bgra)
        pil_overlay = Image.fromarray(overlay_rgba, mode="RGBA")
        pil_mask = Image.fromarray(mask, mode="L")
        draw = ImageDraw.Draw(pil_overlay)
        drawm = ImageDraw.Draw(pil_mask)

        font_path = _pick_font(Path(text_cfg.fonts_dir), rng)
        font_size = int(rng.integers(max(10, text_cfg.font_size_px[0]), min(28, text_cfg.font_size_px[1]) + 1))
        font = ImageFont.truetype(str(font_path), size=font_size)

        tx = x
        ty = max(0, y - font_size - 4)
        draw.text((tx, ty), label, font=font, fill=color_rgba)
        drawm.text((tx, ty), label, font=font, fill=255)

        overlay_rgba = np.array(pil_overlay, dtype=np.uint8)
        mask2 = np.array(pil_mask, dtype=np.uint8)
        mask_u8 = np.where(mask2 > 0, 255, 0).astype(np.uint8)

        ys, xs = np.where(mask_u8 > 0)
        bbox_xyxy = [0, 0, 0, 0] if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        meta = {
            "type": "scalebar",
            "color_rgba": list(color_rgba),
            "length_px": int(length),
            "thickness_px": int(thickness),
            "anchor": anchor,
            "label": label,
            "font_path": str(font_path),
            "font_size": int(font_size),
            "bbox_xyxy": bbox_xyxy,
        }
        return overlay_rgba, mask_u8, meta

    overlay_rgba = _bgra_to_rgba(overlay_bgra)
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 0)
    bbox_xyxy = [0, 0, 0, 0] if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    meta = {
        "type": "scalebar",
        "color_rgba": list(color_rgba),
        "length_px": int(length),
        "thickness_px": int(thickness),
        "anchor": anchor,
        "label": None,
        "bbox_xyxy": bbox_xyxy,
    }
    return overlay_rgba, mask_u8, meta


def render_callout_or_highlight(H: int, W: int, rng: np.random.Generator, kind: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if H < 16 or W < 16:
        return np.zeros((H, W, 4), np.uint8), np.zeros((H, W), np.uint8), {
            "type": kind, "skipped": True, "reason": "panel_too_small", "bbox_xyxy": [0, 0, 0, 0]
        }

    overlay_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    color_rgba = _rand_color_rgba(rng)
    pad = max(4, _safe_pad_for_image(H, W))
    x1 = int(rng.integers(pad, max(pad + 1, W - pad)))
    y1 = int(rng.integers(pad, max(pad + 1, H - pad)))
    x2 = int(np.clip(x1 + rng.integers(max(8, W // 8), max(9, W // 2 + 1)), pad, W - pad))
    y2 = int(np.clip(y1 + rng.integers(max(8, H // 8), max(9, H // 2 + 1)), pad, H - pad))

    x1, x2 = (min(x1, x2), max(x1, x2))
    y1, y2 = (min(y1, y2), max(y1, y2))

    if kind == "highlight":
        a = int(rng.integers(60, 160))
        color_rgba2 = (color_rgba[0], color_rgba[1], color_rgba[2], a)
        color_bgra2 = _rgba_to_bgra(color_rgba2)
        cv2.rectangle(overlay_bgra, (x1, y1), (x2, y2), color=color_bgra2, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1, lineType=cv2.LINE_AA)
    else:
        thickness = int(rng.integers(1, 4))
        color_bgra = _rgba_to_bgra(color_rgba)
        cv2.rectangle(overlay_bgra, (x1, y1), (x2, y2), color=color_bgra, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=thickness, lineType=cv2.LINE_AA)

    overlay_rgba = _bgra_to_rgba(overlay_bgra)
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)

    meta = {"type": kind, "color_rgba": list(color_rgba), "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]}
    return overlay_rgba, mask_u8, meta


# -------------------------
# Artifacts / mask
# -------------------------

def apply_artifacts(bgr: np.ndarray, rng: np.random.Generator, acfg: ArtifactConfig, op_trace: List[Dict[str, Any]]) -> np.ndarray:
    out = bgr.copy()

    if rng.random() < acfg.p_resample:
        s = float(rng.uniform(acfg.resample_scale[0], acfg.resample_scale[1]))
        H, W = out.shape[:2]
        nh, nw = max(2, int(H * s)), max(2, int(W * s))
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
        op_trace.append({"op": "artifact_resample", "scale": s})

    if rng.random() < acfg.p_blur:
        k = _odd_ksize(int(rng.integers(acfg.blur_ksize[0], acfg.blur_ksize[1] + 1)))
        if k > 1:
            out = cv2.GaussianBlur(out, (k, k), 0)
            op_trace.append({"op": "artifact_blur", "ksize": k})

    if rng.random() < acfg.p_sharpen:
        amount = float(rng.uniform(acfg.sharpen_amount[0], acfg.sharpen_amount[1]))
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.0 + amount, blur, -amount, 0)
        op_trace.append({"op": "artifact_sharpen", "amount": amount})

    if rng.random() < acfg.p_noise:
        sigma = float(rng.uniform(acfg.noise_sigma[0], acfg.noise_sigma[1]))
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        op_trace.append({"op": "artifact_noise", "sigma": sigma})

    if rng.random() < acfg.p_jpeg:
        q = int(rng.integers(acfg.jpeg_quality[0], acfg.jpeg_quality[1] + 1))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            op_trace.append({"op": "artifact_jpeg", "quality": q})

    return out


def build_mask_from_overlay_alpha(overlay_rgba: np.ndarray, mcfg: MaskConfig) -> np.ndarray:
    alpha = overlay_rgba[..., 3].astype(np.uint8)
    thr = int(mcfg.alpha_threshold if mcfg.include_antialias else 127)
    mask = (alpha >= thr).astype(np.uint8) * 255
    if int(mcfg.dilate_px) > 0:
        k = 2 * int(mcfg.dilate_px) + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


# -------------------------
# Dataset IO helpers
# -------------------------

def list_images(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    if not root.exists():
        return []
    exts_set = set([e.lower() for e in exts])
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_set]


def pick_density_bucket(rng: np.random.Generator, scfg: SamplingConfig) -> str:
    probs = np.array([scfg.p_none, scfg.p_light, scfg.p_medium, scfg.p_heavy], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(["none", "light", "medium", "heavy"], p=probs))


def overlays_count_for_bucket(rng: np.random.Generator, scfg: SamplingConfig, bucket: str) -> int:
    if bucket == "none":
        return 0
    lo, hi = scfg.light_range if bucket == "light" else scfg.medium_range if bucket == "medium" else scfg.heavy_range
    return int(rng.integers(lo, hi + 1))


def pick_overlay_type(rng: np.random.Generator, scfg: SamplingConfig) -> str:
    types = ["text", "arrow", "scalebar", "callout", "highlight"]
    probs = np.array([scfg.p_text, scfg.p_arrow, scfg.p_scalebar, scfg.p_callout, scfg.p_highlight], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(types, p=probs))


# -------------------------
# Generation core
# -------------------------

def generate_one_sample(
    sample_id: str,
    split: str,
    bg_path: Path,
    panel_type: str,
    cfg: GeneratorConfig,
    seed_sample: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed_sample)
    op_trace: List[Dict[str, Any]] = []

    bg_pil = Image.open(bg_path).convert("RGB")
    bg_bgr = _pil_to_bgr(bg_pil)

    # NEW: preprocess (downscale/crop) for speed + baseline
    bg_bgr = preprocess_background(bg_bgr, rng, cfg.preprocess, op_trace)

    H, W = bg_bgr.shape[:2]
    overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    instances: List[Dict[str, Any]] = []

    bucket = pick_density_bucket(rng, cfg.sampling)
    n_over = overlays_count_for_bucket(rng, cfg.sampling, bucket)

    for k in range(n_over):
        t = pick_overlay_type(rng, cfg.sampling)
        inst_seed = int(rng.integers(0, 2**31 - 1))
        rng_k = np.random.default_rng(inst_seed)

        if t == "text":
            ov, m, meta = render_text_pil(H, W, rng_k, cfg.text)
            overlay_rgba = _alpha_composite_rgba(overlay_rgba, ov)
            meta["instance_seed"] = inst_seed
            meta["instance_id"] = f"{sample_id}_t{k:02d}"
            instances.append(meta)
            op_trace.append({"op": "render_text", "instance_id": meta["instance_id"]})

        elif t == "arrow":
            ov, m, meta = render_arrow_cv(H, W, rng_k, cfg.arrow)
            overlay_rgba = _alpha_composite_rgba(overlay_rgba, ov)
            meta["instance_seed"] = inst_seed
            meta["instance_id"] = f"{sample_id}_a{k:02d}"
            instances.append(meta)
            op_trace.append({"op": "render_arrow", "instance_id": meta["instance_id"]})

        elif t == "scalebar":
            ov, m, meta = render_scalebar(H, W, rng_k, cfg.scalebar, cfg.text)
            overlay_rgba = _alpha_composite_rgba(overlay_rgba, ov)
            meta["instance_seed"] = inst_seed
            meta["instance_id"] = f"{sample_id}_s{k:02d}"
            instances.append(meta)
            op_trace.append({"op": "render_scalebar", "instance_id": meta["instance_id"]})

        elif t in ("callout", "highlight"):
            ov, m, meta = render_callout_or_highlight(H, W, rng_k, t)
            overlay_rgba = _alpha_composite_rgba(overlay_rgba, ov)
            meta["instance_seed"] = inst_seed
            meta["instance_id"] = f"{sample_id}_c{k:02d}"
            instances.append(meta)
            op_trace.append({"op": f"render_{t}", "instance_id": meta["instance_id"]})

    ann_mask = build_mask_from_overlay_alpha(overlay_rgba, cfg.mask)
    op_trace.append({"op": "build_mask", "include_antialias": cfg.mask.include_antialias, "dilate_px": cfg.mask.dilate_px})

    comp_bgr = _alpha_blend_bgr(bg_bgr, overlay_rgba)
    op_trace.append({"op": "blend_alpha"})

    comp_bgr = apply_artifacts(comp_bgr, rng, cfg.artifacts, op_trace)

    area_px = int((ann_mask > 0).sum())
    area_pct = float(area_px / float(H * W + 1e-9) * 100.0)

    meta = {
        "sample_id": sample_id,
        "split": split,
        "created_utc": _now_utc_iso(),
        "seed_sample": int(seed_sample),
        "background": {
            "path": str(bg_path),
            "panel_type": panel_type,
            "height": int(H),
            "width": int(W),
        },
        "preprocess": dataclasses.asdict(cfg.preprocess),
        "overlay_density_bucket": bucket,
        "overlay_count": int(n_over),
        "instances": instances,
        "mask_policy": dataclasses.asdict(cfg.mask),
        "mask_stats": {
            "annotation_area_px": area_px,
            "annotation_area_pct": area_pct,
        },
        "op_trace": op_trace,
    }

    return comp_bgr, ann_mask, meta


# -------------------------
# Output layout + batching (NEW)
# -------------------------

def _batch_folder(cfg: OutputConfig, sample_index: int) -> Optional[str]:
    if cfg.batch_size and cfg.batch_size > 0:
        bid = sample_index // int(cfg.batch_size)
        return f"{cfg.batch_prefix}_{bid:0{int(cfg.batch_digits)}d}"
    return None


def resolve_out_dirs(out_root: Path, cfg: OutputConfig, split: str, panel_type: str, sample_index: int) -> Dict[str, Path]:
    """
    Returns dict of dirs for 'images','masks','meta' based on layout/panel_type/batching.
    """
    layout = (cfg.layout or "split").lower().strip()
    batch = _batch_folder(cfg, sample_index)

    def maybe_add_parts(base: Path) -> Path:
        p = base
        if cfg.organize_by_panel_type:
            p = p / panel_type
        if batch is not None:
            p = p / batch
        return p

    if layout == "flat":
        base = out_root / split
        base = maybe_add_parts(base)
        return {"images": base, "masks": base, "meta": base}

    # default split layout
    img = maybe_add_parts(out_root / "images" / split)
    msk = maybe_add_parts(out_root / "masks" / split)
    met = maybe_add_parts(out_root / "meta" / split)
    return {"images": img, "masks": msk, "meta": met}


def save_sample(
    out_root: Path,
    cfg_out: OutputConfig,
    split: str,
    panel_type: str,
    sample_id: str,
    sample_index: int,
    comp_bgr: np.ndarray,
    mask: np.ndarray,
    meta: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, str]:
    dirs = resolve_out_dirs(out_root, cfg_out, split, panel_type, sample_index)
    for d in dirs.values():
        _ensure_dir(d, dry_run)

    img_path = dirs["images"] / f"{sample_id}.png"
    mask_path = dirs["masks"] / f"{sample_id}_ann.png"
    meta_path = dirs["meta"] / f"{sample_id}.json"

    if not dry_run:
        cv2.imwrite(str(img_path), comp_bgr)
        cv2.imwrite(str(mask_path), mask)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"image": str(img_path), "mask": str(mask_path), "meta": str(meta_path)}


# -------------------------
# Config conversion
# -------------------------

def dict_to_config(d: Dict[str, Any]) -> GeneratorConfig:
    io_d = dict(d["io"])
    if isinstance(io_d.get("exts"), list):
        io_d["exts"] = tuple(io_d["exts"])

    split_d = dict(d.get("split", {}))
    sampling_d = dict(d.get("sampling", {}))
    preprocess_d = dict(d.get("preprocess", {}))
    output_d = dict(d.get("output", {}))

    # tuple conversions
    for k in ("light_range", "medium_range", "heavy_range"):
        if isinstance(sampling_d.get(k), list):
            sampling_d[k] = tuple(sampling_d[k])

    if isinstance(preprocess_d.get("resized_crop_scale_range"), list):
        preprocess_d["resized_crop_scale_range"] = tuple(map(float, preprocess_d["resized_crop_scale_range"]))
    if isinstance(preprocess_d.get("interp_methods"), list):
        preprocess_d["interp_methods"] = tuple(preprocess_d["interp_methods"])

    text_d = dict(d["text"])
    for k in ("font_size_px", "outline_px", "box_pad_px", "rotate_deg"):
        if isinstance(text_d.get(k), list):
            text_d[k] = tuple(text_d[k])

    arrow_d = dict(d.get("arrow", {}))
    for k in ("thickness_px", "dash_len_px", "gap_len_px"):
        if isinstance(arrow_d.get(k), list):
            arrow_d[k] = tuple(arrow_d[k])

    sb_d = dict(d.get("scalebar", {}))
    for k in ("length_px", "thickness_px"):
        if isinstance(sb_d.get(k), list):
            sb_d[k] = tuple(sb_d[k])

    art_d = dict(d.get("artifacts", {}))
    for k in ("jpeg_quality", "blur_ksize"):
        if isinstance(art_d.get(k), list):
            art_d[k] = tuple(art_d[k])
    if isinstance(art_d.get("resample_scale"), list):
        art_d["resample_scale"] = tuple(map(float, art_d["resample_scale"]))
    if isinstance(art_d.get("sharpen_amount"), list):
        art_d["sharpen_amount"] = tuple(map(float, art_d["sharpen_amount"]))
    if isinstance(art_d.get("noise_sigma"), list):
        art_d["noise_sigma"] = tuple(map(float, art_d["noise_sigma"]))

    mask_d = dict(d.get("mask", {}))
    runtime_d = dict(d.get("runtime", {}))

    return GeneratorConfig(
        io=IOConfig(**io_d),
        split=SplitConfig(**split_d),
        sampling=SamplingConfig(**sampling_d),
        preprocess=PreprocessConfig(**preprocess_d),
        output=OutputConfig(**output_d),
        text=TextConfig(**text_d),
        arrow=ArrowConfig(**arrow_d),
        scalebar=ScaleBarConfig(**sb_d),
        artifacts=ArtifactConfig(**art_d),
        mask=MaskConfig(**mask_d),
        runtime=RuntimeConfig(**runtime_d),
    )


# -------------------------
# Main run
# -------------------------

def run_generation(cfg_path: Path, out_override: Optional[str], dry_run_flag: bool) -> None:
    raw = _load_config(cfg_path)
    cfg = dict_to_config(raw)

    if out_override is not None:
        cfg.io.out_dir = out_override
    if dry_run_flag:
        cfg.runtime.dry_run = True

    out_root = Path(cfg.io.out_dir)
    dry_run = bool(cfg.runtime.dry_run)

    mic_paths = list_images(Path(cfg.io.microscopy_dir), cfg.io.exts)
    blot_paths = list_images(Path(cfg.io.blot_dir), cfg.io.exts)

    rng_split = np.random.default_rng(cfg.split.seed)
    rng_split.shuffle(mic_paths)
    rng_split.shuffle(blot_paths)

    if len(mic_paths) == 0 and len(blot_paths) == 0:
        raise RuntimeError("No backgrounds found in microscopy_dir or blot_dir.")

    def split_list(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        n = len(paths)
        n_tr = int(round(n * cfg.split.train_ratio))
        return paths[:n_tr], paths[n_tr:]

    mic_tr, mic_va = split_list(mic_paths)
    blot_tr, blot_va = split_list(blot_paths)

    n_train = int(cfg.sampling.n_train)
    n_val = int(cfg.sampling.n_val)
    if cfg.runtime.limit and cfg.runtime.limit > 0:
        n_train = min(n_train, cfg.runtime.limit)
        n_val = min(n_val, cfg.runtime.limit)

    print("=== Synthetic Overlay Dataset Generator ===")
    print(f"Config: {cfg_path}")
    print(f"Dry-run: {dry_run}")
    print(f"Out: {out_root}")
    print(f"Microscopy backgrounds: {len(mic_paths)} (train={len(mic_tr)} val={len(mic_va)})")
    print(f"Blot backgrounds:       {len(blot_paths)} (train={len(blot_tr)} val={len(blot_va)})")
    print(f"Planned samples: train={n_train}, val={n_val}")
    print("Preprocess:", dataclasses.asdict(cfg.preprocess))
    print("Output:", dataclasses.asdict(cfg.output))
    print("Mask policy:", dataclasses.asdict(cfg.mask))
    print("Output format: PNG for images + masks")

    if dry_run:
        sim_n = min(25, n_train + n_val)
        print(f"\n[Dry-run] Simulating {sim_n} samples (no writes) ...")
        cover = []
        for i in range(sim_n):
            split = "train" if i < min(sim_n, n_train) else "val"
            rng = np.random.default_rng(cfg.split.seed + 777 + i)
            use_mic = (rng.random() < cfg.sampling.p_use_microscopy)

            if split == "train":
                mic_pool, blot_pool = mic_tr, blot_tr
            else:
                mic_pool, blot_pool = mic_va, blot_va

            if use_mic and mic_pool:
                bg = mic_pool[int(rng.integers(0, len(mic_pool)))]
                pt = "microscopy"
            elif blot_pool:
                bg = blot_pool[int(rng.integers(0, len(blot_pool)))]
                pt = "blot"
            else:
                pool_all = mic_pool + blot_pool
                bg = pool_all[int(rng.integers(0, len(pool_all)))]
                pt = "unknown"

            sid = f"dry_{split}_{i:06d}"
            comp, msk, meta = generate_one_sample(sid, split, bg, pt, cfg, seed_sample=cfg.split.seed + 1000 + i)
            cover.append(meta["mask_stats"]["annotation_area_pct"])

        cover = np.array(cover, dtype=np.float32)
        print(f"[Dry-run] annotation_area_pct: mean={cover.mean():.3f}, p50={np.median(cover):.3f}, p90={np.percentile(cover, 90):.3f}, max={cover.max():.3f}")
        print("\nDry-run done.")
        return

    _ensure_dir(out_root, dry_run=False)

    manifest_train: List[Dict[str, Any]] = []
    manifest_val: List[Dict[str, Any]] = []

    def gen_split(split: str, n_samples: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        iterator = range(n_samples)

        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"generate:{split}", unit="sample")

        for i in iterator:
            if tqdm is None and (i % max(1, n_samples // 10) == 0):
                print(f"[{split}] {i}/{n_samples}")

            rng = np.random.default_rng(cfg.split.seed + (0 if split == "train" else 9999) + i)
            use_mic = (rng.random() < cfg.sampling.p_use_microscopy)

            if split == "train":
                mic_pool, blot_pool = mic_tr, blot_tr
            else:
                mic_pool, blot_pool = mic_va, blot_va

            if use_mic and mic_pool:
                bg = mic_pool[int(rng.integers(0, len(mic_pool)))]
                pt = "microscopy"
            elif blot_pool:
                bg = blot_pool[int(rng.integers(0, len(blot_pool)))]
                pt = "blot"
            else:
                pool_all = mic_pool + blot_pool
                bg = pool_all[int(rng.integers(0, len(pool_all)))]
                pt = "unknown"

            sample_id = f"{split}_{i:06d}"
            seed_sample = int(cfg.split.seed + 100000 + i + (0 if split == "train" else 500000))
            comp_bgr, mask, meta = generate_one_sample(sample_id, split, bg, pt, cfg, seed_sample=seed_sample)

            paths = save_sample(
                out_root=out_root,
                cfg_out=cfg.output,
                split=split,
                panel_type=pt,
                sample_id=sample_id,
                sample_index=i,
                comp_bgr=comp_bgr,
                mask=mask,
                meta=meta,
                dry_run=False,
            )

            record = {
                "sample_id": sample_id,
                "split": split,
                "panel_type": pt,
                "image": paths["image"],
                "mask": paths["mask"],
                "meta": paths["meta"],
                "H": int(meta["background"]["height"]),
                "W": int(meta["background"]["width"]),
                "annotation_area_pct": float(meta["mask_stats"]["annotation_area_pct"]),
            }
            out.append(record)

        return out

    manifest_train = gen_split("train", n_train)
    manifest_val = gen_split("val", n_val)

    def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_root / "manifest_train.jsonl", manifest_train)
    write_jsonl(out_root / "manifest_val.jsonl", manifest_val)

    all_rows = manifest_train + manifest_val
    cov = np.array([r["annotation_area_pct"] for r in all_rows], dtype=np.float32) if all_rows else np.array([], dtype=np.float32)
    stats = {
        "created_utc": _now_utc_iso(),
        "n_train": len(manifest_train),
        "n_val": len(manifest_val),
        "coverage_pct": {
            "mean": float(cov.mean()) if cov.size else 0.0,
            "p50": float(np.median(cov)) if cov.size else 0.0,
            "p90": float(np.percentile(cov, 90)) if cov.size else 0.0,
            "max": float(cov.max()) if cov.size else 0.0,
        },
    }
    (out_root / "stats_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Wrote: {out_root/'manifest_train.jsonl'}")
    print(f"Wrote: {out_root/'manifest_val.jsonl'}")
    print(f"Wrote: {out_root/'stats_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config")
    ap.add_argument("--out", default=None, help="Override output directory")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; prints plan + simulates a few samples")
    args = ap.parse_args()
    run_generation(Path(args.config), args.out, args.dry_run)


if __name__ == "__main__":
    main()
