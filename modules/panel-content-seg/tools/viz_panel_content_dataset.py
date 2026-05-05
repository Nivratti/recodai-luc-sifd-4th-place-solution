#!/usr/bin/env python3
"""
Visualize panel-content-seg binary content-mask synthetic dataset.

Mask semantics:
  255 = Content (true image pixels inside sampled content region)
  0   = Non-content (padding/border/canvas/background/gutters/neighbor fragments)

Shows:
  - RGB image
  - mask RGB (Content vs Non-content)
  - overlay (alpha-blended, content highlighted)
  - optional edges view (content boundary on image)

Also prints:
  - meta path
  - class pixel stats (content/non-content)
  - content bbox + margins (top/right/bottom/left) in pixels

Usage examples:
  python tools/viz_panel_content_dataset.py --data ./out/panel_content_synth/demo/v1 --split train --random --n 50 --save-dir ./out/viz_samples/demo/v1
  python tools/viz_panel_content_dataset.py --data ./out/panel_content_synth/demo/v1 --split train --index 0 --verbose-meta
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# -----------------------------
# Mask values (NEW)
# -----------------------------
CONTENT_VAL = 255
NONCONTENT_VAL = 0


# -----------------------------
# IO helpers (safe with unicode paths)
# -----------------------------
def imread_rgb(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imread_gray(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return img


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Manifest + metadata
# -----------------------------
@dataclass
class Entry:
    id: str
    image: Path
    mask: Path
    meta: Path


def load_manifest(manifest_path: Path, root: Path) -> List[Entry]:
    entries: List[Entry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = root / obj["image"]
            mask = root / obj["mask"]
            meta = root / obj.get("meta", "")

            eid = str(obj.get("id", "")).strip()
            if not eid:
                eid = img.stem

            entries.append(Entry(id=eid, image=img, mask=mask, meta=meta))
    return entries


def load_meta(meta_path: Path) -> Dict[str, Any]:
    try:
        if not meta_path or not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# -----------------------------
# Stats / geometry
# -----------------------------
def binary_stats(mask: np.ndarray) -> Dict[str, Tuple[int, float]]:
    total = int(mask.size)
    content_cnt = int((mask == CONTENT_VAL).sum())
    non_cnt = int((mask == NONCONTENT_VAL).sum())
    unknown_cnt = int(total - content_cnt - non_cnt)

    def pct(x: int) -> float:
        return (100.0 * x / total) if total > 0 else 0.0

    return {
        "content": (content_cnt, pct(content_cnt)),
        "noncontent": (non_cnt, pct(non_cnt)),
        "unknown": (unknown_cnt, pct(unknown_cnt)),
    }


def content_bbox_and_margins(mask: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Dict[str, int]]]:
    """
    Returns:
      bbox: (x0,y0,x1,y1) inclusive
      margins: dict {t,r,b,l} where:
        t = y0
        l = x0
        b = (H-1 - y1)
        r = (W-1 - x1)
    """
    ys, xs = np.where(mask == CONTENT_VAL)
    if len(xs) == 0:
        return None, None

    y0 = int(ys.min())
    y1 = int(ys.max())
    x0 = int(xs.min())
    x1 = int(xs.max())

    h, w = mask.shape[:2]
    margins = {
        "t": y0,
        "l": x0,
        "b": (h - 1 - y1),
        "r": (w - 1 - x1),
    }
    return (x0, y0, x1, y1), margins


# -----------------------------
# Visualization
# -----------------------------
def get_colors() -> Dict[str, Tuple[int, int, int]]:
    # Non-glowy content color (muted teal)
    return {
        "content": (0, 170, 140),     # muted teal
        "noncontent": (35, 35, 35),   # dark gray (not "gray mask" style preview; just non-content color)
        "unknown": (255, 0, 255),     # magenta for unexpected values
    }


def mask_to_rgb(mask: np.ndarray, colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask == NONCONTENT_VAL] = colors["noncontent"]
    rgb[mask == CONTENT_VAL] = colors["content"]

    known = (mask == NONCONTENT_VAL) | (mask == CONTENT_VAL)
    rgb[~known] = colors["unknown"]
    return rgb


def overlay_content(img: np.ndarray, mask: np.ndarray, content_color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Only overlays content pixels; non-content remains unchanged."""
    out = img.astype(np.float32).copy()
    m = (mask == CONTENT_VAL)
    if m.any():
        out[m] = out[m] * (1.0 - alpha) + np.array(content_color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def edges_on_image(img: np.ndarray, mask: np.ndarray, edge_color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    m = (mask == CONTENT_VAL).astype(np.uint8) * 255
    if m.max() == 0:
        return out
    edges = cv2.Canny(m, 50, 150)
    out[edges > 0] = edge_color
    return out


def build_legend(colors: Dict[str, Tuple[int, int, int]]) -> List[Patch]:
    def rgb01(c: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    return [
        Patch(facecolor=rgb01(colors["content"]), edgecolor="black", label="Content (255)"),
        Patch(facecolor=rgb01(colors["noncontent"]), edgecolor="black", label="Non-content (0)"),
        Patch(facecolor=rgb01(colors["unknown"]), edgecolor="black", label="Unknown (debug)"),
    ]


def show_entry(
    entry: Entry,
    save_dir: Optional[Path] = None,
    show_edges: bool = True,
    overlay_alpha: float = 0.35,
    verbose_meta: bool = False,
) -> None:
    img = imread_rgb(entry.image)
    mask = imread_gray(entry.mask)
    meta = load_meta(entry.meta)

    # sanitize mask shape to match image if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    colors = get_colors()
    stats = binary_stats(mask)
    bbox, margins = content_bbox_and_margins(mask)

    mask_rgb = mask_to_rgb(mask, colors)
    blended = overlay_content(img, mask, colors["content"], alpha=overlay_alpha)
    edge_view = edges_on_image(img, mask, edge_color=colors["content"]) if show_edges else None

    c_cnt, c_pct = stats["content"]
    n_cnt, n_pct = stats["noncontent"]
    u_cnt, u_pct = stats["unknown"]

    # Title
    parts = [
        f"{entry.id}",
        f"pix%: content={c_pct:.1f} non={n_pct:.1f} unknown={u_pct:.1f}",
    ]
    if margins is not None:
        parts.append(f"margins(px): T{margins['t']} R{margins['r']} B{margins['b']} L{margins['l']}")
    title = " — ".join(parts)

    # layout (NO gray mask panel)
    cols = 4 if show_edges else 3
    fig = plt.figure(figsize=(5 * cols, 5))
    plt.suptitle(title, fontsize=10)

    ax1 = plt.subplot(1, cols, 1)
    ax1.imshow(img)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = plt.subplot(1, cols, 2)
    ax2.imshow(mask_rgb)
    ax2.set_title("Mask (RGB)")
    ax2.axis("off")
    ax2.legend(handles=build_legend(colors), loc="lower left", fontsize=8, framealpha=0.9)

    ax3 = plt.subplot(1, cols, 3)
    ax3.imshow(blended)
    ax3.set_title(f"Overlay (content α={overlay_alpha:.2f})")
    ax3.axis("off")

    if show_edges:
        ax4 = plt.subplot(1, cols, 4)
        ax4.imshow(edge_view)
        ax4.set_title("Edges (content boundary)")
        ax4.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_dir is not None:
        ensure_dir(save_dir)
        out_path = save_dir / f"{entry.id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {out_path}")

    plt.show()
    plt.close(fig)

    # Console prints
    if entry.meta and entry.meta.exists():
        print(f"[META] {entry.meta}")
    else:
        print("[META] (missing)")

    print(f"[MASK_STATS] content={c_cnt} ({c_pct:.2f}%) | noncontent={n_cnt} ({n_pct:.2f}%) | unknown={u_cnt} ({u_pct:.2f}%)")
    if bbox is None:
        print("[CONTENT_BBOX] (no content pixels found)")
    else:
        x0, y0, x1, y1 = bbox
        print(f"[CONTENT_BBOX] x0={x0} y0={y0} x1={x1} y1={y1}")
        print(f"[CONTENT_MARGINS] T{margins['t']} R{margins['r']} B{margins['b']} L{margins['l']}")

    if verbose_meta:
        try:
            print("[META_JSON]\n" + json.dumps(meta, indent=2, ensure_ascii=False))
        except Exception:
            print("[META_JSON] (failed to pretty print)")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Dataset root (contains manifest_*.jsonl)")
    ap.add_argument("--split", type=str, choices=["train", "val"], default="train")
    ap.add_argument("--n", type=int, default=10, help="How many samples to show")
    ap.add_argument("--random", action="store_true", help="Random samples")
    ap.add_argument("--index", type=int, default=None, help="Show a specific index (0-based in chosen split)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for --random")
    ap.add_argument("--save-dir", type=str, default=None, help="If set, save rendered figures here")
    ap.add_argument("--no-edges", action="store_true", help="Disable edge overlay view")
    ap.add_argument("--overlay-alpha", type=float, default=0.35, help="Alpha for content overlay on image")
    ap.add_argument("--verbose-meta", action="store_true", help="Print full meta JSON for each sample")
    args = ap.parse_args()

    root = Path(args.data).expanduser()
    manifest = root / ("manifest_train.jsonl" if args.split == "train" else "manifest_val.jsonl")
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    entries = load_manifest(manifest, root)
    if not entries:
        raise SystemExit("Manifest is empty")

    rng = random.Random(args.seed)
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    show_edges = not args.no_edges

    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= len(entries):
            raise SystemExit(f"--index out of range. split={args.split} size={len(entries)}")
        show_entry(
            entries[idx],
            save_dir=save_dir,
            show_edges=show_edges,
            overlay_alpha=args.overlay_alpha,
            verbose_meta=args.verbose_meta,
        )
        return

    if args.random:
        chosen = [entries[rng.randrange(len(entries))] for _ in range(args.n)]
    else:
        chosen = entries[: args.n]

    for e in chosen:
        show_entry(
            e,
            save_dir=save_dir,
            show_edges=show_edges,
            overlay_alpha=args.overlay_alpha,
            verbose_meta=args.verbose_meta,
        )


if __name__ == "__main__":
    main()
