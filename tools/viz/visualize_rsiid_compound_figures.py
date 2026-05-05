#!/usr/bin/env python3
"""
visualize_recodai_compound_figures.py

Recod.ai compound figure annotation visualizer + dataset-level stats.

Modes:
1) Single file:
   --annotation-json /path/to/annotations.json --outdir /path/out

2) Dataset / directory:
   --root /dataset/root --outdir /path/out
   OR pass a directory to --annotation-json (convenience)

Multiprocessing (dataset mode):
- Default uses all CPU cores (os.cpu_count()).
- Control with --workers N

Optional class filter:
- --only-classes Microscopy WesternBlot __FORGERY__
  Processes only those annotation files where at least one of these classnames appears.
  Others are skipped.

Excludes by default (so you don't produce *_gt__bboxes.png overlays):
- any input image whose name contains: "forgery_gt" or "pristine_gt"
- also excludes any input containing "__bboxes" to avoid re-processing outputs if outdir overlaps.

Outputs (dataset mode):
- overlays (optional, default ON)
- dataset_summary.json
- class_counts.csv

Dependencies:
  pip install pillow tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import traceback
from collections import Counter
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class BoxAnn:
    ann_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    label: str
    image_id: str


# ----------------------------
# Utils
# ----------------------------

DEFAULT_EXCLUDE_SUBSTR = ["forgery_gt", "pristine_gt", "__bboxes"]


def _stable_color_rgb(name: str) -> Tuple[int, int, int]:
    if name == "__FORGERY__":
        return (220, 50, 47)   # red
    if name == "__PRISTINE__":
        return (38, 139, 210)  # blue
    import hashlib
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    r = 64 + (int(h[0:2], 16) % 160)
    g = 64 + (int(h[2:4], 16) % 160)
    b = 64 + (int(h[4:6], 16) % 160)
    return (r, g, b)


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_int(x: Any, *, field: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Expected int for '{field}', got {x!r}") from e


def _split_globs(globs_s: str) -> List[str]:
    # allow: "figure.png,figure_v*.png"
    parts = [p.strip() for p in globs_s.split(",") if p.strip()]
    return parts or [globs_s.strip()]


def _parse_class_args(vals: Optional[List[str]]) -> List[str]:
    """
    Accepts:
      --only-classes A B C
    and also:
      --only-classes "A,B,C"
    """
    if not vals:
        return []
    out: List[str] = []
    for v in vals:
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        if "," in v:
            out.extend([x.strip() for x in v.split(",") if x.strip()])
        else:
            out.append(v)
    return out


# ----------------------------
# Parsing
# ----------------------------

def parse_annotations(json_path: Path) -> Tuple[Dict[str, Any], List[BoxAnn], Optional[Tuple[int, int]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    forgery_info = data.get("forgery_info", {}) or {}
    fig_anns = data.get("figure_annotations", {}) or {}

    expected_w = fig_anns.get("width")
    expected_h = fig_anns.get("height")
    expected_size = None
    if isinstance(expected_w, int) and isinstance(expected_h, int):
        expected_size = (expected_w, expected_h)  # (W,H)

    boxes: List[BoxAnn] = []
    for k, v in fig_anns.items():
        if not (isinstance(k, str) and k.isdigit()):
            continue
        if not isinstance(v, dict):
            continue
        bbox = v.get("bbox", {})
        if not isinstance(bbox, dict):
            continue

        ann_id = _safe_int(k, field="ann_id")

        x0 = _safe_int(bbox.get("x0"), field=f"{k}.bbox.x0")
        x1 = _safe_int(bbox.get("x1"), field=f"{k}.bbox.x1")
        y0 = _safe_int(bbox.get("y0"), field=f"{k}.bbox.y0")
        y1 = _safe_int(bbox.get("y1"), field=f"{k}.bbox.y1")

        label = str(v.get("class", "UNKNOWN"))
        image_id = str(v.get("image_id", ""))

        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)

        boxes.append(
            BoxAnn(
                ann_id=ann_id,
                x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                label=label,
                image_id=image_id,
            )
        )

    boxes.sort(key=lambda b: (b.y0, b.x0, b.ann_id))
    meta = {
        "forgery_info": forgery_info,          # used for per-image header rendering
        "template": fig_anns.get("template"),
        "expected_size": expected_size,
        "num_boxes": len(boxes),
    }
    return meta, boxes, expected_size


# ----------------------------
# Drawing
# ----------------------------

def draw_overlay(
    img: Image.Image,
    boxes: List[BoxAnn],
    meta: Dict[str, Any],
    *,
    draw_labels: bool,
    line_width: int,
    fill_alpha: int,
    label_bg_alpha: int,
    header: bool,
) -> Image.Image:
    base = img.convert("RGBA")
    W, H = base.size

    font_size = max(12, int(round(min(W, H) * 0.018)))
    font = _load_font(font_size)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    if header:
        fi = meta.get("forgery_info", {}) or {}
        header_lines = [
            f"class={fi.get('class','')}, modality={fi.get('modality','')}",
            f"submodality={fi.get('submodality','')}, figure_type={fi.get('figure_type','')}",
        ]
        ops = (((fi.get("args") or {}).get("operations")) or {})
        if isinstance(ops, dict) and ops:
            header_lines.append("ops=" + ",".join([f"{k}:{v}" for k, v in ops.items()]))

        pad = max(6, int(round(min(W, H) * 0.006)))
        x, y = pad, pad
        for line in header_lines:
            if not line:
                continue
            tb = d.textbbox((0, 0), line, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            d.rectangle([x - 3, y - 2, x + tw + 6, y + th + 4], fill=(0, 0, 0, 160))
            d.text((x, y), line, font=font, fill=(255, 255, 255, 230))
            y += th + 6

    for b in boxes:
        rgb = _stable_color_rgb(b.label)
        outline = (*rgb, 230)
        fill = (*rgb, fill_alpha)

        x0 = max(0, min(W - 1, b.x0))
        y0 = max(0, min(H - 1, b.y0))
        x1 = max(0, min(W - 1, b.x1))
        y1 = max(0, min(H - 1, b.y1))

        if fill_alpha > 0:
            d.rectangle([x0, y0, x1, y1], fill=fill)

        for t in range(line_width):
            d.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=outline)

        if draw_labels:
            txt = f"{b.ann_id}: {b.label}"
            tb = d.textbbox((0, 0), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            tx, ty = x0 + 2, max(0, y0 - th - 4)
            d.rectangle([tx - 2, ty - 2, tx + tw + 6, ty + th + 4], fill=(0, 0, 0, label_bg_alpha))
            d.text((tx, ty), txt, font=font, fill=(255, 255, 255, 235))

    return Image.alpha_composite(base, overlay).convert("RGB")


# ----------------------------
# Dataset stats aggregation (minimal)
# ----------------------------

@dataclass
class DatasetStats:
    ann_seen: int = 0
    ann_processed: int = 0
    ann_skipped: int = 0

    figures_processed: int = 0
    boxes_total: int = 0

    class_instances: Counter[str] = None
    class_files: Counter[str] = None

    boxes_per_ann: List[int] = None
    expected_widths: List[int] = None
    expected_heights: List[int] = None

    size_mismatch: int = 0
    missing_images: int = 0

    def __post_init__(self) -> None:
        self.class_instances = Counter()
        self.class_files = Counter()
        self.boxes_per_ann = []
        self.expected_widths = []
        self.expected_heights = []

    def merge_partial(self, d: Dict[str, Any]) -> None:
        self.ann_seen += d.get("ann_seen", 0)
        self.ann_processed += d.get("ann_processed", 0)
        self.ann_skipped += d.get("ann_skipped", 0)

        self.figures_processed += d.get("figures_processed", 0)
        self.boxes_total += d.get("boxes_total", 0)
        self.size_mismatch += d.get("size_mismatch", 0)
        self.missing_images += d.get("missing_images", 0)

        self.class_instances.update(d.get("class_instances", Counter()))
        self.class_files.update(d.get("class_files", Counter()))
        self.boxes_per_ann.extend(d.get("boxes_per_ann", []))
        self.expected_widths.extend(d.get("expected_widths", []))
        self.expected_heights.extend(d.get("expected_heights", []))


def _skip_partial() -> Dict[str, Any]:
    return {
        "ann_seen": 1,
        "ann_processed": 0,
        "ann_skipped": 1,
        "figures_processed": 0,
        "boxes_total": 0,
        "size_mismatch": 0,
        "missing_images": 0,
        "class_instances": Counter(),
        "class_files": Counter(),
        "boxes_per_ann": [],
        "expected_widths": [],
        "expected_heights": [],
    }


def update_stats_local(boxes: List[BoxAnn], expected_size: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    local: Dict[str, Any] = {
        "ann_seen": 1,
        "ann_processed": 1,
        "ann_skipped": 0,
        "figures_processed": 0,
        "boxes_total": len(boxes),
        "size_mismatch": 0,
        "missing_images": 0,
        "class_instances": Counter(),
        "class_files": Counter(),
        "boxes_per_ann": [len(boxes)],
        "expected_widths": [],
        "expected_heights": [],
    }

    per_file_classes = set()
    for b in boxes:
        local["class_instances"][b.label] += 1
        per_file_classes.add(b.label)
    for c in per_file_classes:
        local["class_files"][c] += 1

    if expected_size:
        ew, eh = expected_size
        local["expected_widths"].append(ew)
        local["expected_heights"].append(eh)

    return local


def _safe_summary(nums: List[int]) -> Dict[str, Any]:
    if not nums:
        return {}
    out: Dict[str, Any] = {}
    out["min"] = int(min(nums))
    out["max"] = int(max(nums))
    out["mean"] = float(sum(nums) / len(nums))
    out["median"] = float(stats.median(nums))
    if len(nums) >= 2:
        out["stdev"] = float(stats.pstdev(nums))
    return out


def _write_counter_csv(path: Path, counter: Counter[str], *, extra: Optional[Dict[str, Counter[str]]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = set(counter.keys())
    if extra:
        for ec in extra.values():
            keys |= set(ec.keys())

    keys_sorted = sorted(keys, key=lambda k: (-counter.get(k, 0), k))

    cols = ["key", "count"]
    if extra:
        cols += list(extra.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in keys_sorted:
            row = [k, counter.get(k, 0)]
            if extra:
                for name in extra.keys():
                    row.append(extra[name].get(k, 0))
            w.writerow(row)


def _emit_outputs(outdir: Path, ds: DatasetStats) -> None:
    _write_counter_csv(
        outdir / "class_counts.csv",
        ds.class_instances,
        extra={"files_with_class": ds.class_files},
    )

    summary = {
        "ann_seen": ds.ann_seen,
        "ann_processed": ds.ann_processed,
        "ann_skipped": ds.ann_skipped,
        "figures_processed": ds.figures_processed,
        "boxes_total": ds.boxes_total,
        "unique_classes": len(ds.class_instances),
        "size_mismatch": ds.size_mismatch,
        "missing_images": ds.missing_images,
        "boxes_per_annotation": _safe_summary(ds.boxes_per_ann),
        "expected_widths": _safe_summary(ds.expected_widths),
        "expected_heights": _safe_summary(ds.expected_heights),
        "top_classes": ds.class_instances.most_common(30),
    }

    (outdir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ----------------------------
# Image resolution helper
# ----------------------------

def resolve_images_for_annotation(
    ann_path: Path,
    images_globs: str,
    exclude_substr: List[str],
    explicit_images: Optional[List[str]],
) -> List[Path]:
    if explicit_images:
        candidates = [Path(p) for p in explicit_images]
    else:
        folder = ann_path.parent
        candidates: List[Path] = []
        for pat in _split_globs(images_globs):
            candidates.extend([Path(p) for p in glob(str(folder / pat))])

    seen = set()
    out: List[Path] = []
    for p in sorted(candidates):
        s = p.name.lower()
        if any(sub.lower() in s for sub in exclude_substr):
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def find_annotation_files(root: Path, ann_name: str) -> List[Path]:
    return sorted(root.rglob(ann_name))


def _classes_present(boxes: List[BoxAnn]) -> set[str]:
    return {b.label for b in boxes}


# ----------------------------
# Worker (multiprocessing)
# ----------------------------

def _worker_process_one(
    ann_path_str: str,
    outdir_str: str,
    root_for_rel_str: str,
    *,
    images_glob: str,
    exclude_substr: List[str],
    draw_labels: bool,
    header: bool,
    fill_alpha: int,
    label_bg_alpha: int,
    line_width: int,
    strict_size: bool,
    render: bool,
    only_classes: List[str],
) -> Dict[str, Any]:
    ann_path = Path(ann_path_str)
    outdir = Path(outdir_str)
    root_for_rel = Path(root_for_rel_str) if root_for_rel_str else None

    meta, boxes, expected_size = parse_annotations(ann_path)

    # Optional class filter
    if only_classes:
        wanted = set(only_classes)
        present = _classes_present(boxes)
        if present.isdisjoint(wanted):
            return _skip_partial()

    local = update_stats_local(boxes, expected_size)

    if not render:
        return local

    img_paths = resolve_images_for_annotation(
        ann_path,
        images_glob,
        exclude_substr=exclude_substr,
        explicit_images=None,
    )

    if not img_paths:
        local["missing_images"] += 1
        raise FileNotFoundError(f"No images found for {ann_path} using images_glob={images_glob} (after excludes)")

    for img_path in img_paths:
        if not img_path.exists():
            local["missing_images"] += 1
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as im:
            img = im.convert("RGB")

        W, H = img.size
        if expected_size:
            ew, eh = expected_size
            if (W, H) != (ew, eh):
                msg = f"Size mismatch: {img_path} image={W}x{H} ann={ew}x{eh} (W x H)"
                if strict_size:
                    raise ValueError(msg)
                local["size_mismatch"] += 1

        lw = line_width
        if lw <= 0:
            lw = max(2, int(round(min(W, H) * 0.003)))

        overlay = draw_overlay(
            img,
            boxes,
            meta,
            draw_labels=draw_labels,
            line_width=lw,
            fill_alpha=fill_alpha,
            label_bg_alpha=label_bg_alpha,
            header=header,
        )

        if root_for_rel is not None:
            rel = ann_path.parent.relative_to(root_for_rel)
            out_folder = outdir / rel
        else:
            out_folder = outdir

        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"{img_path.stem}__bboxes.png"
        overlay.save(out_path, quality=95)
        local["figures_processed"] += 1

    return local


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--annotation-json", type=str, help="Single annotations.json path OR a directory (convenience)")
    mode.add_argument("--root", type=str, help="Dataset root directory (recursive)")

    ap.add_argument("--outdir", required=True, type=str, help="Output directory")
    ap.add_argument("--ann-name", default="annotations.json", help="Annotation filename to search in dataset mode (default: annotations.json)")

    ap.add_argument("--images", nargs="*", default=None, help="Explicit images list (single mode only). If omitted, uses --images-glob in annotation folder.")
    ap.add_argument("--images-glob", default="figure*.png", help='Glob(s) inside each annotation folder. You can pass comma-separated globs, e.g. "figure.png,figure_v*.png".')

    ap.add_argument("--exclude-substr", nargs="*", default=DEFAULT_EXCLUDE_SUBSTR,
                    help=f"Exclude input images whose filename contains any of these substrings (default: {DEFAULT_EXCLUDE_SUBSTR})")

    ap.add_argument("--only-classes", nargs="*", default=None,
                    help='Optional: process only annotations containing ANY of these box classnames. '
                         'Example: --only-classes Microscopy WesternBlot __FORGERY__  (comma also allowed: "Microscopy,WesternBlot")')

    ap.add_argument("--draw-labels", action="store_true")
    ap.add_argument("--no-draw-labels", dest="draw_labels", action="store_false")
    ap.set_defaults(draw_labels=True)

    ap.add_argument("--header", action="store_true")
    ap.add_argument("--no-header", dest="header", action="store_false")
    ap.set_defaults(header=True)

    ap.add_argument("--line-width", type=int, default=0, help="0 = auto thickness")
    ap.add_argument("--fill-alpha", type=int, default=0, help="0..255 box translucent fill (0 = none)")
    ap.add_argument("--label-bg-alpha", type=int, default=160)

    ap.add_argument("--strict-size", action="store_true", help="Error if image size != annotation width/height")
    ap.add_argument("--no-render", action="store_true", help="Compute stats only (no overlay rendering)")

    ap.add_argument("--workers", type=int, default=0, help="Dataset mode: number of worker processes. 0 means all cores (default).")
    ap.add_argument("--on-error", choices=["stop", "continue"], default="stop", help="Default: stop with full traceback")

    args = ap.parse_args()

    only_classes = _parse_class_args(args.only_classes)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fill_alpha = max(0, min(255, args.fill_alpha))
    label_bg_alpha = max(0, min(255, args.label_bg_alpha))

    ds = DatasetStats()

    # ---- Single file mode (file only)
    if args.annotation_json and Path(args.annotation_json).expanduser().resolve().is_file():
        ann_path = Path(args.annotation_json).expanduser().resolve()
        try:
            meta, boxes, expected_size = parse_annotations(ann_path)

            if only_classes:
                present = _classes_present(boxes)
                if present.isdisjoint(set(only_classes)):
                    print(f"[SKIP] {ann_path} does not contain any of --only-classes: {only_classes}")
                    return

            partial = update_stats_local(boxes, expected_size)

            if not args.no_render:
                img_paths = resolve_images_for_annotation(
                    ann_path,
                    args.images_glob,
                    exclude_substr=args.exclude_substr,
                    explicit_images=args.images,
                )
                if not img_paths:
                    partial["missing_images"] += 1
                    raise FileNotFoundError(f"No images found for {ann_path} using images_glob={args.images_glob} (after excludes)")

                for img_path in img_paths:
                    with Image.open(img_path) as im:
                        img = im.convert("RGB")
                    W, H = img.size

                    if expected_size:
                        ew, eh = expected_size
                        if (W, H) != (ew, eh):
                            msg = f"Size mismatch: {img_path} image={W}x{H} ann={ew}x{eh} (W x H)"
                            if args.strict_size:
                                raise ValueError(msg)
                            partial["size_mismatch"] += 1

                    lw = args.line_width
                    if lw <= 0:
                        lw = max(2, int(round(min(W, H) * 0.003)))

                    overlay = draw_overlay(
                        img, boxes, meta,
                        draw_labels=args.draw_labels,
                        line_width=lw,
                        fill_alpha=fill_alpha,
                        label_bg_alpha=label_bg_alpha,
                        header=args.header,
                    )
                    overlay.save(outdir / f"{img_path.stem}__bboxes.png", quality=95)
                    partial["figures_processed"] += 1

            ds.merge_partial(partial)
            _emit_outputs(outdir, ds)
            print(f"[OK] Done. ann_processed={ds.ann_processed}, overlays={ds.figures_processed}, boxes_total={ds.boxes_total}")
            return

        except Exception:
            print("\n" + "=" * 80)
            print(f"[ERROR] Failed on: {ann_path}")
            print("=" * 80)
            traceback.print_exc()
            raise

    # ---- Dataset mode (either --root, or a directory passed to --annotation-json)
    root = None
    if args.root:
        root = Path(args.root).expanduser().resolve()
    elif args.annotation_json:
        p = Path(args.annotation_json).expanduser().resolve()
        if p.is_dir():
            root = p

    if root is None:
        raise ValueError("Provide --root for dataset mode, or pass a directory to --annotation-json")

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    ann_files = find_annotation_files(root, args.ann_name)
    if not ann_files:
        raise FileNotFoundError(f"No '{args.ann_name}' found under {root}")

    render = not args.no_render
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for ann_path in ann_files:
            fut = ex.submit(
                _worker_process_one,
                str(ann_path),
                str(outdir),
                str(root),
                images_glob=args.images_glob,
                exclude_substr=args.exclude_substr,
                draw_labels=args.draw_labels,
                header=args.header,
                fill_alpha=fill_alpha,
                label_bg_alpha=label_bg_alpha,
                line_width=args.line_width,
                strict_size=args.strict_size,
                render=render,
                only_classes=only_classes,
            )
            futures[fut] = ann_path

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing annotations (workers={workers})",
            unit="ann",
            dynamic_ncols=True
        )

        for fut in pbar:
            ann_path = futures[fut]
            try:
                partial = fut.result()
                ds.merge_partial(partial)
                pbar.set_postfix_str(
                    f"seen={ds.ann_seen} kept={ds.ann_processed} skip={ds.ann_skipped} "
                    f"imgs={ds.figures_processed} boxes={ds.boxes_total}"
                )

            except Exception:
                print("\n" + "=" * 80)
                print(f"[ERROR] Failed on annotations: {ann_path}")
                print("=" * 80)
                traceback.print_exc()
                if args.on_error == "continue":
                    continue
                raise

    _emit_outputs(outdir, ds)
    print(
        f"[OK] Dataset done. ann_seen={ds.ann_seen}, ann_processed={ds.ann_processed}, "
        f"ann_skipped={ds.ann_skipped}, overlays={ds.figures_processed}, boxes_total={ds.boxes_total}"
    )


if __name__ == "__main__":
    main()
