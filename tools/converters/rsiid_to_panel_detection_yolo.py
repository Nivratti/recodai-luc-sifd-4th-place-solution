#!/usr/bin/env python3
"""
recodai_compound_to_yolo.py

Convert Recod.ai compound figure annotations.json (bbox annotations) to YOLO detection labels.

YOLO label line:
  <class_id> <x_center> <y_center> <width> <height>
(all normalized by image width/height)

NEW:
- --train-root (required) and optional --val-root
- Separate outputs: images/{train,val} and labels/{train,val}
- Separate stats per split + overall in conversion_summary.json
- Class mapping + skipping with full tracking:
    input_class_counts (raw labels)
    output_class_counts (after mapping+skip)
    skipped_input_class_counts (raw label skipped)
    skipped_target_class_counts (mapped label skipped)
    mapping_pair_counts (e.g., WesternBlot->image)

Example:
  python recodai_compound_to_yolo.py \
    --train-root "/path/train" \
    --val-root "/path/val" \
    --outdir "./out/yolo" \
    --class-map "WesternBlot,Microscopy,__FORGERY__,__PRISTINE__=image" \
    --skip-classes Others Graphs \
    --images-action symlink \
    --include-empty

Dependencies:
  pip install pillow tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


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


# ----------------------------
# Defaults / helpers
# ----------------------------

DEFAULT_EXCLUDE_SUBSTR = ["forgery_gt", "pristine_gt", "__bboxes"]


def _safe_int(x: Any, *, field: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Expected int for '{field}', got {x!r}") from e


def _split_globs(globs_s: str) -> List[str]:
    parts = [p.strip() for p in str(globs_s).split(",") if p.strip()]
    return parts or [str(globs_s).strip()]


def _parse_list_args(vals: Optional[List[str]]) -> List[str]:
    """
    Accepts:
      --skip-classes A B C
    and also:
      --skip-classes "A,B,C"
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


def _parse_class_map(specs: Optional[List[str]]) -> Dict[str, str]:
    """
    Repeatable mappings:
      --class-map "A,B,C=NEW"
      --class-map "X=Y"
    Also accepts ":" delimiter.

    Returns dict {src: dst}.
    """
    if not specs:
        return {}
    out: Dict[str, str] = {}
    for raw in specs:
        s = str(raw).strip()
        if not s:
            continue

        if "=" in s:
            left, right = s.split("=", 1)
        elif ":" in s:
            left, right = s.split(":", 1)
        else:
            raise ValueError(f"Invalid --class-map '{s}'. Use 'A,B,C=NEW'.")

        dst = right.strip()
        if not dst:
            raise ValueError(f"Invalid --class-map '{s}': empty target class name.")

        srcs = [x.strip() for x in left.split(",") if x.strip()]
        if not srcs:
            raise ValueError(f"Invalid --class-map '{s}': empty source list.")

        for src in srcs:
            if src in out and out[src] != dst:
                raise ValueError(f"Conflicting mapping for '{src}': '{out[src]}' vs '{dst}'.")
            out[src] = dst
    return out


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_image_size_fast(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W,H)


# ----------------------------
# Parsing annotations.json
# ----------------------------

def parse_annotations(json_path: Path) -> Tuple[List[BoxAnn], Optional[Tuple[int, int]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
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

        # ensure ordering
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)

        boxes.append(BoxAnn(ann_id=ann_id, x0=x_min, y0=y_min, x1=x_max, y1=y_max, label=label))

    boxes.sort(key=lambda b: (b.y0, b.x0, b.ann_id))
    return boxes, expected_size


def resolve_images(ann_path: Path, images_glob: str, exclude_substr: List[str]) -> List[Path]:
    folder = ann_path.parent
    cands: List[Path] = []
    for pat in _split_globs(images_glob):
        cands.extend([Path(p) for p in glob(str(folder / pat))])

    out: List[Path] = []
    seen = set()
    for p in sorted(cands):
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


# ----------------------------
# YOLO conversion (end-exclusive)
# ----------------------------

def _clip_box_end_exclusive(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """
    End-exclusive clipping:
      allow x1==W and y1==H (valid)
    """
    x0c = max(0, min(W, x0))
    x1c = max(0, min(W, x1))
    y0c = max(0, min(H, y0))
    y1c = max(0, min(H, y1))

    if x0c > x1c:
        x0c, x1c = x1c, x0c
    if y0c > y1c:
        y0c, y1c = y1c, y0c

    return x0c, y0c, x1c, y1c


def _to_yolo_line(cls_id: int, x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> str:
    bw = x1 - x0
    bh = y1 - y0
    xc = x0 + bw / 2.0
    yc = y0 + bh / 2.0
    return f"{cls_id} {xc / W:.6f} {yc / H:.6f} {bw / W:.6f} {bh / H:.6f}"


def _apply_map_and_skip(label: str, class_map: Dict[str, str], skip_set: set[str]) -> Tuple[Optional[str], str]:
    """
    Returns (target_label_or_None, reason)
      reason:
        - "kept"
        - "skipped_input"
        - "skipped_target"
    """
    if label in skip_set:
        return None, "skipped_input"
    target = class_map.get(label, label)
    if target in skip_set:
        return None, "skipped_target"
    return target, "kept"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    _ensure_parent(dst)
    if dst.exists():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    raise ValueError(f"Unknown images_action: {mode}")


# ----------------------------
# Split stats helpers
# ----------------------------

def _empty_split_stats() -> Dict[str, Any]:
    return {
        "ann_total": 0,
        "ann_processed": 0,
        "ann_skipped_only": 0,
        "ann_skipped_no_images": 0,
        "images_written": 0,
        "labels_written": 0,
        "boxes_written": 0,
        "size_mismatch": 0,
        "dropped_outside": 0,
        "dropped_degenerate": 0,
        "input_class_counts": {},
        "output_class_counts": {},
        "skipped_input_class_counts": {},
        "skipped_target_class_counts": {},
        "mapping_pair_counts": {},
    }


def _merge_counter_dict(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = int(dst.get(k, 0)) + int(v)


def _merge_split_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    # numeric
    for k in [
        "ann_total", "ann_processed", "ann_skipped_only", "ann_skipped_no_images",
        "images_written", "labels_written", "boxes_written",
        "size_mismatch", "dropped_outside", "dropped_degenerate",
    ]:
        dst[k] = int(dst.get(k, 0)) + int(src.get(k, 0))

    # counters
    for ck in [
        "input_class_counts",
        "output_class_counts",
        "skipped_input_class_counts",
        "skipped_target_class_counts",
        "mapping_pair_counts",
    ]:
        _merge_counter_dict(dst.setdefault(ck, {}), src.get(ck, {}))


# ----------------------------
# Workers
# ----------------------------

def _worker_scan_targets(
    ann_path_str: str,
    class_map: Dict[str, str],
    skip_classes: List[str],
    only_classes: List[str],
) -> Dict[str, Any]:
    """
    Scan to discover OUTPUT (target) classes and count raw input labels.
    """
    skip_set = set(skip_classes)
    wanted = set(only_classes) if only_classes else None

    ann_path = Path(ann_path_str)
    boxes, _ = parse_annotations(ann_path)

    input_counts = Counter([b.label for b in boxes])

    if wanted is not None:
        present_raw = {b.label for b in boxes}
        if present_raw.isdisjoint(wanted):
            return {"input_counts": dict(input_counts), "target_counts": {}, "ann_skipped_only": 1}

    target_counts: Counter[str] = Counter()
    for b in boxes:
        tgt, _reason = _apply_map_and_skip(b.label, class_map, skip_set)
        if tgt is None:
            continue
        target_counts[tgt] += 1

    return {"input_counts": dict(input_counts), "target_counts": dict(target_counts), "ann_skipped_only": 0}


def _worker_convert_one(
    ann_path_str: str,
    split_root_str: str,
    outdir_str: str,
    split_name: str,
    class_names: List[str],
    *,
    images_glob: str,
    exclude_substr: List[str],
    only_classes: List[str],
    class_map: Dict[str, str],
    skip_classes: List[str],
    images_action: str,
    include_empty: bool,
    strict_size: bool,
    on_missing_image: str,
    on_outside: str,
    on_degenerate: str,
) -> Dict[str, Any]:
    ann_path = Path(ann_path_str)
    split_root = Path(split_root_str)
    outdir = Path(outdir_str)

    out_images_root = outdir / "images" / split_name
    out_labels_root = outdir / "labels" / split_name

    class_to_id = {n: i for i, n in enumerate(class_names)}
    skip_set = set(skip_classes)
    wanted = set(only_classes) if only_classes else None

    boxes, expected_size = parse_annotations(ann_path)
    input_counts = Counter([b.label for b in boxes])

    # only-classes filter on RAW labels
    if wanted is not None:
        present_raw = {b.label for b in boxes}
        if present_raw.isdisjoint(wanted):
            return {
                "ann_skipped_only": 1,
                "input_class_counts": dict(input_counts),
                "output_class_counts": {},
                "skipped_input_class_counts": {},
                "skipped_target_class_counts": {},
                "mapping_pair_counts": {},
                "dropped_outside": 0,
                "dropped_degenerate": 0,
            }

    img_paths = resolve_images(ann_path, images_glob, exclude_substr)
    if not img_paths:
        if on_missing_image == "skip":
            return {
                "ann_skipped_no_images": 1,
                "input_class_counts": dict(input_counts),
                "output_class_counts": {},
                "skipped_input_class_counts": {},
                "skipped_target_class_counts": {},
                "mapping_pair_counts": {},
                "dropped_outside": 0,
                "dropped_degenerate": 0,
            }
        raise FileNotFoundError(f"No images found for {ann_path} using images_glob={images_glob} (after excludes).")

    rel = ann_path.parent.relative_to(split_root)

    output_counts: Counter[str] = Counter()
    skipped_input_counts: Counter[str] = Counter()
    skipped_target_counts: Counter[str] = Counter()
    mapping_pairs: Counter[str] = Counter()

    images_written = 0
    labels_written = 0
    boxes_written = 0
    size_mismatch = 0
    dropped_degenerate = 0
    dropped_outside = 0

    for img_path in img_paths:
        if not img_path.exists():
            if on_missing_image == "skip":
                continue
            raise FileNotFoundError(f"Image not found: {img_path}")

        W, H = _read_image_size_fast(img_path)

        if expected_size:
            ew, eh = expected_size
            if (W, H) != (ew, eh):
                msg = f"Size mismatch: {img_path} image={W}x{H} ann={ew}x{eh} (W x H)"
                if strict_size:
                    raise ValueError(msg)
                size_mismatch += 1

        out_img = out_images_root / rel / img_path.name
        out_lbl = out_labels_root / rel / f"{img_path.stem}.txt"

        lines: List[str] = []
        for b in boxes:
            tgt, reason = _apply_map_and_skip(b.label, class_map, skip_set)

            if tgt is None:
                if reason == "skipped_input":
                    skipped_input_counts[b.label] += 1
                else:
                    skipped_target_counts[class_map.get(b.label, b.label)] += 1
                continue

            mapping_pairs[f"{b.label}->{tgt}"] += 1

            if tgt not in class_to_id:
                raise KeyError(
                    f"Target label '{tgt}' not in output classes list. ann={ann_path}"
                )

            x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1

            # End-exclusive validity: x1 can equal W and y1 can equal H
            outside = (x0 < 0 or y0 < 0 or x1 > W or y1 > H)
            if outside:
                if on_outside == "error":
                    raise ValueError(
                        f"Box outside image for {img_path} label={b.label} "
                        f"box=({x0},{y0},{x1},{y1}) img={W}x{H}"
                    )
                x0, y0, x1, y1 = _clip_box_end_exclusive(x0, y0, x1, y1, W, H)
                dropped_outside += 1

            if x1 <= x0 or y1 <= y0:
                if on_degenerate == "error":
                    raise ValueError(
                        f"Degenerate box for {img_path} label={b.label} box=({x0},{y0},{x1},{y1}) img={W}x{H}"
                    )
                dropped_degenerate += 1
                continue

            cls_id = class_to_id[tgt]
            lines.append(_to_yolo_line(cls_id, x0, y0, x1, y1, W, H))
            output_counts[tgt] += 1

        if lines or include_empty:
            _ensure_parent(out_lbl)
            out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            labels_written += 1
            boxes_written += len(lines)

        _link_or_copy(img_path, out_img, images_action)
        if images_action != "none":
            images_written += 1

    return {
        "ann_processed": 1,
        "images_written": images_written,
        "labels_written": labels_written,
        "boxes_written": boxes_written,
        "size_mismatch": size_mismatch,
        "dropped_outside": dropped_outside,
        "dropped_degenerate": dropped_degenerate,
        "input_class_counts": dict(input_counts),
        "output_class_counts": dict(output_counts),
        "skipped_input_class_counts": dict(skipped_input_counts),
        "skipped_target_class_counts": dict(skipped_target_counts),
        "mapping_pair_counts": dict(mapping_pairs),
    }


# ----------------------------
# Writers
# ----------------------------

def write_classes_txt(outdir: Path, class_names: List[str]) -> None:
    (outdir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")


def write_data_yaml(outdir: Path, class_names: List[str], has_val: bool) -> None:
    """
    Ultralytics-style data.yaml.
    If val split is not provided, we set val to images/train (so training won't crash).
    """
    yaml_lines = [
        f"path: {str(outdir.resolve())}",
        "train: images/train",
        f"val: {'images/val' if has_val else 'images/train'}",
        "names:",
    ]
    for i, n in enumerate(class_names):
        yaml_lines.append(f"  {i}: {n}")
    (outdir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-root", required=True, type=str, help="Train split root directory (recursive). REQUIRED.")
    ap.add_argument("--val-root", default="", type=str, help="Val split root directory (recursive). OPTIONAL.")
    ap.add_argument("--outdir", required=True, type=str, help="Output directory for YOLO dataset")

    ap.add_argument("--ann-name", default="annotations.json", help="Annotation filename to search (default: annotations.json)")
    ap.add_argument("--images-glob", default="figure*.png",
                    help='Glob(s) inside each annotation folder. Comma-separated allowed, e.g. "figure.png,figure_v*.png".')
    ap.add_argument("--exclude-substr", nargs="*", default=DEFAULT_EXCLUDE_SUBSTR,
                    help=f"Exclude inputs whose filename contains any of these substrings (default: {DEFAULT_EXCLUDE_SUBSTR})")

    ap.add_argument("--only-classes", nargs="*", default=None,
                    help='Process only annotation files containing ANY of these RAW classes. (Comma allowed)')
    ap.add_argument("--skip-classes", nargs="*", default=None,
                    help='Drop these classes from YOLO labels. (Comma allowed)')
    ap.add_argument("--class-map", nargs="*", default=None,
                    help='Rename classes. Repeatable. Example: --class-map "WesternBlot,Microscopy= image" '
                         'Use "A,B,C=NEW" (or "A,B,C:NEW").')

    ap.add_argument("--order", choices=["alpha", "frequency"], default="alpha",
                    help="Class ID assignment order for OUTPUT classes. alpha = sorted; frequency = most common first.")
    ap.add_argument("--classes-from", type=str, default="",
                    help="Optional: existing classes.txt (OUTPUT classes) to reuse exact class_id mapping.")

    ap.add_argument("--images-action", choices=["none", "symlink", "copy", "hardlink"], default="copy",
                    help="What to do with images in outdir/images/{split}. If none (labels only).")
    ap.add_argument("--include-empty", action="store_true",
                    help="Write empty .txt for images with zero kept boxes.")
    ap.add_argument("--strict-size", action="store_true",
                    help="Error if image size != annotation width/height (if present in JSON).")

    ap.add_argument("--on-missing-image", choices=["error", "skip"], default="error")
    ap.add_argument("--on-outside", choices=["error", "clip"], default="error",
                    help="If a box goes outside image bounds: error or clip to bounds.")
    ap.add_argument("--on-degenerate", choices=["error", "skip"], default="error",
                    help="If a box becomes zero/negative after ordering/clip: error or skip.")

    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores (default)")
    ap.add_argument("--on-error", choices=["stop", "continue"], default="stop",
                    help="Default stop with full traceback")

    args = ap.parse_args()

    train_root = Path(args.train_root).expanduser().resolve()
    if not train_root.is_dir():
        raise NotADirectoryError(f"--train-root is not a directory: {train_root}")

    val_root = Path(args.val_root).expanduser().resolve() if args.val_root else None
    if val_root is not None and not val_root.is_dir():
        raise NotADirectoryError(f"--val-root is not a directory: {val_root}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    only_classes = _parse_list_args(args.only_classes)
    skip_classes = _parse_list_args(args.skip_classes)
    class_map = _parse_class_map(args.class_map)

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # Collect annotation files per split
    train_anns = find_annotation_files(train_root, args.ann_name)
    if not train_anns:
        raise FileNotFoundError(f"No '{args.ann_name}' found under train root: {train_root}")

    val_anns: List[Path] = []
    if val_root is not None:
        val_anns = find_annotation_files(val_root, args.ann_name)
        if not val_anns:
            raise FileNotFoundError(f"--val-root provided but no '{args.ann_name}' found under: {val_root}")

    # Build OUTPUT class list
    if args.classes_from:
        cls_path = Path(args.classes_from).expanduser().resolve()
        if not cls_path.is_file():
            raise FileNotFoundError(f"--classes-from not found: {cls_path}")
        class_names = [ln.strip() for ln in cls_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not class_names:
            raise ValueError(f"--classes-from is empty: {cls_path}")
    else:
        scan_target_counts = Counter()
        # scan both splits so output covers all
        scan_inputs = train_anns + val_anns

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_worker_scan_targets, str(p), class_map, skip_classes, only_classes)
                for p in scan_inputs
            ]
            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc=f"Scan classes (workers={workers})", unit="ann", dynamic_ncols=True)
            for fut in pbar:
                res = fut.result()
                scan_target_counts.update(res.get("target_counts", {}))

        if not scan_target_counts:
            raise ValueError("No OUTPUT classes found after mapping/skipping (maybe filters excluded everything).")

        if args.order == "alpha":
            class_names = sorted(scan_target_counts.keys())
        else:
            class_names = sorted(scan_target_counts.keys(), key=lambda k: (-scan_target_counts[k], k))

    write_classes_txt(outdir, class_names)
    write_data_yaml(outdir, class_names, has_val=(val_root is not None))

    # Prepare summary containers
    overall = _empty_split_stats()
    splits: Dict[str, Dict[str, Any]] = {
        "train": _empty_split_stats()
    }
    splits["train"]["ann_total"] = len(train_anns)
    if val_root is not None:
        splits["val"] = _empty_split_stats()
        splits["val"]["ann_total"] = len(val_anns)

    # Convert split function
    def convert_split(split_name: str, split_root: Path, ann_files: List[Path]) -> None:
        nonlocal splits, overall

        split_stats = splits[split_name]
        split_stats["ann_total"] = len(ann_files)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for ann_path in ann_files:
                fut = ex.submit(
                    _worker_convert_one,
                    str(ann_path),
                    str(split_root),
                    str(outdir),
                    split_name,
                    class_names,
                    images_glob=args.images_glob,
                    exclude_substr=args.exclude_substr,
                    only_classes=only_classes,
                    class_map=class_map,
                    skip_classes=skip_classes,
                    images_action=args.images_action,
                    include_empty=bool(args.include_empty),
                    strict_size=bool(args.strict_size),
                    on_missing_image=args.on_missing_image,
                    on_outside=args.on_outside,
                    on_degenerate=args.on_degenerate,
                )
                futures[fut] = ann_path

            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc=f"Convert {split_name} (workers={workers})", unit="ann", dynamic_ncols=True)

            for fut in pbar:
                ann_path = futures[fut]
                try:
                    res = fut.result()

                    # merge into split stats
                    local = _empty_split_stats()
                    local.update({
                        "ann_processed": int(res.get("ann_processed", 0)),
                        "ann_skipped_only": int(res.get("ann_skipped_only", 0)),
                        "ann_skipped_no_images": int(res.get("ann_skipped_no_images", 0)),
                        "images_written": int(res.get("images_written", 0)),
                        "labels_written": int(res.get("labels_written", 0)),
                        "boxes_written": int(res.get("boxes_written", 0)),
                        "size_mismatch": int(res.get("size_mismatch", 0)),
                        "dropped_outside": int(res.get("dropped_outside", 0)),
                        "dropped_degenerate": int(res.get("dropped_degenerate", 0)),
                        "input_class_counts": res.get("input_class_counts", {}) or {},
                        "output_class_counts": res.get("output_class_counts", {}) or {},
                        "skipped_input_class_counts": res.get("skipped_input_class_counts", {}) or {},
                        "skipped_target_class_counts": res.get("skipped_target_class_counts", {}) or {},
                        "mapping_pair_counts": res.get("mapping_pair_counts", {}) or {},
                    })

                    _merge_split_stats(split_stats, local)
                    _merge_split_stats(overall, local)

                    pbar.set_postfix_str(
                        f"proc={split_stats['ann_processed']} skip={split_stats['ann_skipped_only']} "
                        f"lbl={split_stats['labels_written']} boxes={split_stats['boxes_written']}"
                    )

                except Exception:
                    print("\n" + "=" * 80)
                    print(f"[ERROR] Failed on: {ann_path}")
                    print("=" * 80)
                    traceback.print_exc()
                    if args.on_error == "continue":
                        continue
                    raise

    # Run splits
    overall["ann_total"] = len(train_anns) + (len(val_anns) if val_root is not None else 0)

    convert_split("train", train_root, train_anns)
    if val_root is not None:
        convert_split("val", val_root, val_anns)

    # Final summary JSON
    summary: Dict[str, Any] = {
        "outdir": str(outdir),
        "workers": workers,
        "images_action": args.images_action,
        "include_empty": bool(args.include_empty),
        "strict_size": bool(args.strict_size),
        "on_missing_image": args.on_missing_image,
        "on_outside": args.on_outside,
        "on_degenerate": args.on_degenerate,

        "train_root": str(train_root),
        "val_root": str(val_root) if val_root is not None else "",

        "class_map": class_map,
        "skip_classes": skip_classes,
        "only_classes": only_classes,

        "output_classes": class_names,
        "output_class_count": len(class_names),

        "splits": splits,
        "overall": overall,
    }

    (outdir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[OK] Done.\n"
        f"  Train: ann_total={splits['train']['ann_total']} ann_processed={splits['train']['ann_processed']} "
        f"labels={splits['train']['labels_written']} boxes={splits['train']['boxes_written']}\n"
        + (f"  Val:   ann_total={splits['val']['ann_total']} ann_processed={splits['val']['ann_processed']} "
           f"labels={splits['val']['labels_written']} boxes={splits['val']['boxes_written']}\n"
           if "val" in splits else "")
        + f"  Overall: ann_total={overall['ann_total']} boxes={overall['boxes_written']}\n"
        f"  classes.txt + data.yaml + conversion_summary.json written to: {outdir}"
    )


if __name__ == "__main__":
    main()
