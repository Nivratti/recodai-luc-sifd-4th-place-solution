#!/usr/bin/env python3
"""
Generate inter-panel reuse validation dataset from:
- Compound figures (images)
- Instance masks per figure (.npy): shape (N,H,W) or (H,W,N), binary
- Panel boxes from LabelMe JSON (rectangles; polygon also supported via bbox)

Outputs:
- ops.jsonl: one record per instance-channel (operation), includes component -> panel assignment
- pairs.csv: positive panel pairs from operations spanning >=2 panels
- panels.csv: panel metadata (bbox, label)
- panel_crops/: crops per panel
- panel_masks/: per operation per panel binary masks (crop coordinates)
- debug_overlays/: overlay image showing panels + colored assigned components (quick QA)

Requires:
- numpy
- pillow
Optional (recommended):
- opencv-python or opencv-python-headless (fast connected components)
Fallback:
- scipy (ndimage.label) if cv2 not available
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy import ndimage as ndi  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


@dataclass
class Panel:
    panel_id: str
    label: str
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2 (x2,y2 exclusive)


@dataclass
class ComponentAssign:
    comp_id: int
    area: int
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    centroid: Tuple[float, float]
    best_panel_id: Optional[str]
    best_overlap: int
    best_frac: float


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def shrink_bbox(b: Tuple[int, int, int, int], shrink: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1s = _clamp(x1 + shrink, 0, W)
    y1s = _clamp(y1 + shrink, 0, H)
    x2s = _clamp(x2 - shrink, 0, W)
    y2s = _clamp(y2 - shrink, 0, H)
    if x2s <= x1s or y2s <= y1s:
        return b
    return (x1s, y1s, x2s, y2s)


def bbox_from_points(points: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = int(np.floor(min(xs)))
    y1 = int(np.floor(min(ys)))
    x2 = int(np.ceil(max(xs)))
    y2 = int(np.ceil(max(ys)))
    # make x2,y2 exclusive by +1 if points are integer-like and same edge:
    return (x1, y1, x2, y2)


def load_labelme_panels(json_path: Path, panel_id_order: str = "yx") -> Tuple[List[Panel], int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    H = int(data.get("imageHeight"))
    W = int(data.get("imageWidth"))
    shapes = data.get("shapes", [])

    panels: List[Panel] = []
    for sh in shapes:
        label = str(sh.get("label", "")).strip()
        points = sh.get("points", [])
        shape_type = sh.get("shape_type", None)

        if not points or len(points) < 2:
            continue

        if shape_type == "rectangle" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[0], points[1]
            x1i, x2i = int(round(min(x1, x2))), int(round(max(x1, x2)))
            y1i, y2i = int(round(min(y1, y2))), int(round(max(y1, y2)))
            bbox = (x1i, y1i, x2i, y2i)
        else:
            # polygon/other -> bbox
            bbox = bbox_from_points(points)

        x1, y1, x2, y2 = bbox
        x1 = _clamp(x1, 0, W)
        x2 = _clamp(x2, 0, W)
        y1 = _clamp(y1, 0, H)
        y2 = _clamp(y2, 0, H)
        if x2 <= x1 or y2 <= y1:
            continue

        panels.append(Panel(panel_id="__tmp__", label=label, bbox=(x1, y1, x2, y2)))

    # sort panels for stable ids
    if panel_id_order == "yx":
        panels.sort(key=lambda p: (p.bbox[1], p.bbox[0], p.bbox[3], p.bbox[2]))
    elif panel_id_order == "xy":
        panels.sort(key=lambda p: (p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]))

    for i, p in enumerate(panels):
        p.panel_id = f"p{i:03d}"

    return panels, W, H


def load_mask_instances(npy_path: Path, H: int, W: int) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D instance mask, got {arr.ndim}D at {npy_path}")

    # Normalize to (N,H,W)
    if arr.shape[1] == H and arr.shape[2] == W:
        # (N,H,W)
        inst = arr
    elif arr.shape[0] == H and arr.shape[1] == W:
        # (H,W,N)
        inst = np.transpose(arr, (2, 0, 1))
    else:
        # try heuristic
        if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
            inst = arr
        elif arr.shape[2] <= 64 and arr.shape[0] > 64 and arr.shape[1] > 64:
            inst = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Cannot infer mask layout for {npy_path} with shape {arr.shape} vs H,W=({H},{W})")

    inst = (inst > 0).astype(np.uint8)
    return inst


def connected_components(mask2d: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]], List[int], List[Tuple[float,float]]]:
    """
    Returns:
      labels: int32 HxW (0=background, 1..K components)
      bboxes: list of (x1,y1,x2,y2) per component id in 1..K
      areas: list of areas per component id
      centroids: list of (cx,cy) per component id
    """
    if mask2d.dtype != np.uint8:
        mask2d = mask2d.astype(np.uint8)

    if _HAS_CV2:
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask2d, connectivity=8)
        labels = labels.astype(np.int32)
        bboxes: List[Tuple[int,int,int,int]] = []
        areas: List[int] = []
        centroids: List[Tuple[float,float]] = []
        for cid in range(1, num):
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            w = int(stats[cid, cv2.CC_STAT_WIDTH])
            h = int(stats[cid, cv2.CC_STAT_HEIGHT])
            area = int(stats[cid, cv2.CC_STAT_AREA])
            cx, cy = float(cents[cid, 0]), float(cents[cid, 1])
            bboxes.append((x, y, x + w, y + h))
            areas.append(area)
            centroids.append((cx, cy))
        return labels, bboxes, areas, centroids

    if _HAS_SCIPY:
        structure = np.ones((3,3), dtype=np.uint8)
        labels, num = ndi.label(mask2d, structure=structure)
        labels = labels.astype(np.int32)
        bboxes: List[Tuple[int,int,int,int]] = []
        areas: List[int] = []
        centroids: List[Tuple[float,float]] = []
        for cid in range(1, num + 1):
            ys, xs = np.where(labels == cid)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            area = int(xs.size)
            cx, cy = float(xs.mean()), float(ys.mean())
            bboxes.append((x1, y1, x2, y2))
            areas.append(area)
            centroids.append((cx, cy))
        # scipy labels are 1..num (but we might have skipped empties; keep as-is)
        return labels, bboxes, areas, centroids

    raise RuntimeError("Need either opencv-python (preferred) or scipy installed for connected components.")


def intersect(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> Optional[Tuple[int,int,int,int]]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def color_from_ids(fig_id: str, inst_id: int, comp_id: int) -> Tuple[int,int,int]:
    # deterministic pseudo-random
    h = abs(hash((fig_id, inst_id, comp_id))) % (256**3)
    r = (h // (256*256)) % 256
    g = (h // 256) % 256
    b = h % 256
    # avoid too-dark
    r = max(r, 60); g = max(g, 60); b = max(b, 60)
    return (r, g, b)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate inter-panel reuse dataset from LabelMe panels + instance masks.")
    ap.add_argument("--supp-images-dir", type=Path, required=True, help="Directory with compound figure images (*.png/*.jpg).")
    ap.add_argument("--supp-masks-dir", type=Path, required=True, help="Directory with instance masks (*.npy), per figure.")
    ap.add_argument("--labelme-json-dir", type=Path, required=True, help="Directory with LabelMe JSONs (panel boxes).")
    ap.add_argument("--out", type=Path, required=True, help="Output directory.")

    ap.add_argument("--panel-id-order", choices=["yx", "xy"], default="yx", help="Panel ID ordering.")
    ap.add_argument("--panel-shrink-px", type=int, default=6, help="Shrink panel bbox by this many pixels before overlap tests.")
    ap.add_argument("--min-comp-pixels", type=int, default=150, help="Ignore components smaller than this.")
    ap.add_argument("--overlap-frac-thresh", type=float, default=0.20, help="Assign component to panel if overlap/area >= this.")
    ap.add_argument("--save-debug", action="store_true", help="Save debug overlays (recommended).")
    ap.add_argument("--save-crops", action="store_true", help="Save panel crops.")
    ap.add_argument("--save-panel-masks", action="store_true", help="Save per-op per-panel masks (cropped).")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N figures (0 = all).")

    args = ap.parse_args()

    supp_images_dir: Path = args.supp_images_dir
    supp_masks_dir: Path = args.supp_masks_dir
    labelme_json_dir: Path = args.labelme_json_dir
    out_dir: Path = args.out

    ensure_dir(out_dir)
    crops_dir = out_dir / "panel_crops"
    pmasks_dir = out_dir / "panel_masks"
    debug_dir = out_dir / "debug_overlays"
    ensure_dir(crops_dir); ensure_dir(pmasks_dir); ensure_dir(debug_dir)

    ops_path = out_dir / "ops.jsonl"
    pairs_path = out_dir / "pairs.csv"
    panels_path = out_dir / "panels.csv"
    summary_path = out_dir / "summary.json"

    # Collect figures by matching stems across image/mask/json
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        img_paths.extend(sorted(supp_images_dir.glob(ext)))
    img_by_stem = {p.stem: p for p in img_paths}

    json_paths = sorted(labelme_json_dir.glob("*.json"))
    json_by_stem = {p.stem: p for p in json_paths}

    mask_paths = sorted(supp_masks_dir.glob("*.npy"))
    mask_by_stem = {p.stem: p for p in mask_paths}

    stems = sorted(set(img_by_stem) & set(json_by_stem) & set(mask_by_stem))
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]

    if not stems:
        print("[ERROR] No matching figure stems found across image/mask/json dirs.")
        print(f"  images: {supp_images_dir}")
        print(f"  masks : {supp_masks_dir}")
        print(f"  json  : {labelme_json_dir}")
        return 2

    it = tqdm(stems, desc="Figures") if _HAS_TQDM else stems

    # outputs
    ops_f = ops_path.open("w", encoding="utf-8")
    pairs_f = pairs_path.open("w", newline="", encoding="utf-8")
    panels_f = panels_path.open("w", newline="", encoding="utf-8")

    pairs_writer = csv.DictWriter(pairs_f, fieldnames=[
        "figure_id", "instance_id", "op_type", "panel_a", "panel_b",
        "panel_a_label", "panel_b_label"
    ])
    pairs_writer.writeheader()

    panels_writer = csv.DictWriter(panels_f, fieldnames=[
        "figure_id", "panel_id", "panel_label", "x1", "y1", "x2", "y2"
    ])
    panels_writer.writeheader()

    summary = {
        "num_figures": 0,
        "num_ops": 0,
        "num_ops_inter": 0,
        "num_ops_intra": 0,
        "num_ops_hybrid": 0,
        "num_ops_unassigned": 0,
        "num_positive_pairs": 0,
    }

    for stem in it:
        fig_id = stem
        img_path = img_by_stem[stem]
        json_path = json_by_stem[stem]
        mask_path = mask_by_stem[stem]

        # load panels
        panels, Wj, Hj = load_labelme_panels(json_path, panel_id_order=args.panel_id_order)
        if not panels:
            continue

        # write panel metadata
        panel_by_id = {p.panel_id: p for p in panels}
        for p in panels:
            x1, y1, x2, y2 = p.bbox
            panels_writer.writerow({
                "figure_id": fig_id,
                "panel_id": p.panel_id,
                "panel_label": p.label,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

        # load image (for crops/debug)
        img = Image.open(img_path).convert("RGB")
        Wi, Hi = img.size
        if (Wi, Hi) != (Wj, Hj):
            # This should not happen in your current data, but keep safe.
            print(f"[WARN] Size mismatch {fig_id}: image={Wi}x{Hi}, labelme={Wj}x{Hj}. Using image size.")
            W, H = Wi, Hi
        else:
            W, H = Wj, Hj

        # load instance masks
        inst_masks = load_mask_instances(mask_path, H=H, W=W)  # (N,H,W)
        N = int(inst_masks.shape[0])

        # pre-shrink panel bboxes for overlap logic
        shrunk_panel_bbox = {
            p.panel_id: shrink_bbox(p.bbox, args.panel_shrink_px, W=W, H=H)
            for p in panels
        }

        # optionally save crops (once per panel)
        if args.save_crops:
            fig_crop_dir = crops_dir / fig_id
            ensure_dir(fig_crop_dir)
            for p in panels:
                crop = img.crop(p.bbox)
                crop.save(fig_crop_dir / f"{p.panel_id}_{p.label}.png")

        # debug overlay base
        overlay_img = img.copy()
        draw = ImageDraw.Draw(overlay_img, "RGBA")
        # draw panel boxes
        for p in panels:
            x1,y1,x2,y2 = p.bbox
            draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,180), width=3)
            draw.text((x1+3, y1+3), f"{p.panel_id}:{p.label}", fill=(0,255,0,220))

        figure_positive_pairs = 0
        any_ops = False

        for inst_id in range(N):
            m = inst_masks[inst_id]
            if int(m.sum()) == 0:
                continue

            labels, bboxes, areas, cents = connected_components(m)
            # labels contains component ids. bboxes/areas/cents are aligned with component id order:
            # cv2: comp_id = 1..K matches index 0..K-1 in lists (cid-1)
            # scipy: same intention; we assume dense enough in practice.

            comp_assigns: List[ComponentAssign] = []
            panel_to_compids: Dict[str, List[int]] = {}

            # determine K from labels max
            K = int(labels.max())
            if K == 0:
                continue

            for cid in range(1, K+1):
                # get stats from lists if available; otherwise derive from labels
                idx = cid - 1
                if idx < len(areas):
                    area = int(areas[idx])
                    bbox = bboxes[idx]
                    cx, cy = cents[idx]
                else:
                    ys, xs = np.where(labels == cid)
                    if xs.size == 0:
                        continue
                    area = int(xs.size)
                    bbox = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
                    cx, cy = float(xs.mean()), float(ys.mean())

                if area < args.min_comp_pixels:
                    continue

                best_panel = None
                best_overlap = 0
                best_frac = 0.0

                # component bbox
                cb = bbox

                for p in panels:
                    pb = shrunk_panel_bbox[p.panel_id]
                    inter_box = intersect(cb, pb)
                    if inter_box is None:
                        continue
                    x1,y1,x2,y2 = inter_box
                    sub = labels[y1:y2, x1:x2]
                    overlap = int(np.count_nonzero(sub == cid))
                    if overlap <= 0:
                        continue
                    frac = overlap / float(area)
                    if frac > best_frac:
                        best_frac = frac
                        best_overlap = overlap
                        best_panel = p.panel_id

                if best_panel is not None and best_frac >= args.overlap_frac_thresh:
                    panel_to_compids.setdefault(best_panel, []).append(cid)

                comp_assigns.append(ComponentAssign(
                    comp_id=cid,
                    area=area,
                    bbox=cb,
                    centroid=(cx, cy),
                    best_panel_id=best_panel if (best_panel is not None and best_frac >= args.overlap_frac_thresh) else None,
                    best_overlap=best_overlap,
                    best_frac=float(best_frac),
                ))

            assigned_panels = sorted([pid for pid, cids in panel_to_compids.items() if len(cids) > 0])
            if not assigned_panels:
                op_type = "unassigned"
            else:
                if len(assigned_panels) >= 2:
                    # hybrid heuristic: inter + some panel has multiple blobs
                    if any(len(panel_to_compids[pid]) >= 2 for pid in assigned_panels):
                        op_type = "hybrid"
                    else:
                        op_type = "inter"
                else:
                    # one panel only; intra if multiple components in same panel
                    only_pid = assigned_panels[0]
                    op_type = "intra" if len(panel_to_compids.get(only_pid, [])) >= 2 else "single"

            # Build op record
            op = {
                "figure_id": fig_id,
                "instance_id": int(inst_id),
                "op_type": op_type,
                "panels_involved": assigned_panels,
                "components": [
                    {
                        "comp_id": int(c.comp_id),
                        "area": int(c.area),
                        "bbox": list(map(int, c.bbox)),
                        "centroid": [float(c.centroid[0]), float(c.centroid[1])],
                        "assigned_panel_id": c.best_panel_id,
                        "best_overlap": int(c.best_overlap),
                        "best_frac": float(round(c.best_frac, 6)),
                    }
                    for c in comp_assigns
                ],
            }
            ops_f.write(json.dumps(op, ensure_ascii=False) + "\n")
            any_ops = True
            summary["num_ops"] += 1

            if op_type == "inter":
                summary["num_ops_inter"] += 1
            elif op_type == "intra":
                summary["num_ops_intra"] += 1
            elif op_type == "hybrid":
                summary["num_ops_hybrid"] += 1
            elif op_type == "unassigned":
                summary["num_ops_unassigned"] += 1

            # save per-op per-panel masks (cropped)
            if args.save_panel_masks and assigned_panels:
                base_dir = pmasks_dir / fig_id / f"instance_{inst_id:03d}"
                ensure_dir(base_dir)
                for pid in assigned_panels:
                    comp_ids = panel_to_compids.get(pid, [])
                    if not comp_ids:
                        continue
                    p = panel_by_id[pid]
                    x1,y1,x2,y2 = p.bbox
                    sub_labels = labels[y1:y2, x1:x2]
                    # union of assigned components within crop
                    mask_crop = np.isin(sub_labels, np.array(comp_ids, dtype=np.int32)).astype(np.uint8) * 255
                    out_png = base_dir / f"{pid}.png"
                    Image.fromarray(mask_crop, mode="L").save(out_png)

            # generate positive pairs from inter/hybrid (>=2 panels)
            if len(assigned_panels) >= 2:
                for i in range(len(assigned_panels)):
                    for j in range(i+1, len(assigned_panels)):
                        a = assigned_panels[i]
                        b = assigned_panels[j]
                        la = panel_by_id[a].label
                        lb = panel_by_id[b].label
                        pairs_writer.writerow({
                            "figure_id": fig_id,
                            "instance_id": int(inst_id),
                            "op_type": op_type,
                            "panel_a": a,
                            "panel_b": b,
                            "panel_a_label": la,
                            "panel_b_label": lb,
                        })
                        summary["num_positive_pairs"] += 1
                        figure_positive_pairs += 1

            # draw assigned components for debug overlay
            if args.save_debug and comp_assigns:
                # create transparent layer for this instance
                for c in comp_assigns:
                    if c.best_panel_id is None:
                        continue
                    cid = c.comp_id
                    r,g,b = color_from_ids(fig_id, inst_id, cid)
                    # paint component pixels with alpha
                    ys, xs = np.where(labels == cid)
                    if xs.size == 0:
                        continue
                    # draw sparse pixels (fast enough for this dataset)
                    for x, y in zip(xs[::3], ys[::3]):  # decimate a bit to keep it lighter
                        draw.point((int(x), int(y)), fill=(r,g,b,140))

        if args.save_debug and any_ops:
            ensure_dir(debug_dir)
            overlay_img.save(debug_dir / f"{fig_id}.png")

        if any_ops:
            summary["num_figures"] += 1

    ops_f.close()
    pairs_f.close()
    panels_f.close()

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[OK] Wrote:")
    print(f"  {ops_path}")
    print(f"  {pairs_path}")
    print(f"  {panels_path}")
    print(f"  {summary_path}")
    if args.save_crops:
        print(f"  {crops_dir}/...")
    if args.save_panel_masks:
        print(f"  {pmasks_dir}/...")
    if args.save_debug:
        print(f"  {debug_dir}/...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
