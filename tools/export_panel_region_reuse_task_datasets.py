#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from PIL import Image


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def relpath_str(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        # allow paths outside base too (will include ..)
        return str(path.resolve().relative_to(base.resolve())) if base.resolve() in path.resolve().parents else str(path)


def read_panels_csv(panels_csv: Path) -> Dict[str, Dict[str, dict]]:
    panels: Dict[str, Dict[str, dict]] = {}
    with panels_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fig = row["figure_id"]
            pid = row["panel_id"]
            panels.setdefault(fig, {})[pid] = {
                "label": row["panel_label"],
                "bbox": (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
            }
    return panels


def all_panel_pairs(panel_ids: List[str]) -> List[Tuple[str, str]]:
    out = []
    for i in range(len(panel_ids)):
        for j in range(i + 1, len(panel_ids)):
            out.append((panel_ids[i], panel_ids[j]))
    return out


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be copy|link")


def find_panel_crop(panel_crops_dir: Path, fig: str, pid: str) -> Optional[Path]:
    d = panel_crops_dir / fig
    if not d.exists():
        return None
    matches = sorted(d.glob(f"{pid}_*.png"))
    return matches[0] if matches else None


def crop_panel_from_source(
    supp_images_dir: Optional[Path],
    fig: str,
    pid: str,
    bbox: Tuple[int, int, int, int],
    dst_png: Path,
) -> Optional[Path]:
    """
    Fallback if panel_crops are not present.
    Needs --supp-images-dir with images named <fig>.(png/jpg/...)
    """
    if supp_images_dir is None:
        return None

    img_path = None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        p = supp_images_dir / f"{fig}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        return None

    ensure_dir(dst_png.parent)
    img = Image.open(img_path).convert("RGB")
    crop = img.crop(bbox)
    crop.save(dst_png)
    return dst_png


def find_panel_mask(panel_masks_dir: Path, fig: str, inst_id: int, pid: str) -> Optional[Path]:
    p = panel_masks_dir / fig / f"instance_{inst_id:03d}" / f"{pid}.png"
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export task-specific datasets (interpanel pairs + intrapanel samples) with clean folder layout."
    )
    ap.add_argument("--gt-root", type=Path, required=True,
                    help="Root output dir from GT script (contains ops.jsonl, panels.csv, panel_crops, panel_masks).")
    ap.add_argument("--supp-images-dir", type=Path, default=None,
                    help="Optional: directory with source figure images for cropping panels if panel_crops missing.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Where to write tasks/. Default: <gt-root>/tasks")
    ap.add_argument("--neg-per-pos", type=int, default=10,
                    help="Negatives to sample per positive (per figure) for interpanel no_match.")
    ap.add_argument("--neg-same-label", action="store_true",
                    help="Sample interpanel negatives only when panel labels match (Microscopy/Blots).")
    ap.add_argument("--include-hybrid", action="store_true",
                    help="Include hybrid ops as interpanel positives.")
    ap.add_argument("--materialize", choices=["copy", "link", "none"], default="copy",
                    help="copy/link files into tasks folders, or 'none' to only emit CSVs with paths.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    gt_root: Path = args.gt_root
    out_root: Path = args.out_root or (gt_root / "tasks")
    ensure_dir(out_root)

    ops_jsonl = gt_root / "ops.jsonl"
    panels_csv = gt_root / "panels.csv"
    panel_crops_dir = gt_root / "panel_crops"
    panel_masks_dir = gt_root / "panel_masks"

    if not ops_jsonl.exists() or not panels_csv.exists():
        raise SystemExit("Missing ops.jsonl or panels.csv in --gt-root")

    panels = read_panels_csv(panels_csv)

    # ---- Parse ops: build positives, intra positives, and "panels involved in any op" ----
    inter_pos: List[dict] = []
    intra_pos: List[dict] = []
    pos_pairs_by_fig: Dict[str, Set[Tuple[str, str]]] = {}
    panels_in_any_op: Dict[str, Set[str]] = {}

    with ops_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            op = json.loads(line)
            fig = op["figure_id"]
            inst_id = int(op["instance_id"])
            op_type = op["op_type"]
            pids = op.get("panels_involved", []) or []

            if fig not in panels:
                continue

            # track panels involved in ANY operation (clean negatives for intrapanel)
            if op_type != "unassigned" and pids:
                panels_in_any_op.setdefault(fig, set()).update(pids)

            # intra positives: include both "intra" and "single" by default
            if op_type in {"intra", "single"} and len(pids) == 1:
                pid = pids[0]
                intra_pos.append({
                    "figure_id": fig,
                    "instance_id": inst_id,
                    "op_type": op_type,              # keep original
                    "panel_id": pid,
                    "panel_label": panels[fig][pid]["label"],
                    "intra_kind": "strong" if op_type == "intra" else "weak"
                })
                continue

            # inter positives
            allow = (op_type == "inter") or (args.include_hybrid and op_type == "hybrid")
            if allow and len(pids) >= 2:
                pids_sorted = sorted(pids)
                for (a, b) in all_panel_pairs(pids_sorted):
                    pos_pairs_by_fig.setdefault(fig, set()).add((a, b))
                    inter_pos.append({
                        "figure_id": fig,
                        "instance_id": inst_id,
                        "op_type": op_type,
                        "panel_a": a,
                        "panel_b": b,
                        "label_a": panels[fig][a]["label"],
                        "label_b": panels[fig][b]["label"],
                    })

    # ---- Inter negatives (no_match): all non-positive pairs sampled per figure ----
    inter_neg: List[dict] = []
    for fig, fig_panels in panels.items():
        pids = sorted(fig_panels.keys())
        if len(pids) < 2:
            continue

        all_pairs = all_panel_pairs(pids)
        pos_set = pos_pairs_by_fig.get(fig, set())
        candidates = [(a, b) for (a, b) in all_pairs if (a, b) not in pos_set]

        if args.neg_same_label:
            candidates = [(a, b) for (a, b) in candidates
                          if fig_panels[a]["label"] == fig_panels[b]["label"]]

        pos_count = sum(1 for r in inter_pos if r["figure_id"] == fig)
        if pos_count == 0 or not candidates:
            continue

        need = min(len(candidates), pos_count * args.neg_per_pos)
        sampled = random.sample(candidates, k=need) if need < len(candidates) else candidates

        for (a, b) in sampled:
            inter_neg.append({
                "figure_id": fig,
                "instance_id": None,      # no instance id
                "op_type": "no_match",
                "panel_a": a,
                "panel_b": b,
                "label_a": fig_panels[a]["label"],
                "label_b": fig_panels[b]["label"],
            })

    # ---- Intrapanel negatives: panels NOT involved in any op ----
    intra_neg: List[dict] = []
    for fig, fig_panels in panels.items():
        involved = panels_in_any_op.get(fig, set())
        for pid, meta in fig_panels.items():
            if pid in involved:
                continue
            intra_neg.append({
                "figure_id": fig,
                "instance_id": None,
                "op_type": "negative",
                "panel_id": pid,
                "panel_label": meta["label"],
            })

    # ---- Helper to obtain panel image (either from panel_crops or source crop fallback) ----
    def get_or_make_panel_png(fig: str, pid: str, dst_png: Path) -> Optional[Path]:
        crop_path = find_panel_crop(panel_crops_dir, fig, pid)
        if crop_path and crop_path.exists():
            if args.materialize == "none":
                return crop_path
            copy_or_link(crop_path, dst_png, args.materialize)
            return dst_png

        # fallback: crop from source figures
        bbox = panels[fig][pid]["bbox"]
        made = crop_panel_from_source(args.supp_images_dir, fig, pid, bbox, dst_png)
        return made

    # =========================
    # Export Interpanel
    # =========================
    inter_dir = out_root / "interpanel"
    match_dir = inter_dir / "match"
    nomatch_dir = inter_dir / "no_match"
    ensure_dir(match_dir)
    ensure_dir(nomatch_dir)

    match_csv = inter_dir / "match_pairs.csv"
    nomatch_csv = inter_dir / "no_match_pairs.csv"

    match_fields = ["figure_id", "instance_id", "panel_a", "panel_b", "label_a", "label_b",
                    "A_img", "B_img", "A_mask", "B_mask", "pair_dir"]
    nomatch_fields = ["figure_id", "panel_a", "panel_b", "label_a", "label_b",
                      "A_img", "B_img", "pair_dir"]

    with match_csv.open("w", newline="", encoding="utf-8") as f_match, \
         nomatch_csv.open("w", newline="", encoding="utf-8") as f_no:

        w_match = csv.DictWriter(f_match, fieldnames=match_fields)
        w_no = csv.DictWriter(f_no, fieldnames=nomatch_fields)
        w_match.writeheader()
        w_no.writeheader()

        # ---- write match (positives) ----
        for r in inter_pos:
            fig = r["figure_id"]
            inst = int(r["instance_id"])
            a, b = r["panel_a"], r["panel_b"]

            pair_leaf = f"{a}_{b}"
            pair_folder = match_dir / fig / f"inst{inst:03d}" / pair_leaf

            A_png = pair_folder / "A.png"
            B_png = pair_folder / "B.png"
            A_mask_out = pair_folder / "A_mask.png"
            B_mask_out = pair_folder / "B_mask.png"
            meta_out = pair_folder / "meta.json"

            # images
            A_src_or_dst = get_or_make_panel_png(fig, a, A_png)
            B_src_or_dst = get_or_make_panel_png(fig, b, B_png)

            # masks must come from panel_masks (generated by GT script)
            A_mask = find_panel_mask(panel_masks_dir, fig, inst, a)
            B_mask = find_panel_mask(panel_masks_dir, fig, inst, b)

            if args.materialize != "none":
                ensure_dir(pair_folder)
                if A_mask and A_mask.exists():
                    copy_or_link(A_mask, A_mask_out, args.materialize)
                if B_mask and B_mask.exists():
                    copy_or_link(B_mask, B_mask_out, args.materialize)
                meta = {
                    "y": 1,
                    "task": "interpanel",
                    **r,
                }
                meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                A_img_rel = relpath_str(A_png, out_root) if A_src_or_dst else ""
                B_img_rel = relpath_str(B_png, out_root) if B_src_or_dst else ""
                A_mask_rel = relpath_str(A_mask_out, out_root) if (A_mask and A_mask.exists()) else ""
                B_mask_rel = relpath_str(B_mask_out, out_root) if (B_mask and B_mask.exists()) else ""
                pair_dir_rel = relpath_str(pair_folder, out_root)
            else:
                A_img_rel = relpath_str(Path(A_src_or_dst), out_root) if A_src_or_dst else ""
                B_img_rel = relpath_str(Path(B_src_or_dst), out_root) if B_src_or_dst else ""
                A_mask_rel = relpath_str(A_mask, out_root) if (A_mask and A_mask.exists()) else ""
                B_mask_rel = relpath_str(B_mask, out_root) if (B_mask and B_mask.exists()) else ""
                pair_dir_rel = relpath_str(pair_folder, out_root)

            w_match.writerow({
                "figure_id": fig,
                "instance_id": inst,
                "panel_a": a,
                "panel_b": b,
                "label_a": r["label_a"],
                "label_b": r["label_b"],
                "A_img": A_img_rel,
                "B_img": B_img_rel,
                "A_mask": A_mask_rel,
                "B_mask": B_mask_rel,
                "pair_dir": pair_dir_rel,
            })

        # ---- write no_match (negatives) ----
        for r in inter_neg:
            fig = r["figure_id"]
            a, b = r["panel_a"], r["panel_b"]

            pair_leaf = f"{a}_{b}"
            pair_folder = nomatch_dir / fig / pair_leaf

            A_png = pair_folder / "A.png"
            B_png = pair_folder / "B.png"
            meta_out = pair_folder / "meta.json"

            A_src_or_dst = get_or_make_panel_png(fig, a, A_png)
            B_src_or_dst = get_or_make_panel_png(fig, b, B_png)

            if args.materialize != "none":
                ensure_dir(pair_folder)
                meta = {
                    "y": 0,
                    "task": "interpanel",
                    **r,
                }
                meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                A_img_rel = relpath_str(A_png, out_root) if A_src_or_dst else ""
                B_img_rel = relpath_str(B_png, out_root) if B_src_or_dst else ""
                pair_dir_rel = relpath_str(pair_folder, out_root)
            else:
                A_img_rel = relpath_str(Path(A_src_or_dst), out_root) if A_src_or_dst else ""
                B_img_rel = relpath_str(Path(B_src_or_dst), out_root) if B_src_or_dst else ""
                pair_dir_rel = relpath_str(pair_folder, out_root)

            w_no.writerow({
                "figure_id": fig,
                "panel_a": a,
                "panel_b": b,
                "label_a": r["label_a"],
                "label_b": r["label_b"],
                "A_img": A_img_rel,
                "B_img": B_img_rel,
                "pair_dir": pair_dir_rel,
            })

    # =========================
    # Export Intrapanel
    # =========================
    intra_dir = out_root / "intrapanel"
    pos_dir = intra_dir / "positive"
    neg_dir = intra_dir / "negative"
    ensure_dir(pos_dir)
    ensure_dir(neg_dir)

    intra_pos_csv = intra_dir / "positive.csv"
    intra_neg_csv = intra_dir / "negative.csv"

    pos_fields = ["figure_id", "instance_id", "panel_id", "panel_label",
                "op_type", "intra_kind",
                "panel_img", "mask", "sample_dir"]

    neg_fields = ["figure_id", "panel_id", "panel_label",
                  "panel_img", "sample_dir"]

    with intra_pos_csv.open("w", newline="", encoding="utf-8") as f_pos, \
         intra_neg_csv.open("w", newline="", encoding="utf-8") as f_neg:

        w_pos = csv.DictWriter(f_pos, fieldnames=pos_fields)
        w_neg = csv.DictWriter(f_neg, fieldnames=neg_fields)
        w_pos.writeheader()
        w_neg.writeheader()

        # positives: intrapanel copy-move
        for s in intra_pos:
            fig = s["figure_id"]
            inst = int(s["instance_id"])
            pid = s["panel_id"]

            # sample_folder = pos_dir / fig / f"inst{inst:03d}" / pid
            # panel_png = sample_folder / "panel.png"
            # mask_png = sample_folder / "mask.png"
            # meta_out = sample_folder / "meta.json"

            # FLAT FILES under inst folder
            sample_folder = pos_dir / fig / f"inst{inst:03d}"
            panel_png = sample_folder / f"{pid}.png"
            mask_png = sample_folder / f"{pid}_mask.png"
            meta_out = sample_folder / f"{pid}_meta.json"

            panel_src_or_dst = get_or_make_panel_png(fig, pid, panel_png)
            pmask = find_panel_mask(panel_masks_dir, fig, inst, pid)

            if args.materialize != "none":
                ensure_dir(sample_folder)
                if pmask and pmask.exists():
                    copy_or_link(pmask, mask_png, args.materialize)
                meta = {"y": 1, "task": "intrapanel", **s}
                meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                panel_rel = relpath_str(panel_png, out_root) if panel_src_or_dst else ""
                mask_rel = relpath_str(mask_png, out_root) if (pmask and pmask.exists()) else ""
                sample_dir_rel = relpath_str(sample_folder, out_root)
            else:
                panel_rel = relpath_str(Path(panel_src_or_dst), out_root) if panel_src_or_dst else ""
                mask_rel = relpath_str(pmask, out_root) if (pmask and pmask.exists()) else ""
                sample_dir_rel = relpath_str(sample_folder, out_root)

            w_pos.writerow({
                "figure_id": fig,
                "instance_id": inst,
                "panel_id": pid,
                "panel_label": s["panel_label"],
                "panel_img": panel_rel,
                "mask": mask_rel,
                "sample_dir": sample_dir_rel,
            })

        # negatives: panels with NO operation involvement at all
        for s in intra_neg:
            fig = s["figure_id"]
            pid = s["panel_id"]

            sample_folder = neg_dir / fig
            panel_png = sample_folder / f"{pid}.png"
            meta_out = sample_folder / f"{pid}_meta.json"

            panel_src_or_dst = get_or_make_panel_png(fig, pid, panel_png)

            if args.materialize != "none":
                ensure_dir(sample_folder)
                meta = {"y": 0, "task": "intrapanel", **s}
                meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                panel_rel = relpath_str(panel_png, out_root) if panel_src_or_dst else ""
                sample_dir_rel = relpath_str(sample_folder, out_root)
            else:
                panel_rel = relpath_str(Path(panel_src_or_dst), out_root) if panel_src_or_dst else ""
                sample_dir_rel = relpath_str(sample_folder, out_root)

            w_neg.writerow({
                "figure_id": fig,
                "panel_id": pid,
                "panel_label": s["panel_label"],
                "panel_img": panel_rel,
                "sample_dir": sample_dir_rel,
            })

    print("[OK] Interpanel match CSV:", match_csv)
    print("[OK] Interpanel no_match CSV:", nomatch_csv)
    print("[OK] Intrapanel positive CSV:", intra_pos_csv)
    print("[OK] Intrapanel negative CSV:", intra_neg_csv)
    print("[INFO] materialize =", args.materialize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
