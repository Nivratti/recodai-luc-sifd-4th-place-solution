#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panel_reuse_synth.config import load_config, Config, KNOWN_TYPES
from panel_reuse_synth.io_utils import (
    list_images,
    read_image_rgb,
    ensure_dir,
    write_image,
    write_json,
    make_relpath,
    resize_to_wh,
)
from panel_reuse_synth.modules.full_duplicate import generate_full_duplicate
from panel_reuse_synth.modules.no_match import generate_no_match
from panel_reuse_synth.modules.full_overlap_crop import generate_full_overlap_crop
from panel_reuse_synth.modules.partial_overlap import generate_partial_overlap


TYPE_DIR = {
    "FULL_DUPLICATE": "full_duplicate",
    "FULL_OVERLAP_CROP": "full_overlap_crop",
    "PARTIAL_OVERLAP": "partial_overlap",
    "NO_MATCH": "no_match",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--input", required=True, help="Input image file OR directory.")
    p.add_argument("--out", default=None, help="Override output.out_dir from YAML.")
    p.add_argument("--limit-images", type=int, default=None, help="Limit number of source images.")
    p.add_argument("--dry-run", action="store_true", help="Do not write outputs.")
    return p.parse_args()


def _normalize_probs(type_probs: Dict[str, float], enabled_types: List[str]) -> Dict[str, float]:
    probs = {t: float(type_probs.get(t, 0.0)) for t in enabled_types}
    s = sum(probs.values())
    if s <= 0:
        u = 1.0 / max(1, len(enabled_types))
        return {t: u for t in enabled_types}
    return {t: probs[t] / s for t in enabled_types}


def _dump_effective_config(out_dir: Path, cfg: Config, args: argparse.Namespace) -> None:
    obj = asdict(cfg)
    obj["_cli_args"] = vars(args)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _sample_types_for_image(cfg: Config, rng) -> List[str]:
    enabled = cfg.sampling.enabled_types
    if cfg.sampling.strategy == "per_type_counts":
        out: List[str] = []
        for t in enabled:
            out.extend([t] * int(cfg.sampling.per_image_counts.get(t, 0)))
        return out

    n = int(cfg.sampling.pairs_per_image)
    probs = _normalize_probs(cfg.sampling.type_probs, enabled)
    ts = enabled[:]
    ps = [probs[t] for t in ts]
    return list(rng.choice(ts, size=n, replace=True, p=ps))


def _scale_box_xyxy(box, sx: float, sy: float):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]


def main() -> None:
    args = parse_args()
    cfg: Config = load_config(args.config)

    if args.out is not None:
        cfg.output.out_dir = str(args.out)

    input_path = Path(args.input)
    out_dir = Path(cfg.output.out_dir)
    ensure_dir(out_dir)

    if input_path.is_file():
        src_paths = [input_path]
    else:
        src_paths = list_images(input_path, recursive=cfg.input.recursive, exts=cfg.input.exts)
    if args.limit_images is not None:
        src_paths = src_paths[: args.limit_images]

    if not args.dry_run:
        _dump_effective_config(out_dir, cfg, args)

    manifest_path = out_dir / "manifest.jsonl"
    stats_path = out_dir / "stats.json"

    n_src = len(src_paths)
    batch_size = max(1, int(cfg.output.batch_size))
    num_batches = max(1, math.ceil(n_src / batch_size))
    src_pad = max(1, len(str(max(0, n_src - 1))))
    batch_pad = max(1, len(str(max(0, num_batches - 1))))

    def pair_pad_for_type(t: str) -> int:
        if cfg.sampling.strategy == "per_type_counts":
            m = int(cfg.sampling.per_image_counts.get(t, 0))
        else:
            m = int(cfg.sampling.pairs_per_image)
        return max(1, len(str(max(0, m - 1))))

    stats = {
        "out_dir": str(out_dir),
        "num_source_images": n_src,
        "enabled_types": cfg.sampling.enabled_types,
        "strategy": cfg.sampling.strategy,
        "batch_size": batch_size,
        "written_per_type": {t: 0 for t in cfg.sampling.enabled_types},
        "skipped": 0,
        "skip_reasons": {},
        "type_probs_report": _normalize_probs(cfg.sampling.type_probs, cfg.sampling.enabled_types)
        if cfg.sampling.strategy == "probabilistic"
        else None,
    }

    mf = None if args.dry_run else manifest_path.open("w", encoding="utf-8")

    try:
        for src_idx, src in enumerate(tqdm(src_paths, desc="Source images")):
            try:
                img = read_image_rgb(src)
                if img is None:
                    raise ValueError("read_image_rgb returned None")
            except Exception as e:
                stats["skipped"] += 1
                r = f"read_error:{type(e).__name__}"
                stats["skip_reasons"][r] = stats["skip_reasons"].get(r, 0) + 1
                continue

            batch_id = src_idx // batch_size
            batch_name = f"batch_{batch_id:0{batch_pad}d}"
            index_name = f"{src_idx:0{src_pad}d}"

            import numpy as np
            img_rng = np.random.default_rng(cfg.seed + (abs(hash(str(src))) % (2**31)))
            types_for_this_image = _sample_types_for_image(cfg, img_rng)

            local_pair_counter = {t: 0 for t in cfg.sampling.enabled_types}

            for t in types_for_this_image:
                t_dir = TYPE_DIR.get(t, t.lower())
                pidx = local_pair_counter[t]
                local_pair_counter[t] += 1

                pp = pair_pad_for_type(t)
                pair_stem = f"{pidx:0{pp}d}"

                img_dir = out_dir / "images" / t_dir / batch_name / index_name
                msk_dir = out_dir / "masks" / t_dir / batch_name / index_name
                meta_dir = out_dir / "meta" / t_dir / batch_name / index_name

                if not args.dry_run:
                    ensure_dir(img_dir)
                    ensure_dir(msk_dir)
                    ensure_dir(meta_dir)

                ext_img = cfg.output.image_format
                ext_msk = cfg.output.mask_format

                a_img_path = img_dir / f"{pair_stem}_a.{ext_img}"
                b_img_path = img_dir / f"{pair_stem}_b.{ext_img}"
                a_msk_path = msk_dir / f"{pair_stem}_a.{ext_msk}"
                b_msk_path = msk_dir / f"{pair_stem}_b.{ext_msk}"
                meta_path = meta_dir / f"{pair_stem}.json"

                sample_key = f"{t}:{src_idx}:{pidx}:{src.name}"

                if t == "FULL_DUPLICATE":
                    rec = generate_full_duplicate(cfg, src, img, sample_key)
                elif t == "NO_MATCH":
                    rec = generate_no_match(cfg, src, img, sample_key)
                elif t == "FULL_OVERLAP_CROP":
                    rec = generate_full_overlap_crop(cfg, src, img, sample_key)
                elif t == "PARTIAL_OVERLAP":
                    rec = generate_partial_overlap(cfg, src, img, sample_key)
                else:
                    rec = None

                if rec is None:
                    stats["skipped"] += 1
                    r = f"sampling_failed:{t}"
                    stats["skip_reasons"][r] = stats["skip_reasons"].get(r, 0) + 1
                    continue

                A, B, Am, Bm, meta = rec["A_img"], rec["B_img"], rec["A_mask"], rec["B_mask"], rec["meta"]

                # layout block
                meta["layout"] = {
                    "source_index": int(src_idx),
                    "batch_id": int(batch_id),
                    "batch_name": batch_name,
                    "index_name": index_name,
                    "pair_index_in_source_for_type": int(pidx),
                    "pair_stem": pair_stem,
                    "type_dir": t_dir,
                }

                # resizing policy
                if cfg.output.fixed_size is not None:
                    new_w, new_h = cfg.output.fixed_size
                    oldAw, oldAh = A.shape[1], A.shape[0]
                    oldBw, oldBh = B.shape[1], B.shape[0]

                    A = resize_to_wh(A, new_w, new_h, is_mask=False)
                    B = resize_to_wh(B, new_w, new_h, is_mask=False)
                    Am = resize_to_wh(Am, new_w, new_h, is_mask=True)
                    Bm = resize_to_wh(Bm, new_w, new_h, is_mask=True)

                    sxA, syA = (new_w / max(1, oldAw)), (new_h / max(1, oldAh))
                    sxB, syB = (new_w / max(1, oldBw)), (new_h / max(1, oldBh))

                    if "match_region_A_xyxy" in meta:
                        meta["match_region_A_xyxy"] = _scale_box_xyxy(meta.get("match_region_A_xyxy"), sxA, syA)
                    if "match_region_B_xyxy" in meta:
                        meta["match_region_B_xyxy"] = _scale_box_xyxy(meta.get("match_region_B_xyxy"), sxB, syB)

                    meta["shapes_after_resize"] = {
                        "A_hw": [int(A.shape[0]), int(A.shape[1])],
                        "B_hw": [int(B.shape[0]), int(B.shape[1])],
                        "fixed_size": [int(new_w), int(new_h)],
                    }

                if args.dry_run:
                    continue

                write_image(a_img_path, A, fmt=ext_img, jpg_quality=cfg.output.jpg_quality)
                write_image(b_img_path, B, fmt=ext_img, jpg_quality=cfg.output.jpg_quality)
                write_image(a_msk_path, Am, fmt=ext_msk, jpg_quality=cfg.output.jpg_quality)
                write_image(b_msk_path, Bm, fmt=ext_msk, jpg_quality=cfg.output.jpg_quality)

                if cfg.output.write_per_sample_meta:
                    write_json(meta_path, meta)

                line = {
                    "type": t,
                    "type_dir": t_dir,
                    "source_index": int(src_idx),
                    "pair_index_in_source_for_type": int(pidx),
                    "batch": batch_name,
                    "index": index_name,
                    "pair_stem": pair_stem,
                    "source": make_relpath(src, out_dir),
                    "A": make_relpath(a_img_path, out_dir),
                    "B": make_relpath(b_img_path, out_dir),
                    "A_mask": make_relpath(a_msk_path, out_dir),
                    "B_mask": make_relpath(b_msk_path, out_dir),
                    "meta": make_relpath(meta_path, out_dir) if cfg.output.write_per_sample_meta else None,
                }
                mf.write(json.dumps(line, ensure_ascii=False) + "\n")
                stats["written_per_type"][t] += 1

    finally:
        if mf is not None:
            mf.close()

    if not args.dry_run:
        write_json(stats_path, stats)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
