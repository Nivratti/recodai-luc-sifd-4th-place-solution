#!/usr/bin/env python3
"""Classify a folder of panel images and save CSV + visualizations.

Example:
  python scripts/classify_folder.py --root /path/to/panels --out out/panel_type --batch-size 32 --recursive

example:
  python scripts/classify_folder.py --root resources/images --out runs/panel_type_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import sys
from pathlib import Path

# --- local-dev import helper (so scripts work without pip install -e .) ---
try:
    from panel_type_classifier import PanelTypeClassifier
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]  # scripts/.. -> repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from panel_type_classifier import PanelTypeClassifier
        print(f"[panel-type-classifier] Added repo root to sys.path: {repo_root}", file=sys.stderr)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import 'panel_type_classifier'.\n"
            f"Tried adding repo root to sys.path: {repo_root}\n\n"
            "Fix options:\n"
            "  1) From repo root, run:  python scripts/classify_folder.py ...\n"
            "  2) Or install editable:  pip install -e .\n"
            "  3) Ensure folder exists: panel_type_classifier/__init__.py\n"
        ) from e
# -------------------------------------------------------------------------

from panel_type_classifier.viz import list_images, save_bar_counts, save_samples_grid


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Input root folder containing images.")
    ap.add_argument("--out", type=str, required=True, help="Output directory.")
    ap.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recurse into subfolders (default: enabled). Use --no-recursive to disable.",
    )
    ap.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursion (only scan the top-level folder).",
    )
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto).")
    ap.add_argument("--save-probs", action="store_true", help="Save per-class probabilities (many columns).")
    ap.add_argument("--topk-probs", type=int, default=0, help="If >0, keep only top-k probs per head.")
    ap.add_argument("--drop-filename", action="store_true", help="Do not include filename column in CSV.")
    ap.add_argument("--n-samples", type=int, default=8, help="Random samples per category for montage.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for deterministic sampling.")
    ap.add_argument("--thumb-size", type=int, default=320, help="Thumbnail size for montage.")
    ap.add_argument("--ncols", type=int, default=4, help="Columns in montage.")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PanelTypeClassifier")
    clf = PanelTypeClassifier(device=args.device)

    print(f"Loaded PanelTypeClassifier")

    image_paths = list_images(root, recursive=args.recursive)

    if not image_paths:
        print(f"[panel-type-classifier] No images found under: {args.root} (recursive={args.recursive})", file=sys.stderr)
        sys.exit(2)  # non-zero so CI / scripts can detect “nothing to do”
        
    print(f"Found images: {len(image_paths)}")
    
    rows = []
    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(image_paths), bs), desc="Predicting"):
        batch = image_paths[i:i+bs]
        rows.extend(
            clf.predict_paths(
                batch,
                batch_size=len(batch),
                return_probs=args.save_probs,
                topk_probs=args.topk_probs,
            )
        )

    df = pd.DataFrame(rows)
    if args.drop_filename and "filename" in df.columns:
        df = df.drop(columns=["filename"])

    out_csv = out_dir / "predictions.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Summary
    summary = {
        "n_images": int(len(df)),
        "group_counts": df["group_pred"].value_counts().to_dict() if len(df) else {},
    }
    df_im = df[df["group_pred"] == "imaging"].copy()
    if len(df_im):
        summary["subtype_counts"] = df_im["subtype_pred"].value_counts().to_dict()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "summary.json")

    # Charts
    if len(df):
        save_bar_counts(df["group_pred"], out_dir / "counts_group.png", "Group: imaging vs non_imaging")
        if len(df_im):
            save_bar_counts(df_im["subtype_pred"], out_dir / "counts_subtype.png", "Subtype (imaging only)")

        # Montages
        save_samples_grid(
            df,
            category_col="group_pred",
            out_path=out_dir / "samples_group.png",
            n_per_cat=args.n_samples,
            seed=args.seed,
            extra_title_cols=["group_conf", "subtype_pred", "subtype_conf"],
            decimals=2,
            show_filename=not args.drop_filename,
            ncols=args.ncols,
            thumb_size=args.thumb_size,
        )
        if len(df_im):
            save_samples_grid(
                df_im,
                category_col="subtype_pred",
                out_path=out_dir / "samples_subtype.png",
                n_per_cat=args.n_samples,
                seed=args.seed,
                extra_title_cols=["subtype_conf", "group_conf"],
                decimals=2,
                show_filename=not args.drop_filename,
                ncols=args.ncols,
                thumb_size=args.thumb_size,
            )

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
