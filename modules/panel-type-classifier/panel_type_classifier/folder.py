from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .classifier import PanelTypeClassifier
from .viz import list_images, save_bar_counts, save_samples_grid

def classify_folder(
    root: Path,
    out_dir: Path,
    batch_size: int = 16,
    recursive: bool = True,
    save_probs: bool = False,
    topk_probs: int = 0,
    show_filename: bool = False,
    n_samples_per_cat: int = 8,
    seed: Optional[int] = None,
) -> Dict[str, Path]:
    """Classify a folder and save CSV + basic visualizations.

    Returns dict of produced file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = PanelTypeClassifier()

    paths = list_images(Path(root), recursive=recursive)
    rows = clf.predict_paths(paths, batch_size=batch_size, return_probs=save_probs, topk_probs=topk_probs)
    df = pd.DataFrame(rows)

    out_csv = out_dir / "predictions.csv"
    df.to_csv(out_csv, index=False)

    produced: Dict[str, Path] = {"predictions_csv": out_csv}

    if len(df) > 0:
        # Charts
        out_group_png = out_dir / "counts_group.png"
        save_bar_counts(df["group_pred"], out_group_png, "Group: imaging vs non_imaging")
        produced["counts_group_png"] = out_group_png

        out_sub_png = out_dir / "counts_subtype.png"
        df_im = df[df["group_pred"] == "imaging"].copy()
        if len(df_im) > 0:
            save_bar_counts(df_im["subtype_pred"], out_sub_png, "Subtype (imaging only)")
            produced["counts_subtype_png"] = out_sub_png

        # Montages
        out_group_samples = out_dir / "samples_group.png"
        save_samples_grid(
            df, "group_pred", out_group_samples,
            n_per_cat=n_samples_per_cat, seed=seed,
            extra_title_cols=["group_conf", "subtype_pred", "subtype_conf"],
            show_filename=show_filename
        )
        produced["samples_group_png"] = out_group_samples

        out_sub_samples = out_dir / "samples_subtype.png"
        if len(df_im) > 0:
            save_samples_grid(
                df_im, "subtype_pred", out_sub_samples,
                n_per_cat=n_samples_per_cat, seed=seed,
                extra_title_cols=["subtype_conf", "group_conf"],
                show_filename=show_filename
            )
            produced["samples_subtype_png"] = out_sub_samples

    return produced
