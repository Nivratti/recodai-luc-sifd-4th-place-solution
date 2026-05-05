from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .metrics import Curve, score_summary


def _try_import_matplotlib():
    """Import matplotlib in a headless-safe way.

    Returns:
        plt module, or None if matplotlib isn't available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_curve_json(out_path: Path, *, pr: Optional[Curve], roc: Optional[Curve], extra: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "precision_recall": pr.to_dict() if pr is not None else None,
        "roc": roc.to_dict() if roc is not None else None,
    }
    payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_benchmark_plots(
    out_dir: Path,
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: Sequence[float],
    title_prefix: str = "",
) -> Dict[str, str]:
    """Save basic evaluation plots (robust for imbalanced datasets).

    Saves into: <out_dir>/plots/

    Returns:
        Dict of {plot_name: relative_path}
    """
    plots_dir = out_dir / "plots"
    _ensure_dir(plots_dir)

    yt = np.asarray(list(y_true), dtype=np.int32)
    yp = np.asarray(list(y_pred), dtype=np.int32)
    sc = np.asarray(list(scores), dtype=np.float64)

    # Filter non-finite scores for score-based plots
    finite = np.isfinite(sc)
    yt_f = yt[finite]
    sc_f = sc[finite]

    score_sum, pr, roc = score_summary(yt_f, sc_f)
    _save_curve_json(
        plots_dir / "curves.json",
        pr=pr,
        roc=roc,
        extra={"score_summary": score_sum.to_dict()},
    )

    plt = _try_import_matplotlib()
    if plt is None:
        return {"curves_json": str((plots_dir / "curves.json").relative_to(out_dir))}

    saved: Dict[str, str] = {"curves_json": str((plots_dir / "curves.json").relative_to(out_dir))}

    # Confusion matrix
    try:
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        cm = np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm)
        ax.set_title((title_prefix + " " if title_prefix else "") + "Confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no_match", "match"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no_match", "match"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(int(v)), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out_path = plots_dir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved["confusion_matrix"] = str(out_path.relative_to(out_dir))
    except Exception:
        pass

    # Score histogram (pos vs neg)
    try:
        if yt_f.size > 0:
            pos = sc_f[yt_f == 1]
            neg = sc_f[yt_f == 0]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(neg, bins=50, alpha=0.6, label="no_match")
            ax.hist(pos, bins=50, alpha=0.6, label="match")
            ax.legend()
            ax.set_title((title_prefix + " " if title_prefix else "") + "Score distribution")
            ax.set_xlabel("score")
            ax.set_ylabel("count")
            out_path = plots_dir / "score_hist.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved["score_hist"] = str(out_path.relative_to(out_dir))
    except Exception:
        pass

    # PR curve
    try:
        if pr is not None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(pr.x, pr.y)
            ax.set_title((title_prefix + " " if title_prefix else "") + "Precision-Recall")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            out_path = plots_dir / "pr_curve.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved["pr_curve"] = str(out_path.relative_to(out_dir))
    except Exception:
        pass

    # ROC curve
    try:
        if roc is not None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(roc.x, roc.y)
            ax.plot([0, 1], [0, 1])
            ax.set_title((title_prefix + " " if title_prefix else "") + "ROC")
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            out_path = plots_dir / "roc_curve.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved["roc_curve"] = str(out_path.relative_to(out_dir))
    except Exception:
        pass

    return saved
