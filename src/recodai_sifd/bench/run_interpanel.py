from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from .interpanel_dataset import InterpanelDataset
from .metrics import classification_metrics, dice_f1, iou, mask_metrics, score_summary
from .plotting import save_benchmark_plots
from .types import PairExample
from .backends.base import MatcherBackend
from .backends.copy_move_det_keypoint import CopyMoveDetKeypointBackend, CopyMoveDetKeypointBackendConfig

# Optional backend: gmberton/image-matching-models
try:
    from .backends.imm import ImageMatchingModelsBackend, ImageMatchingModelsBackendConfig
except Exception:
    ImageMatchingModelsBackend = None  # type: ignore
    ImageMatchingModelsBackendConfig = None  # type: ignore


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def load_mask(path: Path) -> np.ndarray:
    cv2 = _try_import_cv2()
    if cv2 is not None:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        return (m > 127).astype(np.uint8)
    # pillow fallback
    from PIL import Image
    m = np.array(Image.open(path).convert("L"))
    return (m > 127).astype(np.uint8)


def save_mask(arr: np.ndarray, path: Path) -> None:
    arr_u8 = (arr > 0).astype(np.uint8) * 255
    cv2 = _try_import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(path), arr_u8)
        return
    from PIL import Image
    Image.fromarray(arr_u8).save(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_pair_files(ex: PairExample, out_dir: Path, *, pred_mask_a: Optional[np.ndarray], pred_mask_b: Optional[np.ndarray]) -> None:
    ensure_dir(out_dir)
    shutil.copy2(ex.a_path, out_dir / "A.png")
    shutil.copy2(ex.b_path, out_dir / "B.png")
    if ex.a_mask_path and ex.a_mask_path.exists():
        shutil.copy2(ex.a_mask_path, out_dir / "A_mask_gt.png")
    if ex.b_mask_path and ex.b_mask_path.exists():
        shutil.copy2(ex.b_mask_path, out_dir / "B_mask_gt.png")
    if ex.meta_path and ex.meta_path.exists():
        shutil.copy2(ex.meta_path, out_dir / "meta.json")
    if pred_mask_a is not None:
        save_mask(pred_mask_a, out_dir / "A_mask_pred.png")
    if pred_mask_b is not None:
        save_mask(pred_mask_b, out_dir / "B_mask_pred.png")


def build_backend_copy_move_det_keypoint(args: argparse.Namespace) -> MatcherBackend:
    cfg = CopyMoveDetKeypointBackendConfig(
        descriptor_type=args.descriptor_type,
        alignment_strategy=args.alignment_strategy,
        matching_method=args.matching_method,
        min_keypoints=args.min_keypoints,
        min_area=args.min_area,
        check_flip=not args.no_flip,
        cross_kp_count=args.cross_kp_count,
        keep_image=args.keep_image_for_viz,
        assume_bgr=not args.assume_rgb,
        score_key=args.score_key,
        prep_cache=args.prep_cache,
        prep_cache_dir=args.prep_cache_dir,
        prep_cache_max=args.prep_cache_max,
    )
    return CopyMoveDetKeypointBackend(cfg)


def build_backend_imm(args: argparse.Namespace, out_dir: Path) -> MatcherBackend:
    if ImageMatchingModelsBackend is None or ImageMatchingModelsBackendConfig is None:
        raise SystemExit(
            "IMM backend requested but not importable. "
            "Copy/enable src/recodai_sifd/bench/backends/imm.py and ensure deps are installed."
        )
    cfg = ImageMatchingModelsBackendConfig(
        imm_root=args.imm_root,
        matcher=args.imm_matcher,
        device=args.imm_device,
        max_num_keypoints=args.imm_max_num_keypoints,
        min_inliers=args.imm_min_inliers,
        min_inlier_ratio=args.imm_min_inlier_ratio,
        score_mode=args.imm_score_mode,
        mask_mode=args.imm_mask_mode,
        mask_dilate=args.imm_mask_dilate,
        max_side_sum=int(args.imm_max_side_sum or 0),
        resize_cache_dir=(args.imm_resize_cache_dir or str(out_dir / "_cache" / "imm_resize")),
    )
    return ImageMatchingModelsBackend(cfg)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


CSV_FIELDS = [
    "pair_id",
    "label",
    "pred_is_match",
    "score",
    "matched_keypoints",
    "shared_area_a",
    "shared_area_b",
    "is_flipped",
    "time_ms",
    "status",          # ok | error | skipped
    "error_type",
    "error_msg",
    "a_path",
    "b_path",
    "a_mask_path",
    "b_mask_path",
    "iou_a",
    "iou_b",
    "dice_a",
    "dice_b",
]


def _read_existing_predictions(
    preds_csv: Path,
    *,
    retry_errors: bool,
) -> tuple[set[str], List[int], List[int], List[float], List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]], List[float], int]:
    """Return:
      done_pair_ids, y_true, y_pred, scores, pos_mask_stats, neg_pred_area_fracs, n_errors
    """
    done: set[str] = set()
    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    pos_mask_stats: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    neg_pred_area_fracs: List[float] = []
    n_errors = 0

    if not preds_csv.exists():
        return done, y_true, y_pred, scores, pos_mask_stats, neg_pred_area_fracs, n_errors

    with preds_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("pair_id") or "").strip()
            if not pid:
                continue
            status = (row.get("status") or "ok").strip().lower()
            if status == "error":
                n_errors += 1
                if retry_errors:
                    # Don't mark it done so we can re-run it.
                    continue
            done.add(pid)

            try:
                yt = int(float(row.get("label") or 0))
                yp = int(float(row.get("pred_is_match") or 0))
                sc = float(row.get("score") or 0.0)
            except Exception:
                continue
            y_true.append(yt)
            y_pred.append(yp)
            scores.append(sc)

            # Restore mask metrics for summary
            iou_a = row.get("iou_a", "")
            iou_b = row.get("iou_b", "")
            dice_a = row.get("dice_a", "")
            dice_b = row.get("dice_b", "")
            if iou_a != "" and iou_b != "" and dice_a != "" and dice_b != "":
                try:
                    pos_mask_stats.append((float(iou_a), float(iou_b), float(dice_a), float(dice_b)))
                except Exception:
                    pass

    return done, y_true, y_pred, scores, pos_mask_stats, neg_pred_area_fracs, n_errors


def _open_csv_writer(path: Path, append: bool) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = path.open("a" if append else "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not exists or not append:
        w.writeheader()
        f.flush()
    return w, f


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark inter-panel match/no_match pairs.")

    p.add_argument("--tasks-root", required=True, help="Path to .../tasks (contains interpanel/match and interpanel/no_match)")
    p.add_argument("--out", required=True, help="Output directory for benchmark results")
    p.add_argument("--limit", type=int, default=None, help="Limit number of pairs (after optional shuffle)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle pairs before limiting")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed")

    p.add_argument("--only", choices=["match", "no_match", "both"], default="both", help="Which subset to evaluate")
    p.add_argument("--save-examples", type=int, default=20, help="Copy up to N examples per bucket (fp/fn) to out/examples/")
    p.add_argument("--save-dir-per-pair", action="store_true", help="Pass a save_dir to backend for every pair (very heavy).")
    p.add_argument("--no-plots", action="store_true", help="Disable saving plots (default: plots are saved to out/plots/)")

    # Resume / robustness
    p.add_argument("--resume", action="store_true", help="Resume from existing predictions.csv (skip already processed pair_id rows)")
    p.add_argument("--retry-errors", action="store_true", help="With --resume, re-run rows with status=error (keep others skipped)")
    p.add_argument(
        "--on-error",
        choices=["assume_no_match", "assume_match", "skip"],
        default="assume_no_match",
        help="What to do if backend crashes on a pair",
    )

    # Backend selection
    p.add_argument("--backend", choices=["copy-move-det-keypoint", "imm"], default="copy-move-det-keypoint")

    # copy-move-det-keypoint knobs
    p.add_argument("--descriptor-type", default="cv_rsift")
    p.add_argument("--alignment-strategy", default="CV_MAGSAC")
    p.add_argument("--matching-method", default="BF")
    p.add_argument("--min-keypoints", type=int, default=20)
    p.add_argument("--min-area", type=float, default=0.01)
    p.add_argument("--no-flip", action="store_true", help="Disable flip checking")
    p.add_argument("--cross-kp-count", type=int, default=1000)
    p.add_argument("--score-key", choices=["shared_area_min", "shared_area_mean", "matched_keypoints"], default="shared_area_min")
    p.add_argument("--assume-rgb", action="store_true", help="If set, backend will treat numpy/PIL images as RGB (for path inputs it's irrelevant).")
    p.add_argument("--keep-image-for-viz", action="store_true", help="Keep image_bgr inside FeatureSet (needed only if backend writes visualizations).")
    p.add_argument("--prep-cache", choices=["none", "mem", "disk"], default="mem", help="Prepared feature caching mode")
    p.add_argument("--prep-cache-dir", default=None, help="If prep-cache=disk, directory for cached prepared features")
    p.add_argument("--prep-cache-max", type=int, default=10000, help="Max prepared features kept in memory (LRU). 0 = unlimited.")

    # IMM knobs
    p.add_argument("--imm-root", default="modules/image-matching-models", help="Path to IMM repo root that contains matching/")
    p.add_argument("--imm-matcher", default="loftr", help="IMM matcher name (e.g., loftr, eloftr, matchformer, sift-lg, superpoint-lg, ...)")
    p.add_argument("--imm-device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--imm-max-num-keypoints", type=int, default=2048, help="Forwarded to matching.get_matcher(max_num_keypoints=...)")
    p.add_argument("--imm-min-inliers", type=int, default=20, help="Match decision: min inliers after RANSAC")
    p.add_argument("--imm-min-inlier-ratio", type=float, default=0.0, help="Match decision: min inlier_ratio (0 disables)")
    p.add_argument("--imm-score-mode", choices=["num_inliers", "inlier_ratio"], default="num_inliers")
    p.add_argument("--imm-mask-mode", choices=["none", "convex_hull"], default="none", help="Optional rough masks from inlier keypoints")
    p.add_argument("--imm-mask-dilate", type=int, default=0, help="Dilate convex hull mask by k pixels (requires OpenCV)")
    p.add_argument("--imm-max-side-sum", type=int, default=0, help="If >0, downscale each image so (H+W) <= this value before IMM runs (aspect preserved).")
    p.add_argument("--imm-resize-cache-dir", default=None, help="Cache dir for resized images (default: <out>/_cache/imm_resize).")

    return p.parse_args(argv)


def _run_single(args: argparse.Namespace) -> None:

    tasks_root = Path(args.tasks_root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    ds = InterpanelDataset(tasks_root=tasks_root)

    include_match = args.only in ("match", "both")
    include_no = args.only in ("no_match", "both")

    examples = ds.list_examples(limit=args.limit, shuffle=args.shuffle, seed=args.seed,
                                include_match=include_match, include_no_match=include_no)

    if len(examples) == 0:
        raise SystemExit(f"No pairs found under: {tasks_root}/interpanel (check path and folder layout).")

    if args.backend == "copy-move-det-keypoint":
        backend = build_backend_copy_move_det_keypoint(args)
    elif args.backend == "imm":
        backend = build_backend_imm(args, out_dir)
    else:
        raise SystemExit(f"Unknown backend: {args.backend}")

    preds_csv = out_dir / "predictions.csv"
    # Keep backward compatibility, but also write result.json for clarity.
    summary_json = out_dir / "summary.json"
    result_json = out_dir / "result.json"
    examples_dir = out_dir / "examples"

    done_pair_ids: set[str] = set()
    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    pos_mask_stats: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    neg_pred_area_fracs: List[float] = []
    n_errors = 0

    if args.resume:
        done_pair_ids, y_true, y_pred, scores, pos_mask_stats, neg_pred_area_fracs, n_errors = _read_existing_predictions(
            preds_csv, retry_errors=bool(args.retry_errors)
        )

    writer, fout = _open_csv_writer(preds_csv, append=bool(args.resume))

    # Save some examples
    saved_fp = saved_fn = 0

    t0 = time.perf_counter()
    total_ms = 0.0
    pred_match_cnt = 0
    pred_nomatch_cnt = 0

    iterator: Iterable[PairExample]
    if tqdm is not None:
        iterator = tqdm(examples, desc=f"Interpanel benchmark ({args.backend})", unit="pair")
    else:
        iterator = examples

    for idx, ex in enumerate(iterator):
        if args.resume and ex.pair_id in done_pair_ids:
            continue

        pair_save_dir = None
        if args.save_dir_per_pair:
            pair_save_dir = str(out_dir / "pair_outputs" / ex.pair_id)

        st = time.perf_counter()
        status = "ok"
        error_type = ""
        error_msg = ""
        pred = None

        try:
            pred = backend.predict_pair(str(ex.a_path), str(ex.b_path), save_dir=pair_save_dir)
        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            # keep msg short for csv
            error_msg = str(e).replace("\n", " ")[:500]
            n_errors += 1

            if args.on_error == "skip":
                dt_ms = (time.perf_counter() - st) * 1000.0
                row = {
                    "pair_id": ex.pair_id,
                    "label": int(ex.label),
                    "pred_is_match": "",
                    "score": "",
                    "matched_keypoints": "",
                    "shared_area_a": "",
                    "shared_area_b": "",
                    "is_flipped": "",
                    "time_ms": float(dt_ms),
                    "status": status,
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "a_path": str(ex.a_path),
                    "b_path": str(ex.b_path),
                    "a_mask_path": str(ex.a_mask_path) if ex.a_mask_path else "",
                    "b_mask_path": str(ex.b_mask_path) if ex.b_mask_path else "",
                    "iou_a": "",
                    "iou_b": "",
                    "dice_a": "",
                    "dice_b": "",
                }
                writer.writerow(row)
                fout.flush()
                done_pair_ids.add(ex.pair_id)
                continue

            # default: treat as no_match (or match)
            assume_match = (args.on_error == "assume_match")
            from .types import MatchPrediction
            pred = MatchPrediction(
                is_match=assume_match,
                score=float("-inf") if not assume_match else float("inf"),
                matched_keypoints=0,
                shared_area_a=0.0,
                shared_area_b=0.0,
                is_flipped=False,
                mask_a=None,
                mask_b=None,
                extras={"error_type": error_type, "error_msg": error_msg},
            )

        dt_ms = (time.perf_counter() - st) * 1000.0
        total_ms += dt_ms

        yt = int(ex.label)
        yp = 1 if bool(pred.is_match) else 0
        if yp == 1:
            pred_match_cnt += 1
        else:
            pred_nomatch_cnt += 1

        y_true.append(yt)
        y_pred.append(yp)
        scores.append(float(pred.score) if pred.score not in (None,) else 0.0)

        # mask eval (only if GT exists and pred masks exist)
        iou_a = iou_b = dice_a = dice_b = ""
        if ex.a_mask_path is not None and ex.b_mask_path is not None and pred.mask_a is not None and pred.mask_b is not None:
            try:
                gt_a = load_mask(ex.a_mask_path)
                gt_b = load_mask(ex.b_mask_path)
                iou_a_v = iou(pred.mask_a, gt_a)
                iou_b_v = iou(pred.mask_b, gt_b)
                dice_a_v = dice_f1(pred.mask_a, gt_a)
                dice_b_v = dice_f1(pred.mask_b, gt_b)
                pos_mask_stats.append((iou_a_v, iou_b_v, dice_a_v, dice_b_v))
                iou_a = float(iou_a_v)
                iou_b = float(iou_b_v)
                dice_a = float(dice_a_v)
                dice_b = float(dice_b_v)
            except Exception:
                pass
        else:
            if pred.mask_a is not None and pred.mask_b is not None:
                try:
                    ma = int((pred.mask_a > 0).sum())
                    mb = int((pred.mask_b > 0).sum())
                    ha, wa = pred.mask_a.shape[:2]
                    hb, wb = pred.mask_b.shape[:2]
                    frac = 0.5 * (ma / max(1, ha * wa) + mb / max(1, hb * wb))
                    neg_pred_area_fracs.append(float(frac))
                except Exception:
                    pass

        row = {
            "pair_id": ex.pair_id,
            "label": yt,
            "pred_is_match": yp,
            "score": float(pred.score),
            "matched_keypoints": int(getattr(pred, "matched_keypoints", 0) or 0),
            "shared_area_a": float(getattr(pred, "shared_area_a", 0.0) or 0.0),
            "shared_area_b": float(getattr(pred, "shared_area_b", 0.0) or 0.0),
            "is_flipped": bool(getattr(pred, "is_flipped", False)),
            "time_ms": float(dt_ms),
            "status": status,
            "error_type": error_type,
            "error_msg": error_msg,
            "a_path": str(ex.a_path),
            "b_path": str(ex.b_path),
            "a_mask_path": str(ex.a_mask_path) if ex.a_mask_path else "",
            "b_mask_path": str(ex.b_mask_path) if ex.b_mask_path else "",
            "iou_a": iou_a,
            "iou_b": iou_b,
            "dice_a": dice_a,
            "dice_b": dice_b,
        }
        writer.writerow(row)

        # Flush regularly for Kaggle resume safety
        if (idx + 1) % 25 == 0:
            fout.flush()

        done_pair_ids.add(ex.pair_id)

        # Save some failure examples for quick inspection
        if args.save_examples > 0:
            try:
                if yt == 0 and yp == 1 and saved_fp < args.save_examples:
                    ex_dir = examples_dir / "fp" / ex.pair_id
                    copy_pair_files(ex, ex_dir, pred_mask_a=pred.mask_a, pred_mask_b=pred.mask_b)
                    saved_fp += 1
                if yt == 1 and yp == 0 and saved_fn < args.save_examples:
                    ex_dir = examples_dir / "fn" / ex.pair_id
                    copy_pair_files(ex, ex_dir, pred_mask_a=pred.mask_a, pred_mask_b=pred.mask_b)
                    saved_fn += 1
            except Exception:
                pass

        # Progress postfix
        if tqdm is not None and idx % 25 == 0:
            try:
                iterator.set_postfix({
                    "pred_match": pred_match_cnt,
                    "pred_no": pred_nomatch_cnt,
                    "avg_ms": round(total_ms / max(1, (len(y_true))), 1),
                    "errs": n_errors,
                })
            except Exception:
                pass
        elif tqdm is None and (len(y_true) % 200 == 0):
            print(f"[{len(y_true)} done] avg_ms={total_ms/max(1,len(y_true)):.1f} pred_match={pred_match_cnt} errs={n_errors}")

    try:
        fout.flush()
        fout.close()
    except Exception:
        pass

    total_s = time.perf_counter() - t0

    # Metrics (combined)
    cls_all = classification_metrics(y_true, y_pred)

    # Subset stats
    tp, fp, tn, fn = int(cls_all.tp), int(cls_all.fp), int(cls_all.tn), int(cls_all.fn)
    n_pos = int(sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    recall_pos = float(tp / max(1, (tp + fn)))
    specificity_neg = float(tn / max(1, (tn + fp)))

    # Score-based metrics (AUCs + best thresholds). Robust for heavy imbalance.
    score_sum, pr_curve, roc_curve = score_summary(y_true, scores)

    # Simple score summaries per class
    try:
        pos_scores = [s for t, s in zip(y_true, scores) if t == 1 and np.isfinite(s)]
        neg_scores = [s for t, s in zip(y_true, scores) if t == 0 and np.isfinite(s)]
        pos_score_mean = float(np.mean(pos_scores)) if len(pos_scores) else None
        neg_score_mean = float(np.mean(neg_scores)) if len(neg_scores) else None
    except Exception:
        pos_score_mean = None
        neg_score_mean = None

    mm = mask_metrics(pos_mask_stats)

    backend_info = backend.info()
    try:
        backend_info_d = asdict(backend_info)
    except Exception:
        backend_info_d = backend_info.__dict__

    # Result JSON ordering (match -> no_match -> combined)
    summary = {
        "backend": backend_info_d,
        "run": {
            "tasks_root": str(tasks_root),
            "out_dir": str(out_dir),
            "n_pairs_total_listed": int(len(examples)),
            "n_pairs_scored": int(len(y_true)),
            "n_match": int(n_pos),
            "n_no_match": int(n_neg),
            "match_rate": float(n_pos / max(1, len(y_true))),
            "n_errors": int(n_errors),
            "total_seconds": float(total_s),
            "pairs_per_second": float(len(y_true) / max(1e-9, total_s)),
        },

        # 1) Match (positive class) details first
        "match": {
            "n": int(n_pos),
            "tp": int(tp),
            "fn": int(fn),
            "recall": float(recall_pos),
            "miss_rate": float(fn / max(1, n_pos)),
            "score_mean": pos_score_mean,
        },

        # 2) No-match (negative class) metrics
        "no_match": {
            "n": int(n_neg),
            "tn": int(tn),
            "fp": int(fp),
            "specificity": float(specificity_neg),
            "false_alarm_rate": float(fp / max(1, n_neg)),
            "score_mean": neg_score_mean,
        },

        # 3) Combined / overall metrics
        "combined": {
            "classification": cls_all.to_dict(),
            "score": {
                "summary": score_sum.to_dict(),
                "precision_recall_curve": pr_curve.to_dict() if pr_curve is not None else None,
                "roc_curve": roc_curve.to_dict() if roc_curve is not None else None,
            },
        },

        "mask_metrics_on_gt_match_pairs": mm.to_dict(),
        "negatives_pred_mask_area_frac_mean": float(np.mean(neg_pred_area_fracs)) if len(neg_pred_area_fracs) else None,
    }

    # Save plots (optional)
    plots_saved = None
    if not bool(args.no_plots):
        try:
            plots_saved = save_benchmark_plots(
                out_dir,
                y_true=y_true,
                y_pred=y_pred,
                scores=scores,
                title_prefix=str(backend_info_d.get("name") or ""),
            )
        except Exception:
            plots_saved = None
    if plots_saved is not None:
        summary["plots"] = plots_saved

    text = json.dumps(summary, indent=2)
    _atomic_write_text(result_json, text)
    _atomic_write_text(summary_json, text)

    try:
        backend.close()
    except Exception:
        pass

    print("\nSaved:")
    print(f"- {preds_csv}")
    print(f"- {result_json}")
    print(f"- {summary_json} (same content, legacy name)")
    if args.save_examples > 0:
        print(f"- {examples_dir} (fp/fn samples)")



def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # IMM: allow comma-separated matcher list and run each matcher into a subfolder under --out.
    if args.backend == "imm":
        matchers = [m.strip() for m in str(getattr(args, "imm_matcher", "")).split(",") if m.strip()]
        if len(matchers) > 1:
            root_out = Path(args.out)
            for mname in matchers:
                sub_args = argparse.Namespace(**vars(args))
                sub_args.imm_matcher = mname
                sub_args.out = str(root_out / f"imm__{mname}")
                _run_single(sub_args)
            return
        if len(matchers) == 1:
            args.imm_matcher = matchers[0]

    _run_single(args)
if __name__ == "__main__":
    main()
