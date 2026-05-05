from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _to_bool_mask(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if m is None:
        return None
    if m.dtype != np.bool_:
        # accept 0/255, 0/1, float, etc.
        m = m > 0
    return m


def iou(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Optional[float]:
    pred = _to_bool_mask(pred)
    gt = _to_bool_mask(gt)
    if pred is None or gt is None:
        return None
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def dice_f1(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Optional[float]:
    pred = _to_bool_mask(pred)
    gt = _to_bool_mask(gt)
    if pred is None or gt is None:
        return None
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2 * inter) / float(denom)


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


@dataclass
class ClsMetrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    balanced_accuracy: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tnr: float  # specificity
    fpr: float
    fnr: float
    predicted_match_rate: float
    prevalence: float
    npv: float
    f1_neg: float
    macro_f1: float
    mcc: float
    kappa: float
    gmean: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> ClsMetrics:
    y_true = list(y_true)
    y_pred = list(y_pred)
    c = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = len(y_true)
    acc = safe_div(tp + tn, n)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    tnr = safe_div(tn, tn + fp)
    fpr = 1.0 - tnr
    fnr = safe_div(fn, fn + tp)
    pmr = safe_div(sum(y_pred), n)

    prevalence = safe_div(sum(y_true), n)
    npv = safe_div(tn, tn + fn)
    # Negative-class F1, treating class 0 as the "positive" label.
    f1_neg = safe_div(2 * tn, (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    macro_f1 = float((f1 + f1_neg) / 2.0)
    balanced_acc = float((rec + tnr) / 2.0)

    # Matthews correlation coefficient (robust for imbalance)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = float((tp * tn - fp * fn) / np.sqrt(denom)) if denom else 0.0

    # Cohen's kappa
    po = acc
    pe = (safe_div((tp + fp) * (tp + fn), n * n) + safe_div((tn + fn) * (tn + fp), n * n)) if n else 0.0
    kappa = float((po - pe) / (1.0 - pe)) if (1.0 - pe) else 0.0

    gmean = float(np.sqrt(rec * tnr))
    return ClsMetrics(
        n=n, tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        tnr=tnr, fpr=fpr, fnr=fnr,
        predicted_match_rate=pmr,
        prevalence=prevalence,
        npv=npv,
        f1_neg=f1_neg,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_acc,
        mcc=mcc,
        kappa=kappa,
        gmean=gmean,
    )


def _as_score_arrays(y_true: Iterable[int], scores: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(list(y_true), dtype=np.int32)
    sc = np.asarray(list(scores), dtype=np.float64)
    if yt.shape[0] != sc.shape[0]:
        raise ValueError(f"y_true and scores length mismatch: {yt.shape[0]} vs {sc.shape[0]}")
    # Keep only finite scores
    m = np.isfinite(sc)
    yt = yt[m]
    sc = sc[m]
    # Ensure binary 0/1
    yt = (yt > 0).astype(np.int32)
    return yt, sc


@dataclass
class Curve:
    x: List[float]
    y: List[float]
    thresholds: List[float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {"x": self.x, "y": self.y, "thresholds": self.thresholds}


def precision_recall_curve(y_true: Iterable[int], scores: Iterable[float]) -> Optional[Curve]:
    """Compute a precision-recall curve (no sklearn dependency).

    Returns None if only one class is present.
    """
    yt, sc = _as_score_arrays(y_true, scores)
    if yt.size == 0:
        return None
    P = int(yt.sum())
    if P == 0 or P == int(yt.size):
        return None

    order = np.argsort(-sc, kind="mergesort")
    yt = yt[order]
    sc = sc[order]

    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(1, (tp + fp))
    recall = tp / float(P)

    # Keep points at score changes only
    distinct = np.r_[True, sc[1:] != sc[:-1]]
    precision = precision[distinct]
    recall = recall[distinct]
    thr = sc[distinct]

    # Add (recall=0, precision=1) anchor
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thr = np.r_[np.inf, thr]

    return Curve(x=recall.astype(float).tolist(), y=precision.astype(float).tolist(), thresholds=thr.astype(float).tolist())


def roc_curve(y_true: Iterable[int], scores: Iterable[float]) -> Optional[Curve]:
    """Compute an ROC curve (no sklearn dependency). Returns None if only one class."""
    yt, sc = _as_score_arrays(y_true, scores)
    if yt.size == 0:
        return None
    P = int(yt.sum())
    N = int(yt.size - P)
    if P == 0 or N == 0:
        return None

    order = np.argsort(-sc, kind="mergesort")
    yt = yt[order]
    sc = sc[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    tpr = tp / float(P)
    fpr = fp / float(N)

    distinct = np.r_[True, sc[1:] != sc[:-1]]
    tpr = tpr[distinct]
    fpr = fpr[distinct]
    thr = sc[distinct]

    # Add (0,0) anchor
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thr = np.r_[np.inf, thr]
    return Curve(x=fpr.astype(float).tolist(), y=tpr.astype(float).tolist(), thresholds=thr.astype(float).tolist())


def auc_trapezoid(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    # ensure sorted by x
    order = np.argsort(xa)
    xa = xa[order]
    ya = ya[order]
    return float(np.trapz(ya, xa))


def average_precision(y_true: Iterable[int], scores: Iterable[float]) -> Optional[float]:
    """Average precision (AP), robust metric for heavy imbalance."""
    pr = precision_recall_curve(y_true, scores)
    if pr is None:
        return None
    # Use a precision envelope for standard AP behavior
    recall = np.asarray(pr.x, dtype=np.float64)
    precision = np.asarray(pr.y, dtype=np.float64)
    # Make precision non-increasing when recall increases
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    dr = np.diff(recall)
    if dr.size == 0:
        return 0.0
    return float(np.sum(dr * precision[1:]))


@dataclass
class ScoreSummary:
    n_scored: int
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    best_f1_threshold: Optional[float]
    best_f1: Optional[float]
    best_balanced_acc_threshold: Optional[float]
    best_balanced_accuracy: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


def score_summary(y_true: Iterable[int], scores: Iterable[float]) -> Tuple[ScoreSummary, Optional[Curve], Optional[Curve]]:
    """Compute score-based summary + PR/ROC curves.

    - pr_auc: Average precision (AP)
    - roc_auc: AUC of ROC
    - best thresholds for F1 and balanced accuracy
    """
    yt, sc = _as_score_arrays(y_true, scores)
    n_scored = int(yt.size)
    if n_scored == 0:
        s = ScoreSummary(
            n_scored=0, pr_auc=None, roc_auc=None,
            best_f1_threshold=None, best_f1=None,
            best_balanced_acc_threshold=None, best_balanced_accuracy=None,
        )
        return s, None, None

    pr = precision_recall_curve(yt, sc)
    roc = roc_curve(yt, sc)
    pr_auc = average_precision(yt, sc)
    roc_auc = auc_trapezoid(roc.x, roc.y) if roc is not None else None

    # Threshold sweep on sorted scores to find best F1 / best balanced accuracy.
    P = int(yt.sum())
    N = int(n_scored - P)
    if P == 0 or N == 0:
        s = ScoreSummary(
            n_scored=n_scored, pr_auc=pr_auc, roc_auc=roc_auc,
            best_f1_threshold=None, best_f1=None,
            best_balanced_acc_threshold=None, best_balanced_accuracy=None,
        )
        return s, pr, roc

    order = np.argsort(-sc, kind="mergesort")
    yt2 = yt[order]
    sc2 = sc[order]
    tp = np.cumsum(yt2)
    fp = np.cumsum(1 - yt2)
    fn = P - tp
    tn = N - fp

    prec = tp / np.maximum(1, (tp + fp))
    rec = tp / np.maximum(1, P)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    bal_acc = 0.5 * (rec + (tn / np.maximum(1, (tn + fp))))

    # Only consider thresholds at distinct scores
    distinct = np.r_[True, sc2[1:] != sc2[:-1]]
    f1_d = f1[distinct]
    bal_d = bal_acc[distinct]
    thr_d = sc2[distinct]

    i_f1 = int(np.argmax(f1_d)) if f1_d.size else 0
    i_bal = int(np.argmax(bal_d)) if bal_d.size else 0

    s = ScoreSummary(
        n_scored=n_scored,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        best_f1_threshold=float(thr_d[i_f1]) if thr_d.size else None,
        best_f1=float(f1_d[i_f1]) if f1_d.size else None,
        best_balanced_acc_threshold=float(thr_d[i_bal]) if thr_d.size else None,
        best_balanced_accuracy=float(bal_d[i_bal]) if bal_d.size else None,
    )
    return s, pr, roc


@dataclass
class MaskMetrics:
    n_with_gt: int
    iou_a_mean: float
    iou_b_mean: float
    iou_mean: float
    dice_a_mean: float
    dice_b_mean: float
    dice_mean: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def mask_metrics(per_pair: Iterable[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]) -> MaskMetrics:
    # per_pair: (iou_a, iou_b, dice_a, dice_b) - None if missing gt/pred
    iou_a = []
    iou_b = []
    dice_a = []
    dice_b = []
    for ia, ib, da, db in per_pair:
        if ia is None or ib is None or da is None or db is None:
            continue
        iou_a.append(ia); iou_b.append(ib); dice_a.append(da); dice_b.append(db)

    n = len(iou_a)
    if n == 0:
        return MaskMetrics(
            n_with_gt=0,
            iou_a_mean=0.0, iou_b_mean=0.0, iou_mean=0.0,
            dice_a_mean=0.0, dice_b_mean=0.0, dice_mean=0.0,
        )

    iou_a_m = float(np.mean(iou_a))
    iou_b_m = float(np.mean(iou_b))
    dice_a_m = float(np.mean(dice_a))
    dice_b_m = float(np.mean(dice_b))
    return MaskMetrics(
        n_with_gt=n,
        iou_a_mean=iou_a_m,
        iou_b_mean=iou_b_m,
        iou_mean=float((iou_a_m + iou_b_m) / 2.0),
        dice_a_mean=dice_a_m,
        dice_b_mean=dice_b_m,
        dice_mean=float((dice_a_m + dice_b_m) / 2.0),
    )
