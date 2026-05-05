from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, Sequence, Tuple


class HasXYXY(Protocol):
    """Protocol for detection objects coming from your panel detector."""
    xyxy: Sequence[float]


class FigureKind(str, Enum):
    SIMPLE = "simple_figure"
    COMPOUND_SINGLE = "compound_single_panel"
    COMPOUND_MULTI = "compound_multi_panel"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FigureKindDecision:
    kind: FigureKind
    n_panels: int
    edge_margin_px: int
    main_xyxy: Optional[Tuple[float, float, float, float]]
    touches_edge: Optional[bool]
    reason: str


def _as_xyxy(det: Any) -> Tuple[float, float, float, float]:
    xyxy = getattr(det, "xyxy", det)  # allow passing raw (x1,y1,x2,y2) tuples too
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        raise ValueError(f"Invalid xyxy on detection: {xyxy}")
    x1, y1, x2, y2 = xyxy
    return float(x1), float(y1), float(x2), float(y2)


def _clip_xyxy(xyxy: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    # ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _area(xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _touches_edge(
    xyxy: Tuple[float, float, float, float],
    w: int,
    h: int,
    margin: int,
) -> bool:
    x1, y1, x2, y2 = xyxy
    # "near edge" if any side is within margin pixels of the border
    return (x1 <= margin) or (y1 <= margin) or (x2 >= (w - margin)) or (y2 >= (h - margin))


def _compute_margin_px(
    w: int,
    h: int,
    edge_margin_px: Optional[int],
    edge_margin_ratio: float,
    min_edge_margin_px: int,
    max_edge_margin_px: int,
) -> int:
    if edge_margin_px is not None:
        return max(0, int(edge_margin_px))
    base = int(round(min(w, h) * float(edge_margin_ratio)))
    base = max(min_edge_margin_px, base)
    base = min(max_edge_margin_px, base)
    return base


def classify_figure_kind(
    image_size_wh: Tuple[int, int],
    detections: Iterable[HasXYXY],
    *,
    # edge logic
    edge_margin_px: Optional[int] = None,
    edge_margin_ratio: float = 0.02,   # 2% of min(image side)
    min_edge_margin_px: int = 8,
    max_edge_margin_px: int = 64,
    # noise control (optional but helps avoid tiny FPs deciding the layout)
    min_panel_area_ratio: float = 0.0,  # set 0.005–0.02 if you see tiny false positives
    # which "panel" to use when n==1 (or if you later decide to handle n>1 differently)
    main_panel_strategy: str = "largest",  # "largest" or "first"
) -> FigureKindDecision:
    """
    Rules (your requested logic):
      1) If panel detector predicts multiple panels -> compound multi panel
      2) If exactly one panel predicted:
           - if panel bbox is near image edge -> simple figure
           - else -> compound single panel
    """
    w, h = int(image_size_wh[0]), int(image_size_wh[1])
    if w <= 0 or h <= 0:
        return FigureKindDecision(
            kind=FigureKind.UNKNOWN,
            n_panels=0,
            edge_margin_px=0,
            main_xyxy=None,
            touches_edge=None,
            reason=f"Invalid image_size_wh={image_size_wh}",
        )

    margin = _compute_margin_px(
        w=w,
        h=h,
        edge_margin_px=edge_margin_px,
        edge_margin_ratio=edge_margin_ratio,
        min_edge_margin_px=min_edge_margin_px,
        max_edge_margin_px=max_edge_margin_px,
    )

    img_area = float(w * h)

    # normalize + optionally filter by area ratio
    norm: list[Tuple[float, float, float, float]] = []
    for det in detections:
        xyxy = _clip_xyxy(_as_xyxy(det), w=w, h=h)
        if min_panel_area_ratio > 0.0:
            if img_area <= 0:
                continue
            if (_area(xyxy) / img_area) < float(min_panel_area_ratio):
                continue
        norm.append(xyxy)

    n = len(norm)

    if n >= 2:
        return FigureKindDecision(
            kind=FigureKind.COMPOUND_MULTI,
            n_panels=n,
            edge_margin_px=margin,
            main_xyxy=None,
            touches_edge=None,
            reason=f"n_panels={n} (>=2) => compound multi-panel",
        )

    if n == 1:
        if main_panel_strategy == "first":
            main_xyxy = norm[0]
        else:
            main_xyxy = max(norm, key=_area)  # "largest" (default)

        touches = _touches_edge(main_xyxy, w=w, h=h, margin=margin)
        if touches:
            return FigureKindDecision(
                kind=FigureKind.SIMPLE,
                n_panels=1,
                edge_margin_px=margin,
                main_xyxy=main_xyxy,
                touches_edge=True,
                reason="n_panels=1 and bbox near edge => simple figure",
            )
        return FigureKindDecision(
            kind=FigureKind.COMPOUND_SINGLE,
            n_panels=1,
            edge_margin_px=margin,
            main_xyxy=main_xyxy,
            touches_edge=False,
            reason="n_panels=1 and bbox NOT near edge => compound single-panel",
        )

    # n == 0
    return FigureKindDecision(
        kind=FigureKind.UNKNOWN,
        n_panels=0,
        edge_margin_px=margin,
        main_xyxy=None,
        touches_edge=None,
        reason="No panels detected",
    )
