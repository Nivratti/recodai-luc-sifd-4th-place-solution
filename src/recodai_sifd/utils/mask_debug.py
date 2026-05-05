from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import numpy as np


@dataclass(frozen=True)
class MaskStats:
    key: Any
    shape: Tuple[int, int]
    dtype: str
    unique_values: np.ndarray          # sorted
    unique_counts: np.ndarray          # aligned with unique_values
    nonzero_pixels: int
    total_pixels: int
    nonzero_fraction: float
    min_value: int
    max_value: int

    def is_binary(self, allowed: Tuple[Tuple[int, ...], ...] = ((0, 255), (0, 1), (0,), (255,), (1,))) -> bool:
        u = tuple(int(x) for x in self.unique_values.tolist())
        return u in allowed


def mask_unique_counts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (unique_values, counts).
    Uses np.unique which is fine for masks; avoids any heavy tricks.
    """
    vals, cnts = np.unique(mask, return_counts=True)
    # vals already sorted
    return vals, cnts


def describe_mask(mask: np.ndarray, key: Any = None) -> MaskStats:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    vals, cnts = mask_unique_counts(mask)
    nonzero = int(np.count_nonzero(mask))
    total = int(mask.size)

    # np.min/max return numpy scalars; cast to int for cleanliness
    mn = int(vals[0]) if vals.size else 0
    mx = int(vals[-1]) if vals.size else 0

    return MaskStats(
        key=key,
        shape=(int(mask.shape[0]), int(mask.shape[1])),
        dtype=str(mask.dtype),
        unique_values=vals,
        unique_counts=cnts,
        nonzero_pixels=nonzero,
        total_pixels=total,
        nonzero_fraction=(nonzero / total) if total else 0.0,
        min_value=mn,
        max_value=mx,
    )

def assert_masks_are_01(instance_masks_by_id: Dict[int, np.ndarray]) -> None:
    bad = []
    for gid, m in instance_masks_by_id.items():
        u = np.unique(m)
        if not np.all((u == 0) | (u == 1)):
            bad.append((gid, u.tolist(), str(m.dtype), m.shape))
    if bad:
        raise ValueError(f"Non-(0/1) masks found in instance masks: {bad}")

def summarize_instance_masks(
    instance_masks_by_id: Dict[int, np.ndarray],
    *,
    max_print: int = 50,
    show_counts: bool = True,
) -> Dict[int, MaskStats]:
    """
    Returns stats dict and prints a compact summary.
    """
    stats_by_id: Dict[int, MaskStats] = {}

    ids = sorted(instance_masks_by_id.keys())
    if not ids:
        print("[mask] No instance masks.")
        return stats_by_id

    print(f"[mask] Instances: {len(ids)}")
    for i, gid in enumerate(ids):
        st = describe_mask(instance_masks_by_id[gid], key=gid)
        stats_by_id[gid] = st

        if i < max_print:
            u = st.unique_values.tolist()
            msg = f"  id={gid:03d} shape={st.shape} dtype={st.dtype} uniq={u} nnz={st.nonzero_pixels} ({st.nonzero_fraction:.4%})"
            if show_counts:
                msg += f" counts={st.unique_counts.tolist()}"
            if not st.is_binary():
                msg += "  <-- NOT BINARY?"
            print(msg)

    if len(ids) > max_print:
        print(f"  ... (printed first {max_print})")

    # asser mask values are 0 and 1
    assert_masks_are_01(instance_masks_by_id)

    return stats_by_id

def print_mask_uniques(instance_masks_by_id: Dict[int, np.ndarray], max_print: int = 50) -> None:
    for i, (gid, m) in enumerate(sorted(instance_masks_by_id.items(), key=lambda x: x[0])):
        if i >= max_print:
            print(f"... (printed first {max_print})")
            break
        u, c = np.unique(m, return_counts=True)
        print(f"id={gid:03d} shape={m.shape} dtype={m.dtype} uniq={u.tolist()} counts={c.tolist()}")
