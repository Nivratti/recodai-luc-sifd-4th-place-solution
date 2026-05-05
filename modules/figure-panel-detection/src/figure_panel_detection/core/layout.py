from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

from .types import LayoutCfg


def decide_pads(total_files: int, layout_batch_size: int) -> Tuple[int, int]:
    pad_file = 1 if total_files <= 1 else len(str(total_files - 1))
    total_batches = max(1, math.ceil(total_files / max(1, layout_batch_size)))
    pad_batch = 1 if total_batches <= 1 else len(str(total_batches - 1))
    return pad_file, pad_batch


def make_out_rel(rel_in: Path, suffix: str, layout_cfg: LayoutCfg, idx_in_bucket: int) -> Path:
    """Return output relative path (under a bucket) for a given input-relative path."""
    if layout_cfg.layout == "preserve":
        return rel_in.with_suffix(suffix)

    fname = f"{idx_in_bucket:0{layout_cfg.pad_file}d}{suffix}"

    if layout_cfg.layout == "flat":
        return Path(fname)

    if layout_cfg.layout == "batch":
        b = idx_in_bucket // max(1, layout_cfg.layout_batch_size)
        bname = f"batch_{b:0{layout_cfg.pad_batch}d}"
        return Path(bname) / Path(fname)

    raise ValueError(f"Unknown layout: {layout_cfg.layout}")


def bucket_order(using_keep: bool) -> List[str]:
    return ["kept", "ignored", "no_objects"] if using_keep else ["has_objects", "no_objects"]
