from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .types import CBIRIndex


def save_index_npz(index: CBIRIndex, path: str | Path) -> Path:
    """Save a :class:`~panel_cbir.types.CBIRIndex` to a single .npz file."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    meta_json = json.dumps(index.meta, ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        p,
        embeddings=index.embeddings.astype(np.float32, copy=False),
        paths=np.asarray(index.paths, dtype=np.unicode_),
        meta_json=np.asarray(meta_json, dtype=np.unicode_),
    )
    return p


def load_index_npz(path: str | Path) -> CBIRIndex:
    """Load an index saved by :func:`save_index_npz`."""

    p = Path(path)
    data = np.load(p, allow_pickle=False)

    embeddings = data["embeddings"].astype(np.float32, copy=False)
    paths = [str(x) for x in data["paths"].tolist()]

    meta_json = str(data["meta_json"].tolist()) if "meta_json" in data.files else "{}"
    try:
        meta: Dict[str, Any] = json.loads(meta_json)
    except Exception:
        meta = {}

    return CBIRIndex(paths=paths, embeddings=embeddings, meta=meta)
