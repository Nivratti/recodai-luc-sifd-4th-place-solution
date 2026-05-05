"""Public types used by panel_cbir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image as PILImage


# Accept image data in multiple common formats.
# For numpy arrays, RGB is expected (H,W,3) unless config/grayscale converts.
ImageInput = Union[str, Path, np.ndarray, PILImage.Image]


@dataclass(frozen=True)
class Match:
    """A single ranked match result.

    Notes
    -----
    The `path` field is an identifier for the matched item. When you pass image paths,
    it will be the filesystem path. When you pass in-memory images (numpy/PIL), it will
    be an auto-generated id like `mem:12` unless you provided explicit ids.
    """

    path: str
    score: float
    rank: int


@dataclass
class CBIRIndex:
    """A reusable index of candidate embeddings.

    Attributes
    ----------
    paths:
        Candidate image paths (aligned with rows in `embeddings`).
    embeddings:
        L2-normalized embeddings of shape (N, D), dtype float32.
    meta:
        Optional metadata, e.g., config fingerprint.
    """

    paths: List[str]  # item ids (often filesystem paths)
    embeddings: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy.ndarray")
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (N,D); got shape={self.embeddings.shape}")
        if len(self.paths) != int(self.embeddings.shape[0]):
            raise ValueError(
                f"paths/embeddings length mismatch: len(paths)={len(self.paths)} vs embeddings.shape[0]={self.embeddings.shape[0]}"
            )
        if self.embeddings.dtype != np.float32:
            self.embeddings = self.embeddings.astype(np.float32, copy=False)
