from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ClassifierResult:
    """Single-image prediction result."""

    group_label: str
    group_score: float

    subtype_label: str
    subtype_score: float

    # Optional probability dicts (only if requested)
    group_probs: Optional[Dict[str, float]] = None
    subtype_probs: Optional[Dict[str, float]] = None

    # Optional embedding (unit-norm, float32, shape [D])
    embedding: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "group_label": self.group_label,
            "group_score": float(self.group_score),
            "subtype_label": self.subtype_label,
            "subtype_score": float(self.subtype_score),
        }
        if self.group_probs is not None:
            d["group_probs"] = {k: float(v) for k, v in self.group_probs.items()}
        if self.subtype_probs is not None:
            d["subtype_probs"] = {k: float(v) for k, v in self.subtype_probs.items()}
        if self.embedding is not None:
            # Leave as-is (could be numpy array). Caller can serialize as needed.
            d["embedding"] = self.embedding
        return d
