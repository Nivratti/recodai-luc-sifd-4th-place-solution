from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..types import MatchPrediction


@dataclass
class BackendInfo:
    name: str
    version: Optional[str] = None
    notes: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None


class MatcherBackend(ABC):
    """A pluggable backend for inter-panel matching.

    Contract:
      - predict_pair() must be deterministic for a given backend config.
      - score should increase with confidence (used for thresholding / ranking later).
    """

    @abstractmethod
    def info(self) -> BackendInfo:
        raise NotImplementedError

    @abstractmethod
    def predict_pair(self, a_path: str, b_path: str, *, save_dir: Optional[str] = None) -> MatchPrediction:
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup."""
        return
