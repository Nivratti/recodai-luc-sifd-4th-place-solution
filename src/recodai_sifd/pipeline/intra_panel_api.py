from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Any, Optional

import numpy as np


class IntraPanelCopyMoveModel(Protocol):
    """
    Later: your real intra-panel copy-move model.
    Input: a single panel image (RGB, HxWx3) as numpy.
    Output: list of instance masks in *panel crop coordinates* (HxW, bool/uint8).
    """

    def predict_instances(self, panel_rgb: np.ndarray, *, panel_uid: Optional[str] = None) -> List[np.ndarray]:
        ...


@dataclass
class NoopIntraPanelModel:
    """Placeholder: returns no instances."""
    def predict_instances(self, panel_rgb: np.ndarray, *, panel_uid: Optional[str] = None) -> List[np.ndarray]:
        return []
