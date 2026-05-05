"""panel-cbir: content-based image retrieval for scientific panels.

Public API
----------
- :class:`panel_cbir.CBIRConfig`
- :class:`panel_cbir.PanelCBIR`
- :class:`panel_cbir.CBIRIndex`
- :class:`panel_cbir.Match`
"""

from .config import CBIRConfig
from .engine import PanelCBIR
from .types import CBIRIndex, Match, ImageInput

__all__ = ["CBIRConfig", "PanelCBIR", "CBIRIndex", "Match", "ImageInput"]
