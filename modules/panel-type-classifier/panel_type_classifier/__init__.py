"""panel_type_classifier

Zero-shot panel-type classification (imaging vs non-imaging + imaging subtype) using BiomedCLIP.

Main entrypoint:
  - PanelTypeClassifier
"""

from .types import ClassifierResult
from .classifier import PanelTypeClassifier
from .prompts import DEFAULT_PROMPT_PACK, PromptPack

__all__ = ["PanelTypeClassifier", "ClassifierResult", "PromptPack", "DEFAULT_PROMPT_PACK"]
