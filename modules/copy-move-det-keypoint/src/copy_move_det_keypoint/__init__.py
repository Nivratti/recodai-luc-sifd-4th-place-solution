"""copy_move_det_keypoint

High-level API:
- detect(...)
- CopyMoveDetector
- DetectorConfig, DetectionResult

Batch / optimization API:
- FeatureSet
- prepare(...)
- match_prepared(...)
- optional pruning: match_keypoints_only(...) + build_masks_from_matches(...)
- MatchInfo

Enums:
- DescriptorType
- AlignmentStrategy
- MatchingMethod
"""

from .api import (
    detect,
    CopyMoveDetector,
    DetectorConfig,
    DetectionResult,
    FeatureSet,
    MatchInfo,
    prepare,
    match_prepared,
    match_keypoints_only,
    build_masks_from_matches,
)
from .feature_extraction import DescriptorType
from .matching import AlignmentStrategy, MatchingMethod

__all__ = [
    "detect",
    "CopyMoveDetector",
    "DetectorConfig",
    "DetectionResult",
    "FeatureSet",
    "MatchInfo",
    "prepare",
    "match_prepared",
    "match_keypoints_only",
    "build_masks_from_matches",
    "DescriptorType",
    "AlignmentStrategy",
    "MatchingMethod",
]
