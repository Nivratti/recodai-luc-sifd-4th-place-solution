"""
Keypoint matching and geometric verification module.

Implements:
- G2NN (Generalized 2-Nearest Neighbor) keypoint selection
- Geometric consistency verification (MAGSAC, RANSAC, LMEDS)
- Shared content area calculation via convex hull
"""

import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class AlignmentStrategy(str, Enum):
    """Supported geometric alignment strategies."""
    CV_MAGSAC = "CV_MAGSAC"       # MAGSAC++ (recommended)
    CV_RANSAC = "CV_RANSAC"       # Classic RANSAC
    CV_LMEDS = "CV_LMEDS"         # Least Median of Squares


class MatchingMethod(str, Enum):
    """Keypoint matching methods."""
    BF = "BF"          # Brute Force
    FLANN = "FLANN"    # FLANN-based


def g2nn_keypoint_selection(
    keypoints1: np.ndarray,
    descriptions1: np.ndarray,
    keypoints2: np.ndarray,
    descriptions2: np.ndarray,
    k_rate: float = 0.5,
    nndr_threshold: float = 0.75,
    matching_method: MatchingMethod = MatchingMethod.BF,
    eps: float = 1e-7,
    ignore_self_matches: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Perform G2NN (Generalized 2-Nearest Neighbor) keypoint selection.
    
    This method selects keypoints that have consistent matches based on
    the nearest neighbor distance ratio test.
    
    Args:
        keypoints1: Keypoints from image 1 (Nx2)
        descriptions1: Descriptors from image 1 (NxD)
        keypoints2: Keypoints from image 2 (Mx2)
        descriptions2: Descriptors from image 2 (MxD)
        k_rate: Rate to define number of neighbors to match
        nndr_threshold: NNDR threshold for match selection
        matching_method: BF (Brute Force) or FLANN
        eps: Small value to avoid division by zero
        ignore_self_matches: If True, ignores matches where queryIdx == trainIdx
        
    Returns:
        Tuple of (indices1, indices2) for matched keypoints
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return [], []
    
    # Ensure correct dtype for matchers
    descriptions1 = descriptions1.astype(np.float32)
    descriptions2 = descriptions2.astype(np.float32)
    
    # Swap so smaller set is keypoints1
    swapped = False
    if len(keypoints2) < len(keypoints1):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True
    
    # Compute k for kNN
    k = max(2, int(round(len(keypoints1) * k_rate)))
    k = min(k, len(descriptions2))  # Can't have k larger than training set
    
    # Match keypoints
    if matching_method == MatchingMethod.BF:
        matcher = cv2.BFMatcher()
        knn_matches = matcher.knnMatch(descriptions1, descriptions2, k=k)
    else:  # FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = matcher.knnMatch(descriptions1, descriptions2, k=min(k, 2))
    
    # G2NN match selection
    selected_matches = []
    for matches in knn_matches:
        if ignore_self_matches:
            matches = [m for m in matches if m.queryIdx != m.trainIdx]
        
        if len(matches) < 2:
            continue
        for i in range(len(matches) - 1):
            if matches[i].distance / (matches[i + 1].distance + eps) < nndr_threshold:
                selected_matches.append(matches[i])
            else:
                break
    
    # Select unique keypoint pairs
    indices1 = []
    indices2 = []
    distances = []
    
    for match in selected_matches:
        if match.queryIdx not in indices1 and match.trainIdx not in indices2:
            indices1.append(match.queryIdx)
            indices2.append(match.trainIdx)
            distances.append(match.distance)
        else:
            # If already matched, keep the one with smaller distance
            if match.queryIdx in indices1:
                i = indices1.index(match.queryIdx)
            else:
                i = indices2.index(match.trainIdx)
            
            if distances[i] > match.distance:
                indices1[i] = match.queryIdx
                indices2[i] = match.trainIdx
                distances[i] = match.distance
    
    # Undo swap if necessary
    if swapped:
        indices1, indices2 = indices2, indices1
    
    return indices1, indices2


def verify_geometric_consistency(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC,
    displacement_thresh: float = 5.0,
    min_keypoints: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify geometric consistency of matched keypoints using homography or
    fundamental matrix estimation.
    
    Args:
        keypoints1: Matched keypoints from image 1 (Nx2)
        keypoints2: Matched keypoints from image 2 (Nx2)
        alignment_strategy: Method for geometric verification
        displacement_thresh: Maximum displacement for inliers
        min_keypoints: Minimum keypoints required
        
    Returns:
        Tuple of (consistent_kpts1, consistent_kpts2)
    """
    if len(keypoints1) < min_keypoints:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    keypoints1 = keypoints1.astype(np.float32)
    keypoints2 = keypoints2.astype(np.float32)
    
    # Use fundamental matrix for MAGSAC, homography for others
    use_homography = alignment_strategy in [
        AlignmentStrategy.CV_RANSAC,
        AlignmentStrategy.CV_LMEDS,
    ]
    
    if use_homography:
        # Estimate homography
        method_map = {
            AlignmentStrategy.CV_RANSAC: cv2.RANSAC,
            AlignmentStrategy.CV_LMEDS: cv2.LMEDS,
        }
        method = method_map.get(alignment_strategy, cv2.RANSAC)
        
        H, inliers = cv2.findHomography(
            keypoints1, keypoints2, method,
            ransacReprojThreshold=3.0,
            confidence=0.999,
            maxIters=100000
        )
        
        if H is None or inliers is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        # Verify alignment by transforming points
        pts = keypoints1.reshape(-1, 1, 2)
        aligned = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        
        # Check displacement
        displacements = np.sqrt(np.sum((aligned - keypoints2) ** 2, axis=1))
        consistent = displacements < displacement_thresh
        
    else:
        try:
            F, inliers = cv2.findFundamentalMat(
                keypoints1, keypoints2, cv2.USAC_MAGSAC,
                ransacReprojThreshold=0.5,
                confidence=0.999,
                maxIters=100000
            )
        except cv2.error:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
        if F is None or inliers is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        consistent = inliers.flatten() > 0
    
    return keypoints1[consistent], keypoints2[consistent]


def compute_shared_area(
    image_shape: Tuple[int, int],
    keypoints: np.ndarray
) -> float:
    """
    Compute the fraction of image area covered by keypoints using convex hull.
    
    Args:
        image_shape: (height, width) of the image
        keypoints: Nx2 array of keypoint coordinates
        
    Returns:
        Fraction of image area covered (0.0 to 1.0)
    """
    if len(keypoints) < 3:
        return 0.0
    
    # Compute convex hull
    hull = cv2.convexHull(keypoints.astype(np.int32))
    
    # Create mask and compute area
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    
    hull_area = np.sum(mask)
    total_area = image_shape[0] * image_shape[1]
    
    return hull_area / total_area


def match_keypoints(
    keypoints1: np.ndarray,
    descriptions1: np.ndarray,
    keypoints2: np.ndarray,
    descriptions2: np.ndarray,
    k_rate: float = 0.5,
    nndr_threshold: float = 0.75,
    matching_method: MatchingMethod = MatchingMethod.BF,
    ignore_self_matches: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Match keypoints between two images using G2NN selection.
    
    Wrapper for g2nn_keypoint_selection for backward compatibility.
    """
    return g2nn_keypoint_selection(
        keypoints1, descriptions1,
        keypoints2, descriptions2,
        k_rate=k_rate,
        nndr_threshold=nndr_threshold,
        matching_method=matching_method,
        ignore_self_matches=ignore_self_matches
    )


def match_and_verify(
    keypoints1: np.ndarray,
    descriptors1: np.ndarray,
    keypoints2: np.ndarray,
    descriptors2: np.ndarray,
    flip_keypoints1: Optional[np.ndarray] = None,
    flip_descriptors1: Optional[np.ndarray] = None,
    image1_shape: Optional[Tuple[int, int]] = None,
    image2_shape: Optional[Tuple[int, int]] = None,
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC,
    matching_method: MatchingMethod = MatchingMethod.BF,
    min_keypoints: int = 20,
    min_area: float = 0.01,
    check_flip: bool = True
) -> dict:
    """
    Match two images and determine if they share content.
    
    This function:
    1. Matches keypoints between the two images
    2. Verifies geometric consistency
    3. Computes shared content area
    4. Optionally checks flipped version of image 1
    
    Args:
        keypoints1: Keypoints from image 1
        descriptors1: Descriptors from image 1
        keypoints2: Keypoints from image 2
        descriptors2: Descriptors from image 2
        flip_keypoints1: Keypoints from flipped image 1
        flip_descriptors1: Descriptors from flipped image 1
        image1_shape: (height, width) of image 1
        image2_shape: (height, width) of image 2
        alignment_strategy: Geometric verification method
        matching_method: Keypoint matching method
        min_keypoints: Minimum matches required
        min_area: Minimum shared area threshold
        check_flip: Whether to check flipped version
        
    Returns:
        Dictionary with match results
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return {
            'matched_kpts1': np.array([]).reshape(0, 2),
            'matched_kpts2': np.array([]).reshape(0, 2),
            'shared_area_img1': 0.0,
            'shared_area_img2': 0.0,
            'matched_keypoints': 0,
            'is_flipped_match': False,
            'is_match': False
        }
    
    # Step 1: G2NN keypoint selection
    indices1, indices2 = g2nn_keypoint_selection(
        keypoints1, descriptors1,
        keypoints2, descriptors2,
        matching_method=matching_method
    )
    
    if len(indices1) < min_keypoints:
        matched_kpts1 = np.array([]).reshape(0, 2)
        matched_kpts2 = np.array([]).reshape(0, 2)
        area1, area2, num_matches = 0.0, 0.0, 0
    else:
        matched_kpts1 = keypoints1[indices1]
        matched_kpts2 = keypoints2[indices2]
        
        # Step 2: Geometric consistency verification
        matched_kpts1, matched_kpts2 = verify_geometric_consistency(
            matched_kpts1, matched_kpts2,
            alignment_strategy=alignment_strategy,
            min_keypoints=min_keypoints
        )
        
        num_matches = len(matched_kpts1)
        
        # Step 3: Compute shared area
        if num_matches >= min_keypoints and image1_shape and image2_shape:
            area1 = compute_shared_area(image1_shape, matched_kpts1)
            area2 = compute_shared_area(image2_shape, matched_kpts2)
        else:
            area1, area2 = 0.0, 0.0
    
    is_flipped = False
    
    # Try flipped matching if enabled and regular matching didn't find significant overlap
    if check_flip and flip_keypoints1 is not None and flip_descriptors1 is not None:
        if max(area1, area2) < min_area:
            flip_indices1, flip_indices2 = g2nn_keypoint_selection(
                flip_keypoints1, flip_descriptors1,
                keypoints2, descriptors2,
                matching_method=matching_method
            )
            
            if len(flip_indices1) >= min_keypoints:
                flip_matched_kpts1 = flip_keypoints1[flip_indices1]
                flip_matched_kpts2 = keypoints2[flip_indices2]
                
                flip_matched_kpts1, flip_matched_kpts2 = verify_geometric_consistency(
                    flip_matched_kpts1, flip_matched_kpts2,
                    alignment_strategy=alignment_strategy,
                    min_keypoints=min_keypoints
                )
                
                if len(flip_matched_kpts1) >= min_keypoints and image1_shape and image2_shape:
                    flip_area1 = compute_shared_area(image1_shape, flip_matched_kpts1)
                    flip_area2 = compute_shared_area(image2_shape, flip_matched_kpts2)
                    
                    # Use flipped result if better
                    if max(flip_area1, flip_area2) > max(area1, area2):
                        matched_kpts1 = flip_matched_kpts1
                        matched_kpts2 = flip_matched_kpts2
                        area1, area2 = flip_area1, flip_area2
                        num_matches = len(flip_matched_kpts1)
                        is_flipped = True
    
    return {
        'matched_kpts1': matched_kpts1,
        'matched_kpts2': matched_kpts2,
        'shared_area_img1': area1,
        'shared_area_img2': area2,
        'matched_keypoints': num_matches,
        'is_flipped_match': is_flipped,
        'is_match': min(area1, area2) >= min_area if (area1 > 0 and area2 > 0) else num_matches >= min_keypoints
    }


# Backward compatibility
def verify_matches_geometric(keypoints1, keypoints2, matches_indices):
    """
    Legacy function - filters matches using geometric consistency.
    
    Args:
        keypoints1: numpy array of shape (N, 2)
        keypoints2: numpy array of shape (M, 2)
        matches_indices: List of tuples (idx1, idx2)
        
    Returns:
        List of tuples (idx1, idx2) that are geometrically consistent.
    """
    if len(matches_indices) < 4:
        return matches_indices
    
    # Extract matched keypoints
    kpts1 = np.array([keypoints1[i] for i, _ in matches_indices])
    kpts2 = np.array([keypoints2[j] for _, j in matches_indices])
    
    # Verify geometric consistency
    consistent_kpts1, consistent_kpts2 = verify_geometric_consistency(
        kpts1, kpts2,
        alignment_strategy=AlignmentStrategy.CV_MAGSAC
    )
    
    if len(consistent_kpts1) == 0:
        return []
    
    # Find which original matches are consistent
    result = []
    for i, (idx1, idx2) in enumerate(matches_indices):
        pt1 = keypoints1[idx1]
        for ck in consistent_kpts1:
            if np.allclose(pt1, ck, atol=0.5):
                result.append((idx1, idx2))
                break
    
    return result
