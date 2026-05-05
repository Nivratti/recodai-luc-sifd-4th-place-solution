"""
Keypoint-based Copy-Move Detection Module.

Supports:
- Single-image copy-move detection (within same image)
- Cross-image copy detection (between two images)
- Multiple descriptor types (cv_sift, cv_rsift, vlfeat_sift_heq)
- Multiple alignment strategies (CV_MAGSAC, CV_RANSAC, CV_LMEDS)

Output format compatible with copy-move-detection module:
- mask.png: Binary mask of detected regions
- matches.png: Visualization of matched keypoints
- clusters.png: Colored cluster visualization
"""

import os
import cv2
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

# Local imports: support both package import (recommended) and script-style execution.
try:
    from .feature_extraction import (
        extract_features,
        DescriptorType
    )
    from .matching import (
        match_keypoints,
        match_and_verify,
        verify_geometric_consistency,
        AlignmentStrategy,
        MatchingMethod
    )
    from .clustering import cluster_keypoints
    from .visualization import (
        draw_matches,
        draw_matches_with_hulls,
        draw_matches_on_single_image,
        draw_clusters,
        draw_clusters_with_hulls,
        create_mask_from_keypoints,
        save_mask
    )
except ImportError:  # pragma: no cover
    from feature_extraction import (
        extract_features,
        DescriptorType
    )
    from matching import (
        match_keypoints,
        match_and_verify,
        verify_geometric_consistency,
        AlignmentStrategy,
        MatchingMethod
    )
    from clustering import cluster_keypoints
    from visualization import (
        draw_matches,
        draw_matches_with_hulls,
        draw_matches_on_single_image,
        draw_clusters,
        draw_clusters_with_hulls,
        create_mask_from_keypoints,
        save_mask
    )
class KeypointCopyMoveDetector:
    """
    Keypoint-based copy-move detection for single or cross-image analysis.
    
    Uses algorithms from provenance-analysis:
    - G2NN keypoint selection
    - MAGSAC/RANSAC geometric verification
    - Convex hull shared area computation
    """
    
    def __init__(
        self,
        output_dir: str,
        descriptor_type: DescriptorType = DescriptorType.CV_RSIFT,
        alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC,
        matching_method: MatchingMethod = MatchingMethod.BF,
        check_flip: bool = True,
        min_keypoints: int = 20,
        min_area: float = 0.01,
        timeout: int = 600
    ):
        """
        Initialize the detector.
        
        Args:
            output_dir: Directory for output files
            descriptor_type: Feature descriptor type
            alignment_strategy: Geometric verification method
            matching_method: Keypoint matching method (BF or FLANN)
            check_flip: Whether to check horizontally flipped images
            min_keypoints: Minimum matched keypoints for valid detection
            min_area: Minimum shared area threshold
            timeout: Detection timeout in seconds
        """
        self.output_dir = output_dir
        self.descriptor_type = descriptor_type
        self.alignment_strategy = alignment_strategy
        self.matching_method = matching_method
        self.check_flip = check_flip
        self.min_keypoints = min_keypoints
        self.min_area = min_area
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def detect_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run copy-move detection on a single image.
        
        Detects regions that have been copied and pasted within the same image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with detection results and output file paths
        """
        self.logger.info(f"Processing Single Image: {image_path}")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return {'success': False, 'error': 'Failed to load image'}
        
        img_shape = img.shape[:2]  # (height, width)
        
        # 2. Extract Features (no flip for single-image since we match to self)
        # Use more keypoints for single-image to get better self-matching
        self.logger.info(f"Extracting features using {self.descriptor_type.value}...")
        kps, descs, _, _ = extract_features(
            image_path, 
            descriptor_type=self.descriptor_type,
            extract_flip=False,
            kp_count=5000  # Higher count for single-image detection
        )
        self.logger.info(f"Extracted {len(kps)} keypoints.")
        
        if len(kps) < self.min_keypoints:
            self.logger.warning("Not enough keypoints found.")
            return {'success': False, 'error': 'Not enough keypoints'}
        
        # 3. Match Keypoints (Self-Matching, ignore i==j)
        self.logger.info("Matching keypoints (self-matching)...")
        indices1, indices2 = match_keypoints(
            kps, descs, kps, descs,
            matching_method=self.matching_method,
            ignore_self_matches=True
        )
        
        # Filter out self-matches (redundant but safe)
        initial_matches = [(i, j) for i, j in zip(indices1, indices2) if i != j]
        self.logger.info(f"Found {len(initial_matches)} initial matches.")
        
        if len(initial_matches) < self.min_keypoints:
            self.logger.warning("Not enough matches for geometric verification.")
            # Still create empty output files
            mask = np.zeros(img_shape, dtype=np.uint8)
            mask_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
            save_mask(mask, mask_path)
            
            matches_path = os.path.join(self.output_dir, f"{base_name}_matches.png")
            cv2.imwrite(matches_path, img)
            
            clusters_path = os.path.join(self.output_dir, f"{base_name}_clusters.png")
            cv2.imwrite(clusters_path, img)
            
            return {
                'success': True,
                'found_forgery': False,
                'mask_path': mask_path,
                'matches_path': matches_path,
                'clusters_path': clusters_path
            }
        
        # 4. Geometric Verification
        self.logger.info(f"Verifying matches geometrically using {self.alignment_strategy.value}...")
        matched_kpts1 = kps[np.array([m[0] for m in initial_matches])]
        matched_kpts2 = kps[np.array([m[1] for m in initial_matches])]
        
        consistent_kpts1, consistent_kpts2 = verify_geometric_consistency(
            matched_kpts1, matched_kpts2,
            alignment_strategy=self.alignment_strategy,
            min_keypoints=4  # Lower threshold for single-image
        )
        self.logger.info(f"Found {len(consistent_kpts1)} geometrically consistent matches.")
        
        # Build valid matches list from consistent keypoints
        valid_matches = []
        for i, (kp1_orig, kp2_orig) in enumerate(zip(matched_kpts1, matched_kpts2)):
            for ck1 in consistent_kpts1:
                if np.allclose(kp1_orig, ck1, atol=0.5):
                    valid_matches.append(initial_matches[i])
                    break
        
        # 5. Cluster Keypoints
        self.logger.info("Clustering keypoints...")
        matched_indices = list(set([m[0] for m in valid_matches] + [m[1] for m in valid_matches]))
        
        if len(matched_indices) > 0:
            subset_kps = kps[matched_indices]
            subset_map = {i: orig_idx for i, orig_idx in enumerate(matched_indices)}
            
            clusters_subset = cluster_keypoints(subset_kps, img_shape)
            clusters = [[subset_map[i] for i in cluster] for cluster in clusters_subset]
            self.logger.info(f"Found {len(clusters)} clusters.")
        else:
            clusters = []
        
        # 6. Create Mask from all matched keypoints
        all_matched_kps = np.vstack([consistent_kpts1, consistent_kpts2]) if len(consistent_kpts1) > 0 else np.array([])
        if len(all_matched_kps) > 0:
            mask = create_mask_from_keypoints(all_matched_kps, img_shape)
        else:
            mask = np.zeros(img_shape, dtype=np.uint8)
        
        # 7. Save Outputs
        mask_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
        save_mask(mask, mask_path)
        
        matches_path = os.path.join(self.output_dir, f"{base_name}_matches.png")
        draw_matches_on_single_image(img, kps, valid_matches, matches_path)
        
        clusters_path = os.path.join(self.output_dir, f"{base_name}_clusters.png")
        draw_clusters(img, kps, clusters, clusters_path)
        
        return {
            'success': True,
            'found_forgery': len(valid_matches) >= self.min_keypoints,
            'matched_keypoints': len(valid_matches),
            'num_clusters': len(clusters),
            'mask_path': mask_path,
            'matches_path': matches_path,
            'clusters_path': clusters_path
        }
    
    def detect_cross_image(self, source_path: str, target_path: str) -> Dict[str, Any]:
        """
        Run copy-move detection between two images.
        
        Detects regions that have been copied from source to target image.
        
        Args:
            source_path: Path to the source image
            target_path: Path to the target image
            
        Returns:
            Dictionary with detection results and output file paths
        """
        self.logger.info(f"Processing Cross Image: {source_path} vs {target_path}")
        source_base = os.path.splitext(os.path.basename(source_path))[0]
        target_base = os.path.splitext(os.path.basename(target_path))[0]
        base_name = f"{source_base}_vs_{target_base}"
        
        # 1. Load Images
        img1 = cv2.imread(source_path)
        img2 = cv2.imread(target_path)
        
        if img1 is None or img2 is None:
            self.logger.error("Failed to load one or both images.")
            return {'success': False, 'error': 'Failed to load images'}
        
        img1_shape = img1.shape[:2]
        img2_shape = img2.shape[:2]
        
        # 2. Extract Features
        self.logger.info(f"Extracting features using {self.descriptor_type.value}...")
        kps1, descs1, flip_kps1, flip_descs1 = extract_features(
            source_path,
            descriptor_type=self.descriptor_type,
            extract_flip=self.check_flip
        )
        kps2, descs2, _, _ = extract_features(
            target_path,
            descriptor_type=self.descriptor_type,
            extract_flip=False  # Only flip source image
        )
        self.logger.info(f"Extracted {len(kps1)} (source) and {len(kps2)} (target) keypoints.")
        
        if len(kps1) == 0 or len(kps2) == 0:
            self.logger.warning("Not enough keypoints found.")
            return {'success': False, 'error': 'Not enough keypoints'}
        
        # 3. Match and Verify
        self.logger.info(f"Matching and verifying using {self.alignment_strategy.value}...")
        match_result = match_and_verify(
            kps1, descs1,
            kps2, descs2,
            flip_keypoints1=flip_kps1,
            flip_descriptors1=flip_descs1,
            image1_shape=img1_shape,
            image2_shape=img2_shape,
            alignment_strategy=self.alignment_strategy,
            matching_method=self.matching_method,
            min_keypoints=self.min_keypoints,
            min_area=self.min_area,
            check_flip=self.check_flip
        )
        
        matched_kpts1 = match_result['matched_kpts1']
        matched_kpts2 = match_result['matched_kpts2']
        is_flipped = match_result['is_flipped_match']
        
        self.logger.info(f"Found {match_result['matched_keypoints']} matched keypoints.")
        self.logger.info(f"Shared area: {match_result['shared_area_img1']:.2%} (source), {match_result['shared_area_img2']:.2%} (target)")
        if is_flipped:
            self.logger.info("Match found with flipped image.")
        
        # 4. Cluster matched keypoints
        clusters1 = []
        clusters2 = []
        
        if len(matched_kpts1) >= 3:
            clusters1_raw = cluster_keypoints(matched_kpts1, img1_shape)
            clusters1 = clusters1_raw
            
        if len(matched_kpts2) >= 3:
            clusters2_raw = cluster_keypoints(matched_kpts2, img2_shape)
            clusters2 = clusters2_raw
        
        self.logger.info(f"Found {len(clusters1)} clusters (source), {len(clusters2)} clusters (target).")
        
        # 5. Create Masks
        mask1 = create_mask_from_keypoints(matched_kpts1, img1_shape) if len(matched_kpts1) > 0 else np.zeros(img1_shape, dtype=np.uint8)
        mask2 = create_mask_from_keypoints(matched_kpts2, img2_shape) if len(matched_kpts2) > 0 else np.zeros(img2_shape, dtype=np.uint8)
        
        # 6. Save Outputs
        # Save individual masks
        mask1_path = os.path.join(self.output_dir, f"{base_name}_maskA.png")
        mask2_path = os.path.join(self.output_dir, f"{base_name}_maskB.png")
        save_mask(mask1, mask1_path)
        save_mask(mask2, mask2_path)
        
        # Combined mask (side by side)
        combined_mask_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        combined_h = max(h1, h2)
        combined_mask = np.zeros((combined_h, w1 + w2), dtype=np.uint8)
        combined_mask[:h1, :w1] = mask1
        combined_mask[:h2, w1:] = mask2
        save_mask(combined_mask, combined_mask_path)
        
        # Matches visualization
        matches_path = os.path.join(self.output_dir, f"{base_name}_matches.png")
        draw_matches_with_hulls(img1, matched_kpts1, img2, matched_kpts2, matches_path)
        
        # Clusters visualization (with linked coloring)
        clusters_path = os.path.join(self.output_dir, f"{base_name}_clusters.png")
        
        # Build match indices for linked cluster coloring
        # match_indices maps keypoint indices in matched_kpts arrays
        match_indices = list(range(len(matched_kpts1)))  # 0, 1, 2, ... paired with same index
        match_indices = [(i, i) for i in range(min(len(matched_kpts1), len(matched_kpts2)))]
        
        draw_clusters_with_hulls(
            img1, matched_kpts1,
            img2, matched_kpts2,
            clusters1, clusters2,
            clusters_path,
            match_indices=match_indices
        )
        
        return {
            'success': True,
            'found_forgery': match_result['is_match'],
            'matched_keypoints': match_result['matched_keypoints'],
            'shared_area_source': match_result['shared_area_img1'],
            'shared_area_target': match_result['shared_area_img2'],
            'is_flipped': is_flipped,
            'num_clusters_source': len(clusters1),
            'num_clusters_target': len(clusters2),
            'mask_path': combined_mask_path,
            'mask_source_path': mask1_path,
            'mask_target_path': mask2_path,
            'matches_path': matches_path,
            'clusters_path': clusters_path
        }


# Backward compatibility aliases
def get_descriptor_type(method: int) -> DescriptorType:
    """Convert legacy method int to DescriptorType."""
    mapping = {
        1: DescriptorType.CV_SIFT,
        2: DescriptorType.CV_RSIFT,
        3: DescriptorType.VLFEAT_SIFT_HEQ,
    }
    return mapping.get(method, DescriptorType.CV_RSIFT)
