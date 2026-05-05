"""
Visualization module for keypoint-based copy-move detection.

Generates output files compatible with the copy-move-detection module:
- mask.png: Binary mask of detected regions
- matches.png: Visualization of matched keypoints
- clusters.png: Colored cluster visualization (matching clusters share same color)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)


def create_mask_from_keypoints(
    keypoints: np.ndarray,
    image_shape: Tuple[int, int],
    dilation_radius: int = 15
) -> np.ndarray:
    """
    Create a binary mask from matched keypoints using convex hull.
    
    Args:
        keypoints: Nx2 array of (x, y) keypoint coordinates
        image_shape: (height, width) of the output mask
        dilation_radius: Radius for morphological dilation
        
    Returns:
        Binary mask (0/1) of shape (height, width)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if len(keypoints) < 3:
        # Not enough points for convex hull, just mark individual points
        for kp in keypoints:
            x, y = int(round(kp[0])), int(round(kp[1]))
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                cv2.circle(mask, (x, y), dilation_radius, 1, -1)
        return mask
    
    # Compute convex hull
    hull = cv2.convexHull(keypoints.astype(np.int32))
    cv2.fillPoly(mask, [hull], 1)
    
    # Apply dilation to expand the mask slightly
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2, dilation_radius * 2))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def draw_matches(
    image1: np.ndarray,
    kps1: np.ndarray,
    image2: np.ndarray,
    kps2: np.ndarray,
    matches: List[Tuple[int, int]],
    output_path: str,
    max_lines: int = 500
) -> None:
    """
    Draw lines between matched keypoints on two images (side-by-side).
    """
    # Ensure BGR format
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create canvas
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1:w1+w2] = image2
    
    # Draw matches
    step = max(1, len(matches) // max_lines)
    
    for i in range(0, len(matches), step):
        idx1, idx2 = matches[i]
        
        if idx1 >= len(kps1) or idx2 >= len(kps2):
            continue
            
        pt1 = (int(kps1[idx1][0]), int(kps1[idx1][1]))
        pt2 = (int(kps2[idx2][0]) + w1, int(kps2[idx2][1]))
        
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)
    
    cv2.line(canvas, (w1, 0), (w1, canvas_h), (128, 128, 128), 2)
    cv2.imwrite(output_path, canvas)
    logger.info(f"Saved matches visualization to {output_path}")


def draw_matches_with_hulls(
    image1: np.ndarray,
    matched_kpts1: np.ndarray,
    image2: np.ndarray,
    matched_kpts2: np.ndarray,
    output_path: str,
    max_lines: int = 500
) -> None:
    """
    Draw matched regions with lines connecting matched keypoints (copy-move-detection style).
    Shows: green source points, red target points, white connecting lines.
    """
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create canvas (side by side)
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1:w1+w2] = image2
    
    # Draw lines connecting matched keypoints
    n_matches = min(len(matched_kpts1), len(matched_kpts2))
    step = max(1, n_matches // max_lines)
    
    for i in range(0, n_matches, step):
        pt1 = (int(matched_kpts1[i][0]), int(matched_kpts1[i][1]))
        pt2 = (int(matched_kpts2[i][0]) + w1, int(matched_kpts2[i][1]))
        
        # Draw white line connecting matched points
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw all keypoints on top (source in green, target in red)
    for i in range(n_matches):
        pt1 = (int(matched_kpts1[i][0]), int(matched_kpts1[i][1]))
        pt2 = (int(matched_kpts2[i][0]) + w1, int(matched_kpts2[i][1]))
        cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)  # Green source
        cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)  # Red target
    
    # Draw separator line
    cv2.line(canvas, (w1, 0), (w1, canvas_h), (128, 128, 128), 2)
    
    cv2.imwrite(output_path, canvas)
    logger.info(f"Saved matches visualization to {output_path}")


def draw_matches_on_single_image(
    image: np.ndarray,
    kps: np.ndarray,
    matches: List[Tuple[int, int]],
    output_path: str
) -> None:
    """
    Draw lines connecting matched keypoints on a single image.
    """
    out_img = image.copy()
    if len(out_img.shape) == 2:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        
    for idx1, idx2 in matches:
        if idx1 >= len(kps) or idx2 >= len(kps):
            continue
        pt1 = tuple(map(int, kps[idx1]))
        pt2 = tuple(map(int, kps[idx2]))
        
        cv2.line(out_img, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(out_img, pt1, 3, (0, 0, 255), -1)
        cv2.circle(out_img, pt2, 3, (255, 0, 0), -1)
        
    cv2.imwrite(output_path, out_img)
    logger.info(f"Saved single-image matches to {output_path}")


def draw_clusters(
    image: np.ndarray,
    keypoints: np.ndarray,
    clusters: List[List[int]],
    output_path: str
) -> None:
    """
    Draw convex hulls for each cluster on a single image.
    For single-image copy-move, each cluster represents a source/target region.
    """
    out_img = image.copy()
    if len(out_img.shape) == 2:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    
    if len(clusters) == 0:
        cv2.imwrite(output_path, out_img)
        logger.info(f"Saved clusters visualization to {output_path}")
        return
    
    # Bright distinct colors for source (green) and target (red/blue) regions
    CLUSTER_COLORS = [
        (0, 255, 0),    # Green - source region
        (0, 0, 255),    # Red - target region (copied from source)
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]
    
    overlay = out_img.copy()
    
    # Sort clusters by size (larger first) for better visualization
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)
    
    for rank, (orig_idx, cluster_indices) in enumerate(sorted_clusters):
        if len(cluster_indices) < 3:
            continue
            
        color = CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
        cluster_pts = np.array([keypoints[idx] for idx in cluster_indices if idx < len(keypoints)])
        
        if len(cluster_pts) >= 3:
            hull = cv2.convexHull(cluster_pts.astype(np.int32))
            # Draw filled polygon with stronger transparency
            temp = overlay.copy()
            cv2.fillPoly(temp, [hull], color)
            overlay = cv2.addWeighted(temp, 0.5, overlay, 0.5, 0)
            # Draw white outline (outer, thick)
            cv2.drawContours(overlay, [hull], 0, (255, 255, 255), 5)
            # Draw colored outline (inner)
            cv2.drawContours(overlay, [hull], 0, color, 3)
            # Draw keypoint dots (larger, white with colored center)
            for kp in cluster_pts.astype(np.int32):
                cv2.circle(overlay, tuple(kp), 4, (255, 255, 255), -1)  # White outer
                cv2.circle(overlay, tuple(kp), 2, color, -1)  # Colored center
            
    cv2.imwrite(output_path, overlay)
    logger.info(f"Saved clusters visualization to {output_path}")


def draw_linked_clusters_cross_image(
    image1: np.ndarray,
    matched_kpts1: np.ndarray,
    image2: np.ndarray,
    matched_kpts2: np.ndarray,
    match_indices: List[Tuple[int, int]],
    clusters1: List[List[int]],
    clusters2: List[List[int]],
    output_path: str
) -> np.ndarray:
    """
    Draw clusters on two images with matching clusters sharing the same color.
    
    This follows copy-move-detection's approach:
    1. Build adjacency matrix between clusters based on keypoint matches
    2. Find connected components
    3. Color connected clusters with same color
    
    Args:
        image1: Source image
        matched_kpts1: Matched keypoints in image 1
        image2: Target image
        matched_kpts2: Matched keypoints in image 2
        match_indices: List of (idx1, idx2) pairs linking keypoints
        clusters1: Cluster assignments for image 1 keypoints (list of index lists)
        clusters2: Cluster assignments for image 2 keypoints (list of index lists)
        output_path: Path to save visualization
        
    Returns:
        Combined visualization image
    """
    # Ensure BGR format
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Resize to same height for side-by-side
    if h1 != h2:
        target_h = max(h1, h2)
        if h1 < target_h:
            scale = target_h / h1
            image1 = cv2.resize(image1, (int(w1 * scale), target_h))
            # Scale keypoints
            if len(matched_kpts1) > 0:
                matched_kpts1 = matched_kpts1 * scale
            h1, w1 = image1.shape[:2]
        else:
            scale = target_h / h2
            image2 = cv2.resize(image2, (int(w2 * scale), target_h))
            if len(matched_kpts2) > 0:
                matched_kpts2 = matched_kpts2 * scale
            h2, w2 = image2.shape[:2]
    
    # Create canvas
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1:w1+w2] = image2
    
    nA = len(clusters1) if clusters1 else 0
    nB = len(clusters2) if clusters2 else 0
    n_total = nA + nB
    
    if n_total == 0 or len(matched_kpts1) == 0 or len(matched_kpts2) == 0:
        cv2.imwrite(output_path, canvas)
        logger.info(f"Saved clusters visualization to {output_path}")
        return canvas
    
    # Build mapping from keypoint index to cluster index
    kp1_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters1):
        for kp_idx in cluster:
            kp1_to_cluster[kp_idx] = cluster_idx
    
    kp2_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters2):
        for kp_idx in cluster:
            kp2_to_cluster[kp_idx] = cluster_idx
    
    # Build adjacency matrix based on keypoint matches
    # Nodes 0..nA-1 are image1 clusters, nA..nA+nB-1 are image2 clusters
    adj = np.zeros((n_total, n_total), dtype=int)
    
    for idx1, idx2 in match_indices:
        cluster1 = kp1_to_cluster.get(idx1)
        cluster2 = kp2_to_cluster.get(idx2)
        
        if cluster1 is not None and cluster2 is not None:
            adj[cluster1, nA + cluster2] = 1
            adj[nA + cluster2, cluster1] = 1
    
    # Find connected components
    n_components, labels = connected_components(csr_matrix(adj), directed=False)
    
    # Bright, easily distinguishable colors in BGR format for OpenCV
    VIBRANT_COLORS = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 128, 255),    # Orange
        (128, 0, 255),    # Pink
        (0, 165, 255),    # Orange-red
        (147, 20, 255),   # Deep pink
    ]
    
    # Create overlay canvas
    overlay = canvas.copy()
    
    # Single convex hull color
    hull_color = (0, 255, 0)  # Green in BGR
    
    # Draw single convex hull for all matched keypoints in image 1
    if len(matched_kpts1) >= 3:
        hull1 = cv2.convexHull(matched_kpts1.astype(np.int32))
        # Draw filled polygon with transparency
        temp = overlay.copy()
        cv2.fillPoly(temp, [hull1], hull_color)
        overlay = cv2.addWeighted(temp, 0.4, overlay, 0.6, 0)
        # Draw white outline (outer)
        cv2.drawContours(overlay, [hull1], 0, (255, 255, 255), 4)
        # Draw colored outline (inner)
        cv2.drawContours(overlay, [hull1], 0, hull_color, 2)
        # Draw keypoint dots
        for kp in matched_kpts1.astype(np.int32):
            cv2.circle(overlay, tuple(kp), 2, (0, 0, 255), -1)  # Red dots
    
    # Draw single convex hull for all matched keypoints in image 2
    if len(matched_kpts2) >= 3:
        # Shift keypoints to canvas position
        kpts2_shifted = matched_kpts2.copy()
        kpts2_shifted[:, 0] += w1
        
        hull2 = cv2.convexHull(kpts2_shifted.astype(np.int32))
        # Draw filled polygon with transparency
        temp = overlay.copy()
        cv2.fillPoly(temp, [hull2], hull_color)
        overlay = cv2.addWeighted(temp, 0.4, overlay, 0.6, 0)
        # Draw white outline (outer)
        cv2.drawContours(overlay, [hull2], 0, (255, 255, 255), 4)
        # Draw colored outline (inner)
        cv2.drawContours(overlay, [hull2], 0, hull_color, 2)
        # Draw keypoint dots
        for kp in kpts2_shifted.astype(np.int32):
            cv2.circle(overlay, tuple(kp), 2, (0, 0, 255), -1)  # Red dots
    
    # Draw separator line
    cv2.line(overlay, (w1, 0), (w1, canvas_h), (128, 128, 128), 2)
    
    cv2.imwrite(output_path, overlay)
    logger.info(f"Saved clusters visualization to {output_path}")
    
    return overlay


def draw_clusters_with_hulls(
    image1: np.ndarray,
    matched_kpts1: np.ndarray,
    image2: np.ndarray,
    matched_kpts2: np.ndarray,
    clusters1: Optional[List[List[int]]] = None,
    clusters2: Optional[List[List[int]]] = None,
    output_path: str = None,
    match_indices: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """
    Draw a single convex hull of all matched keypoints on each image.
    """
    # Ensure BGR format
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create canvas (side by side)
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1:w1+w2] = image2
    
    overlay = canvas.copy()
    hull_color = (0, 255, 0)  # Green in BGR
    
    # Draw single convex hull for all matched keypoints in image 1
    if len(matched_kpts1) >= 3:
        hull1 = cv2.convexHull(matched_kpts1.astype(np.int32))
        # Draw filled polygon with transparency
        temp = overlay.copy()
        cv2.fillPoly(temp, [hull1], hull_color)
        overlay = cv2.addWeighted(temp, 0.4, overlay, 0.6, 0)
        # Draw white outline (outer)
        cv2.drawContours(overlay, [hull1], 0, (255, 255, 255), 4)
        # Draw colored outline (inner)
        cv2.drawContours(overlay, [hull1], 0, hull_color, 2)
        # Draw keypoint dots
        for kp in matched_kpts1.astype(np.int32):
            cv2.circle(overlay, tuple(kp), 2, (0, 0, 255), -1)  # Red dots
    
    # Draw single convex hull for all matched keypoints in image 2
    if len(matched_kpts2) >= 3:
        # Shift keypoints to canvas position
        kpts2_shifted = matched_kpts2.copy()
        kpts2_shifted[:, 0] += w1
        
        hull2 = cv2.convexHull(kpts2_shifted.astype(np.int32))
        # Draw filled polygon with transparency
        temp = overlay.copy()
        cv2.fillPoly(temp, [hull2], hull_color)
        overlay = cv2.addWeighted(temp, 0.4, overlay, 0.6, 0)
        # Draw white outline (outer)
        cv2.drawContours(overlay, [hull2], 0, (255, 255, 255), 4)
        # Draw colored outline (inner)
        cv2.drawContours(overlay, [hull2], 0, hull_color, 2)
        # Draw keypoint dots
        for kp in kpts2_shifted.astype(np.int32):
            cv2.circle(overlay, tuple(kp), 2, (0, 0, 255), -1)  # Red dots
    
    # Draw separator line
    cv2.line(overlay, (w1, 0), (w1, canvas_h), (128, 128, 128), 2)
    
    if output_path:
        cv2.imwrite(output_path, overlay)
        logger.info(f"Saved clusters with hulls to {output_path}")
    
    return overlay


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """
    Save a binary mask as an image.
    """
    if mask.max() <= 1:
        mask_img = (mask * 255).astype(np.uint8)
    else:
        mask_img = mask.astype(np.uint8)
    
    cv2.imwrite(output_path, mask_img)
    logger.info(f"Saved mask to {output_path}")
