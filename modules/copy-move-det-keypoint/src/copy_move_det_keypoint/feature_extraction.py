"""
Feature extraction module for keypoint-based copy-move detection.

Supports multiple descriptor types from provenance-analysis:
- VLFeat SIFT with histogram equalization (best for scientific images)
- OpenCV SIFT
- OpenCV RootSIFT
"""

import cv2
import numpy as np
from PIL import Image
from enum import Enum
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DescriptorType(str, Enum):
    """Supported keypoint descriptor types."""
    VLFEAT_SIFT_HEQ = "vlfeat_sift_heq"  # VLFeat SIFT with histogram equalization
    CV_SIFT = "cv_sift"                   # OpenCV SIFT
    CV_RSIFT = "cv_rsift"                 # OpenCV RootSIFT (default)


# Initialize CLAHE for contrast enhancement
CLAHE_APPLIER = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _increase_image_if_necessary(image: np.ndarray, min_size: int = 300) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """
    Upscales the image if it's too small for reliable feature detection.
    
    Returns:
        Tuple of (image, (height, width), resize_count)
    """
    resize_count = 0
    h, w = image.shape[:2]
    
    while h < min_size or w < min_size:
        image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]
        resize_count += 1
        
    return image, (h, w), resize_count


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image: Grayscale image as numpy array
        
    Returns:
        Histogram-equalized image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(image)


def extract_vlfeat_sift_heq(
    image: np.ndarray,
    peak_thresh: float = 0.01,
    edge_thresh: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT keypoints and descriptors using VLFeat with histogram equalization.
    
    This is the recommended descriptor for scientific image analysis as it
    provides better matching for images with similar textures.
    
    Args:
        image: Input image (grayscale or RGB)
        peak_thresh: Peak threshold for SIFT
        edge_thresh: Edge threshold for SIFT
        
    Returns:
        Tuple of (keypoints, descriptors)
        - keypoints: Nx2 array of (x, y) coordinates
        - descriptors: NxD array of descriptors
    """
    try:
        from cyvlfeat.sift import sift
    except ImportError:
        raise ImportError(
            "cyvlfeat is required for vlfeat_sift_heq descriptor. "
            "Install via conda: conda install -c conda-forge cyvlfeat"
        )
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply histogram equalization
    gray_heq = histogram_equalization(gray)
    
    # Convert to float32 for VLFeat (expects values in [0, 255])
    gray_float = gray_heq.astype(np.float32)
    
    # Extract SIFT features using VLFeat
    frames, descriptors = sift(
        gray_float,
        peak_thresh=peak_thresh,
        edge_thresh=edge_thresh,
        compute_descriptor=True
    )
    
    if frames is None or len(frames) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)
    
    # VLFeat returns frames as (y, x, scale, orientation)
    # We need (x, y) for keypoints
    keypoints = frames[:, :2][:, ::-1]  # Swap to (x, y)
    
    return keypoints.astype(np.float32), descriptors.astype(np.float32)


def extract_cv_sift(
    image: np.ndarray,
    n_features: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT keypoints and descriptors using OpenCV.
    
    Args:
        image: Input image (grayscale or RGB)
        n_features: Maximum number of features (0 = unlimited)
        contrast_threshold: Contrast threshold for SIFT
        edge_threshold: Edge threshold for SIFT
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold
    )
    
    # Detect and compute
    kps, descriptors = sift.detectAndCompute(gray, None)
    
    if kps is None or len(kps) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)
    
    # Convert keypoints to numpy array
    keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    
    return keypoints, descriptors.astype(np.float32)


def extract_cv_rsift(
    image: np.ndarray,
    n_features: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    eps: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract RootSIFT keypoints and descriptors using OpenCV.
    
    RootSIFT applies Hellinger normalization to SIFT descriptors,
    which can improve matching performance.
    
    Args:
        image: Input image (grayscale or RGB)
        n_features: Maximum number of features (0 = unlimited)
        contrast_threshold: Contrast threshold for SIFT
        edge_threshold: Edge threshold for SIFT
        eps: Epsilon for numerical stability
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    keypoints, descriptors = extract_cv_sift(
        image, n_features, contrast_threshold, edge_threshold
    )
    
    if len(descriptors) == 0:
        return keypoints, descriptors
    
    # Apply RootSIFT transformation (Hellinger kernel)
    # 1. L1 normalize
    descriptors = descriptors / (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + eps)
    # 2. Square root
    descriptors = np.sqrt(descriptors)
    
    return keypoints, descriptors.astype(np.float32)


def extract_features(
    image_path: str,
    descriptor_type: DescriptorType = DescriptorType.CV_RSIFT,
    extract_flip: bool = True,
    kp_count: int = 2000
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract keypoint descriptors from an image.
    
    Args:
        image_path: Path to the input image
        descriptor_type: Type of descriptor to extract
        extract_flip: Also extract from horizontally flipped image
        kp_count: Maximum number of keypoints to retain (for cv_sift/cv_rsift)
        
    Returns:
        Tuple of (keypoints, descriptors, flip_keypoints, flip_descriptors)
        - keypoints: Nx2 array of (x, y) coordinates
        - descriptors: Nx128 array of descriptors
        - flip_keypoints: Nx2 array for flipped image (or None)
        - flip_descriptors: Nx128 array for flipped image (or None)
    """
    # Load image
    try:
        pil_image = Image.open(image_path)
        image = np.array(pil_image).astype(np.uint8)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128), None, None
    
    # Handle resizing for small images
    original_shape = image.shape[:2]
    image, _, resize_count = _increase_image_if_necessary(image)
    
    # Store width for flip coordinate adjustment
    width = image.shape[1]
    
    # Select extraction function based on descriptor type
    if descriptor_type == DescriptorType.VLFEAT_SIFT_HEQ:
        extract_fn = extract_vlfeat_sift_heq
        keypoints, descriptors = extract_fn(image)
    elif descriptor_type == DescriptorType.CV_SIFT:
        extract_fn = extract_cv_sift
        keypoints, descriptors = extract_fn(image, n_features=kp_count)
    elif descriptor_type == DescriptorType.CV_RSIFT:
        extract_fn = extract_cv_rsift
        keypoints, descriptors = extract_fn(image, n_features=kp_count)
    else:
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")
    
    # Adjust keypoint coordinates back to original scale if resized
    if resize_count > 0:
        scale_factor = 2.0 ** resize_count
        if len(keypoints) > 0:
            keypoints = keypoints / scale_factor
    
    logger.info(f"Extracted {len(keypoints)} keypoints from {image_path} using {descriptor_type.value}")
    
    # Extract from flipped image if requested
    flip_keypoints = None
    flip_descriptors = None
    
    if extract_flip:
        flipped_image = np.fliplr(image)
        
        if descriptor_type == DescriptorType.VLFEAT_SIFT_HEQ:
            flip_keypoints, flip_descriptors = extract_vlfeat_sift_heq(flipped_image)
        elif descriptor_type == DescriptorType.CV_SIFT:
            flip_keypoints, flip_descriptors = extract_cv_sift(flipped_image, n_features=kp_count)
        elif descriptor_type == DescriptorType.CV_RSIFT:
            flip_keypoints, flip_descriptors = extract_cv_rsift(flipped_image, n_features=kp_count)
        
        # Adjust x-coordinates for flip (x' = width - 1 - x) and scale
        if len(flip_keypoints) > 0:
            if resize_count > 0:
                flip_keypoints = flip_keypoints / scale_factor
                adjusted_width = width / scale_factor
            else:
                adjusted_width = width
            flip_keypoints[:, 0] = adjusted_width - 1 - flip_keypoints[:, 0]
        
        logger.info(f"Extracted {len(flip_keypoints)} keypoints from flipped image")
    
    return keypoints, descriptors, flip_keypoints, flip_descriptors


# Backward compatibility: keep old function signature working
def extract_features_legacy(
    image_path: str,
    method: str = 'sift',
    kp_count: int = 2000,
    contrast_threshold: float = 0.04,
    sigma: float = 1.6,
    eps: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy function for backward compatibility.
    Uses CV RootSIFT with CLAHE (original behavior).
    """
    keypoints, descriptors, _, _ = extract_features(
        image_path,
        descriptor_type=DescriptorType.CV_RSIFT,
        extract_flip=False,
        kp_count=kp_count
    )
    return keypoints, descriptors


def extract_features_from_image(
    image: np.ndarray,
    descriptor_type: DescriptorType = DescriptorType.CV_RSIFT,
    extract_flip: bool = True,
    kp_count: int = 2000,
    source_name: str = "<in-memory>"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract keypoint descriptors from an *already loaded* image array.

    This is a non-breaking addition used by the public API to support:
      - numpy arrays
      - PIL.Image objects
      - cached / preloaded images

    The implementation mirrors `extract_features()` after image load to preserve
    behavior and accuracy.

    Args:
        image: Input image as numpy array (uint8). Expected RGB or grayscale.
        descriptor_type: Type of descriptor to extract.
        extract_flip: Also extract from horizontally flipped image.
        kp_count: Maximum number of keypoints to retain (for cv_sift/cv_rsift).
        source_name: Used only for logging/debug messages.

    Returns:
        (keypoints, descriptors, flip_keypoints, flip_descriptors)
    """
    if image is None:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128), None, None

    try:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            # Best-effort conversion; keep behavior stable for uint8 callers.
            # If float in [0,1], scale to [0,255].
            if np.issubdtype(image.dtype, np.floating):
                maxv = float(np.nanmax(image)) if image.size else 0.0
                if maxv <= 1.0:
                    image = (image * 255.0).clip(0, 255)
            image = image.clip(0, 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Error normalizing in-memory image {source_name}: {e}")
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128), None, None

    # Handle resizing for small images
    image, _, resize_count = _increase_image_if_necessary(image)

    # Store width for flip coordinate adjustment (width of the resized image)
    width = image.shape[1]

    # Select extraction function based on descriptor type
    if descriptor_type == DescriptorType.VLFEAT_SIFT_HEQ:
        keypoints, descriptors = extract_vlfeat_sift_heq(image)
    elif descriptor_type == DescriptorType.CV_SIFT:
        keypoints, descriptors = extract_cv_sift(image, n_features=kp_count)
    elif descriptor_type == DescriptorType.CV_RSIFT:
        keypoints, descriptors = extract_cv_rsift(image, n_features=kp_count)
    else:
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")

    # Adjust keypoint coordinates back to original scale if resized
    if resize_count > 0:
        scale_factor = 2.0 ** resize_count
        if len(keypoints) > 0:
            keypoints = keypoints / scale_factor

    logger.info(
        f"Extracted {len(keypoints)} keypoints from {source_name} using {descriptor_type.value}"
    )

    flip_keypoints = None
    flip_descriptors = None

    if extract_flip:
        flipped_image = np.fliplr(image)

        if descriptor_type == DescriptorType.VLFEAT_SIFT_HEQ:
            flip_keypoints, flip_descriptors = extract_vlfeat_sift_heq(flipped_image)
        elif descriptor_type == DescriptorType.CV_SIFT:
            flip_keypoints, flip_descriptors = extract_cv_sift(flipped_image, n_features=kp_count)
        elif descriptor_type == DescriptorType.CV_RSIFT:
            flip_keypoints, flip_descriptors = extract_cv_rsift(flipped_image, n_features=kp_count)

        # Adjust x-coordinates for flip (x' = width - 1 - x) and scale
        if flip_keypoints is not None and len(flip_keypoints) > 0:
            if resize_count > 0:
                flip_keypoints = flip_keypoints / scale_factor
                adjusted_width = width / scale_factor
            else:
                adjusted_width = width
            flip_keypoints[:, 0] = adjusted_width - 1 - flip_keypoints[:, 0]

        logger.info(
            f"Extracted {0 if flip_keypoints is None else len(flip_keypoints)} keypoints from flipped image"
        )

    return keypoints, descriptors, flip_keypoints, flip_descriptors
