"""
Keypoint-based Copy-Move Detection CLI.

Supports:
- Single-image detection (1 input)
- Cross-image detection (2 inputs)
- Multiple descriptor types
- Multiple alignment strategies

Output format compatible with copy-move-detection module:
- {base}_mask.png
- {base}_matches.png  
- {base}_clusters.png
"""

import argparse
import logging
import sys
import os

# Imports:
# - Prefer package imports (python -m copy_move_det_keypoint.run_detection)
# - Fall back to script-style execution (python run_detection.py) by adjusting sys.path
try:
    from .detector import KeypointCopyMoveDetector
    from .feature_extraction import DescriptorType
    from .matching import AlignmentStrategy, MatchingMethod
except ImportError:  # pragma: no cover
    # Add this directory to path for local imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detector import KeypointCopyMoveDetector
    from feature_extraction import DescriptorType
    from matching import AlignmentStrategy, MatchingMethod
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_detection")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keypoint-based Copy-Move Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image detection with default settings
  python run_detection.py --input image.png --output ./output
  
  # Cross-image detection with VLFeat SIFT HEQ
  python run_detection.py --input source.png target.png --output ./output \\
      --descriptor vlfeat_sift_heq
  
  # Cross-image detection with RANSAC alignment
  python run_detection.py --input source.png target.png --output ./output \\
      --alignment CV_RANSAC --no-check-flip

Descriptor Types:
  cv_sift         - OpenCV SIFT
  cv_rsift        - OpenCV RootSIFT (default)
  vlfeat_sift_heq - VLFeat SIFT with histogram equalization

Alignment Strategies:
  CV_MAGSAC       - MAGSAC++ (recommended, default)
  CV_RANSAC       - Classic RANSAC
  CV_LMEDS        - Least Median of Squares
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        nargs='+',
        required=True,
        help="Input image path(s). 1 for single-image, 2 for cross-image detection."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for results."
    )
    
    # Descriptor options
    parser.add_argument(
        "--descriptor", "-d",
        type=str,
        default="cv_rsift",
        choices=["cv_sift", "cv_rsift", "vlfeat_sift_heq"],
        help="Feature descriptor type (default: cv_rsift)"
    )
    
    # Alignment options
    parser.add_argument(
        "--alignment", "-a",
        type=str,
        default="CV_MAGSAC",
        choices=["CV_MAGSAC", "CV_RANSAC", "CV_LMEDS"],
        help="Geometric alignment strategy (default: CV_MAGSAC)"
    )
    
    # Matching options
    parser.add_argument(
        "--matching-method",
        type=str,
        default="BF",
        choices=["BF", "FLANN"],
        help="Keypoint matching method (default: BF)"
    )
    
    # Detection parameters
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=20,
        help="Minimum matched keypoints for valid detection (default: 20)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.01,
        help="Minimum shared area threshold (default: 0.01)"
    )
    
    # Flip detection
    parser.add_argument(
        "--check-flip",
        action="store_true",
        default=True,
        help="Check for horizontally flipped matches (default: True)"
    )
    parser.add_argument(
        "--no-check-flip",
        action="store_true",
        help="Disable flip detection"
    )
    
    # Other options
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Detection timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Legacy compatibility
    parser.add_argument(
        "--method",
        type=int,
        default=None,
        help="Legacy method ID (1=cv_sift, 2=cv_rsift, 3=vlfeat_sift_heq)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Handle legacy method argument
    if args.method is not None:
        method_map = {
            1: "cv_sift",
            2: "cv_rsift",
            3: "vlfeat_sift_heq"
        }
        args.descriptor = method_map.get(args.method, "cv_rsift")
        logger.info(f"Using legacy method {args.method} -> {args.descriptor}")
    
    # Convert string args to enums
    descriptor_type = DescriptorType(args.descriptor)
    alignment_strategy = AlignmentStrategy(args.alignment)
    matching_method = MatchingMethod(args.matching_method)
    
    # Handle flip detection
    check_flip = args.check_flip and not args.no_check_flip
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Descriptor: {descriptor_type.value}")
    logger.info(f"  Alignment: {alignment_strategy.value}")
    logger.info(f"  Matching: {matching_method.value}")
    logger.info(f"  Min keypoints: {args.min_keypoints}")
    logger.info(f"  Min area: {args.min_area}")
    logger.info(f"  Check flip: {check_flip}")
    
    # Initialize detector
    detector = KeypointCopyMoveDetector(
        output_dir=args.output,
        descriptor_type=descriptor_type,
        alignment_strategy=alignment_strategy,
        matching_method=matching_method,
        check_flip=check_flip,
        min_keypoints=args.min_keypoints,
        min_area=args.min_area,
        timeout=args.timeout
    )
    
    # Run detection
    input_paths = args.input
    
    if len(input_paths) == 1:
        # Single Image Detection
        logger.info(f"Running single-image detection on {input_paths[0]}")
        result = detector.detect_single_image(input_paths[0])
        
    elif len(input_paths) == 2:
        # Cross-Image Detection
        logger.info(f"Running cross-image detection: {input_paths[0]} vs {input_paths[1]}")
        result = detector.detect_cross_image(input_paths[0], input_paths[1])
        
    else:
        logger.error("Please provide 1 or 2 input images.")
        sys.exit(1)
    
    # Print results
    if result['success']:
        logger.info("Detection completed successfully.")
        logger.info(f"  Found forgery: {result.get('found_forgery', False)}")
        logger.info(f"  Matched keypoints: {result.get('matched_keypoints', 0)}")
        
        if 'shared_area_source' in result:
            logger.info(f"  Shared area (source): {result['shared_area_source']:.2%}")
            logger.info(f"  Shared area (target): {result['shared_area_target']:.2%}")
        
        logger.info(f"Output files:")
        logger.info(f"  Mask: {result.get('mask_path', 'N/A')}")
        logger.info(f"  Matches: {result.get('matches_path', 'N/A')}")
        logger.info(f"  Clusters: {result.get('clusters_path', 'N/A')}")
    else:
        logger.error(f"Detection failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
