import os
import cv2
import argparse
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# -------------------------------
# 1. Core robust border detector
# -------------------------------
def detect_border_bbox_robust(
    img,
    border_frac=0.08,
    k_clusters=3,
    color_dist_thr=0.25,
    grad_thr=0.3,
    min_fg_area_frac=0.1,
    max_fg_area_frac=0.99,
    max_border_frac=0.15,  # max border thickness as fraction of min(H, W)
):
    """
    Robust border detection with extra geometric prior:
    - Border must be near image edges (within max_border_frac * min(H, W)).

    Returns:
        (top, bottom, left, right)  [bottom/right EXCLUSIVE]
        Falls back to full image if detection looks suspicious.
    """
    h, w = img.shape[:2]

    # Convert to Lab for perceptual color distance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # --- 1. Collect border pixels (outer frame) ---
    bf = border_frac
    t = int(max(1, round(bf * h)))
    l = int(max(1, round(bf * w)))

    border_mask = np.zeros((h, w), np.uint8)
    border_mask[:t, :] = 1
    border_mask[-t:, :] = 1
    border_mask[:, :l] = 1
    border_mask[:, -l:] = 1

    border_pixels = lab[border_mask == 1].reshape(-1, 3)
    if border_pixels.shape[0] < k_clusters:
        # too few border pixels, fallback
        return 0, h, 0, w

    # --- 2. K-means on border pixels to find dominant border color ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    max_samples = 5000
    if border_pixels.shape[0] > max_samples:
        idx = np.random.choice(border_pixels.shape[0], max_samples, replace=False)
        border_pixels_sample = border_pixels[idx]
    else:
        border_pixels_sample = border_pixels

    border_pixels_sample = border_pixels_sample.astype(np.float32)

    _, labels, centers = cv2.kmeans(
        border_pixels_sample,
        k_clusters,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    _, counts = np.unique(labels.flatten(), return_counts=True)
    bg_cluster_idx = int(np.argmax(counts))
    bg_center = centers[bg_cluster_idx]  # (3,)

    # --- 3. For all pixels: color distance + gradient ---
    diff = lab - bg_center.reshape(1, 1, 3)
    color_dist = np.linalg.norm(diff, axis=2)
    color_dist_norm = color_dist / (color_dist.max() + 1e-6)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)

    # background candidate
    bg_cand = (color_dist_norm < color_dist_thr) & (grad_norm < grad_thr)
    bg_cand = bg_cand.astype(np.uint8)

    # --- 4. Keep only components that touch the image border ---
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(bg_cand, connectivity=4)

    if num_labels <= 1:
        return 0, h, 0, w

    edge_mask = np.zeros_like(bg_cand, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    bg_final = np.zeros_like(bg_cand, dtype=np.uint8)
    best_area = 0

    for lbl in range(1, num_labels):
        comp_mask = labels_cc == lbl
        if not np.any(comp_mask & edge_mask):
            continue
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            bg_final[:] = 0
            bg_final[comp_mask] = 1

    if best_area == 0:
        return 0, h, 0, w

    # --- 4.5 NEW: geometric clamp – limit border thickness ---

    # Use np.indices so all arrays are shape (h, w)
    yy, xx = np.indices((h, w))
    dist_to_top = yy
    dist_to_bottom = h - 1 - yy
    dist_to_left = xx
    dist_to_right = w - 1 - xx

    # distance to nearest edge for each pixel
    dist_to_edge = np.minimum.reduce(
        [dist_to_top, dist_to_bottom, dist_to_left, dist_to_right]
    )

    max_border_px = int(max_border_frac * min(h, w))
    # keep only bg pixels within this distance from edge
    bg_final = bg_final & (dist_to_edge <= max_border_px)

    if not np.any(bg_final):
        # after clamping nothing remains → fallback
        return 0, h, 0, w

    # --- 5. Foreground mask & bounding box ---
    fg_mask = (bg_final == 0).astype(np.uint8)

    num_fg, labels_fg, stats_fg, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=4)
    if num_fg <= 1:
        return 0, h, 0, w

    cleaned_fg = np.zeros_like(fg_mask)
    for lbl in range(1, num_fg):
        area = stats_fg[lbl, cv2.CC_STAT_AREA]
        if area < 50:
            continue
        cleaned_fg[labels_fg == lbl] = 1

    ys_fg, xs_fg = np.where(cleaned_fg == 1)
    if len(ys_fg) == 0 or len(xs_fg) == 0:
        return 0, h, 0, w

    top = int(np.min(ys_fg))
    bottom = int(np.max(ys_fg) + 1)
    left = int(np.min(xs_fg))
    right = int(np.max(xs_fg) + 1)

    fg_area = (bottom - top) * (right - left)
    full_area = h * w
    frac = fg_area / float(full_area)

    if frac < min_fg_area_frac or frac > max_fg_area_frac:
        return 0, h, 0, w

    return top, bottom, left, right

# -------------------------------
# 2. Visualization per image
# -------------------------------
def visualize_border_detection(
    image_path,
    out_dir,
    border_frac=0.08,
    k_clusters=3,
    color_dist_thr=0.25,
    grad_thr=0.3,
    min_fg_area_frac=0.1,
    max_fg_area_frac=0.99,
    max_border_frac=0.15,
    save_crop=False,
):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return

    h, w = img.shape[:2]

    top, bottom, left, right = detect_border_bbox_robust(
        img,
        border_frac=border_frac,
        k_clusters=k_clusters,
        color_dist_thr=color_dist_thr,
        grad_thr=grad_thr,
        min_fg_area_frac=min_fg_area_frac,
        max_fg_area_frac=max_fg_area_frac,
        max_border_frac=max_border_frac,
    )

    vis = img.copy()

    # clamp & validate
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h))
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w))

    if bottom > top and right > left:
        cv2.rectangle(vis, (left, top), (right - 1, bottom - 1), (0, 255, 255), 3)
        cv2.putText(
            vis,
            "robust_bbox",
            (left + 5, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    base = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(out_dir, f"{base}_border_robust_vis.png")
    cv2.imwrite(vis_path, vis)

    if save_crop and bottom > top and right > left:
        crop = img[top:bottom, left:right].copy()
        crop_path = os.path.join(out_dir, f"{base}_border_robust_crop.png")
        cv2.imwrite(crop_path, crop)


# -------------------------------
# 3. Folder traversal helpers
# -------------------------------
def list_images(input_dir, exts, recursive=False):
    exts = {e.lower() for e in exts}
    paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            full = os.path.join(input_dir, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in exts:
                paths.append(full)
    paths.sort()
    return paths


# -------------------------------
# 4. CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Robust border detection + visualization using Lab color clustering + gradient + connectivity."
    )
    p.add_argument("input_dir", type=str, help="Input folder with images.")
    p.add_argument(
        "--output-dir",
        type=str,
        default="border_robust_vis",
        help="Folder to save visualizations (and optional crops).",
    )
    p.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"],
        help="Image extensions to include.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input_dir.",
    )
    # Robust detector params
    p.add_argument(
        "--border-frac",
        type=float,
        default=0.08,
        help="Fraction of image from each side treated as border strip (default: 0.08).",
    )
    p.add_argument(
        "--k-clusters",
        type=int,
        default=3,
        help="Number of K-means clusters for border colors (default: 3).",
    )
    p.add_argument(
        "--color-dist-thr",
        type=float,
        default=0.25,
        help="Threshold on normalized Lab distance to border color (default: 0.25).",
    )
    p.add_argument(
        "--grad-thr",
        type=float,
        default=0.3,
        help="Threshold on normalized gradient magnitude for background (default: 0.3).",
    )
    p.add_argument(
        "--min-fg-area-frac",
        type=float,
        default=0.1,
        help="Minimum fraction of image area allowed for foreground box (default: 0.1).",
    )
    p.add_argument(
        "--max-fg-area-frac",
        type=float,
        default=0.99,
        help="Maximum fraction of image area allowed for foreground box (default: 0.99).",
    )
    p.add_argument(
        "--max-border-frac",
        type=float,
        default=0.15,
        help="Max border thickness as fraction of min(H,W) (default: 0.15).",
    )
    p.add_argument(
        "--save-crop",
        action="store_true",
        help="Also save cropped image using detected bbox (default: only visualization).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    img_paths = list_images(input_dir, args.ext, recursive=args.recursive)
    if not img_paths:
        print(f"No images found in {input_dir} with extensions: {args.ext}")
        return

    print(f"Found {len(img_paths)} images to process.")

    iterable = img_paths
    if tqdm is not None:
        iterable = tqdm(img_paths, desc="Detecting borders")

    for img_path in iterable:
        visualize_border_detection(
            img_path,
            output_dir,
            border_frac=args.border_frac,
            k_clusters=args.k_clusters,
            color_dist_thr=args.color_dist_thr,
            grad_thr=args.grad_thr,
            min_fg_area_frac=args.min_fg_area_frac,
            max_fg_area_frac=args.max_fg_area_frac,
            save_crop=args.save_crop,
            max_border_frac=args.max_border_frac,
        )

    print(f"Done. Results saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
