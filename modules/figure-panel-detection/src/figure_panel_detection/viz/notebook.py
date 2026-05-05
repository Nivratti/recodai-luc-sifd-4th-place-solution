from pathlib import Path
import random
import matplotlib.pyplot as plt
import cv2


def show_random_visualized_one_by_one(vis_dir, n=5, seed=None, max_side=1280, show_path=True):
    """
    Display random images from a "visualized" directory one-by-one in a notebook.

    This utility is intended for Kaggle/Jupyter notebooks where you have already
    generated and saved visualized YOLO prediction images (e.g., with bounding boxes).
    It recursively scans the given directory for image files, randomly samples `n`
    images, and displays each image as a separate matplotlib output.

    Parameters
    ----------
    vis_dir : str or pathlib.Path
        Path to the directory containing visualized images. The directory is searched
        recursively using ``Path.rglob("*")`` to find image files.
    n : int, default=5
        Number of random images to display. If ``n`` is larger than the number of
        available images, it will be clipped to the available count.
    seed : int or None, default=None
        Random seed for reproducible sampling. If None, sampling is non-deterministic.
    max_side : int, default=1400
        Maximum size (in pixels) for the largest image dimension (height or width)
        when displaying. Images larger than this are downscaled while preserving
        aspect ratio to speed up rendering.
    show_path : bool, default=True
        If True, displays the full file path as the matplotlib title for each image.

    Returns
    -------
    None
        This function displays images inline (side effect) and returns nothing.

    Raises
    ------
    FileNotFoundError
        If ``vis_dir`` does not exist or if no supported image files are found under it.

    Notes
    -----
    - Supported extensions: ``.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp``.
    - Images are read with OpenCV (BGR) and converted to RGB before displaying.
    - Files that fail to load (``cv2.imread`` returns None) are silently skipped.

    Examples
    --------
    Display 5 random images from a visualized output folder::

        show_random_visualized_one_by_one(
            "/kaggle/working/yolo_inference_output/model_4_class_640px/kept/visualized",
            n=5,
            seed=42
        )

    Display 10 images without showing file paths::

        show_random_visualized_one_by_one(
            "/kaggle/working/yolo_inference_output/model_4_class_640px/ignored/visualized",
            n=10,
            show_path=False
        )
    """
    vis_dir = Path(vis_dir)
    if not vis_dir.exists():
        raise FileNotFoundError(f"Visualized dir not found: {vis_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = [p for p in vis_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not imgs:
        raise FileNotFoundError(f"No images found under: {vis_dir}")

    rng = random.Random(seed)
    n = min(int(n), len(imgs))
    chosen = rng.sample(imgs, n)

    for p in chosen:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # optional downscale for faster display
        h, w = im.shape[:2]
        m = max(h, w)
        if m > max_side:
            s = max_side / m
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.axis("off")
        if show_path:
            plt.title(str(p), fontsize=9)
        plt.show()
