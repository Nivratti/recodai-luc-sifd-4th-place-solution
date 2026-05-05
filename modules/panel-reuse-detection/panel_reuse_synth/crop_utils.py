from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.w() * self.h()

    def to_list(self):
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]


def crop_img(img: np.ndarray, box: Box) -> np.ndarray:
    return img[box.y1:box.y2, box.x1:box.x2].copy()


def intersection_area(a: Box, b: Box) -> int:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih


def intersection_box(a: Box, b: Box) -> Optional[Box]:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return Box(ix1, iy1, ix2, iy2)


def sample_crop_box(
    W: int,
    H: int,
    rng: np.random.Generator,
    area_range: Tuple[float, float],
    aspect_ratio_range: Tuple[float, float],
    min_side_px: int,
    max_tries: int,
) -> Optional[Box]:
    if W < min_side_px or H < min_side_px:
        return None

    area_min, area_max = float(area_range[0]), float(area_range[1])
    ar_min, ar_max = float(aspect_ratio_range[0]), float(aspect_ratio_range[1])
    src_area = W * H

    for _ in range(int(max_tries)):
        target_area = rng.uniform(area_min, area_max) * src_area
        aspect = rng.uniform(ar_min, ar_max)

        w = int(round(np.sqrt(target_area * aspect)))
        h = int(round(np.sqrt(target_area / aspect)))

        if w < min_side_px or h < min_side_px:
            continue
        if w > W or h > H:
            continue

        x1 = int(rng.integers(0, W - w + 1))
        y1 = int(rng.integers(0, H - h + 1))
        return Box(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)

    return None
