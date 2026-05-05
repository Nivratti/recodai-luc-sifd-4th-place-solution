from __future__ import annotations
from typing import Iterator, Tuple, Union


def iter_tiles(
    h: int,
    w: int,
    *,
    tile: Union[int, Tuple[int, int]] = 1024,
    overlap: float = 0.2,  # 0..0.9
) -> Iterator[Tuple[int, int, int, int]]:
    """
    Yield (x1, y1, x2, y2) tile windows covering an image.
    overlap is fraction of tile size.
    """
    if isinstance(tile, int):
        th, tw = tile, tile
    else:
        th, tw = int(tile[0]), int(tile[1])

    if th <= 0 or tw <= 0:
        raise ValueError("tile must be positive")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    if h <= th and w <= tw:
        yield (0, 0, w, h)
        return

    sh = max(1, int(th * (1.0 - overlap)))
    sw = max(1, int(tw * (1.0 - overlap)))

    y = 0
    while True:
        y2 = min(h, y + th)
        y1 = max(0, y2 - th)
        x = 0
        while True:
            x2 = min(w, x + tw)
            x1 = max(0, x2 - tw)
            yield (x1, y1, x2, y2)

            if x2 >= w:
                break
            x += sw

        if y2 >= h:
            break
        y += sh
