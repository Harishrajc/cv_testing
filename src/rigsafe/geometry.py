"""Small pure geometry helpers (no OpenCV, easy to test)."""
from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def point_in_polygon(pt: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting: count how many polygon edges a ray to the right crosses. Odd = inside."""
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def normalize(pt: Point, width: int, height: int) -> Point:
    """Pixel -> 0..1 coordinates, so zones work at any resolution."""
    return pt[0] / width, pt[1] / height


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def sub_region(box: Box, top: float, bottom: float, pad_x: float = 0.0) -> Box:
    """Horizontal band of a box between fractions top..bottom of its height, widened by pad_x."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 - w * pad_x, y1 + h * top, x2 + w * pad_x, y1 + h * bottom


def overlap_of_smaller(a: Box, b: Box) -> float:
    """Intersection area / area of the smaller box. 1.0 = small box fully inside the big one."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / smaller if smaller > 0 else 0.0


def contains(box: Box, pt: Point) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
