"""Geometry primitives and exact/fast SAT overlap checks for unit squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box and enclosing square side length."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    side_length: float


def square_corners(x: float, y: float, angle: float) -> np.ndarray:
    """Return the 4 vertices (shape (4, 2)) of a unit square at (x, y) rotated by angle (rad)."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    # Local corners in CCW order: (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)
    dx = np.array([-0.5, 0.5, 0.5, -0.5], dtype=np.float64)
    dy = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    vx = x + c * dx - s * dy
    vy = y + s * dx + c * dy
    return np.column_stack([vx, vy])


def compute_bounding_box(squares: np.ndarray) -> BoundingBox:
    """Compute the tight axis-aligned bounding box and square side length L_n.

    Args:
        squares: Array of shape (n, 3) with columns (x, y, angle).
    """
    if squares.ndim != 2 or squares.shape[1] != 3 or len(squares) == 0:
        raise ValueError(f"Expected non-empty (n, 3) array, got shape {squares.shape}")

    x = squares[:, 0]
    y = squares[:, 1]
    theta = squares[:, 2]
    half_ext = 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))

    xmin = float(np.min(x - half_ext))
    xmax = float(np.max(x + half_ext))
    ymin = float(np.min(y - half_ext))
    ymax = float(np.max(y + half_ext))
    side_length = max(xmax - xmin, ymax - ymin)
    return BoundingBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, side_length=side_length)


def sat_penetration_pair(sq_a: Sequence[float], sq_b: Sequence[float]) -> float:
    """Return signed SAT penetration depth between two unit squares.

    Positive means interior overlap; 0 means touching; negative means separated.
    """
    xa, ya, ta = float(sq_a[0]), float(sq_a[1]), float(sq_a[2])
    xb, yb, tb = float(sq_b[0]), float(sq_b[1]), float(sq_b[2])
    dx = xb - xa
    dy = yb - ya

    ca, sa = np.cos(ta), np.sin(ta)
    cb, sb = np.cos(tb), np.sin(tb)

    # Combined half-extent along any of the 4 edge-normal axes
    r_sum = 0.5 + 0.5 * (abs(np.cos(ta - tb)) + abs(np.sin(ta - tb)))

    proj_max = max(
        abs(dx * ca + dy * sa),
        abs(-dx * sa + dy * ca),
        abs(dx * cb + dy * sb),
        abs(-dx * sb + dy * cb),
    )
    return float(r_sum - proj_max)


def find_overlaps_float64(
    squares: np.ndarray,
    tol: float = 1e-12,
) -> tuple[list[tuple[int, int, float]], float]:
    """Find all overlapping pairs of unit squares using sweep-and-prune + SAT.

    Args:
        squares: Array of shape (n, 3) with columns (x, y, angle).
        tol: Penetration threshold above which a pair is considered overlapping.

    Returns:
        (overlaps, max_penetration) where overlaps is a list of (i, j, penetration)
        with i < j, and max_penetration is the maximum penetration across checked pairs
        (or -inf if n < 2).
    """
    n = len(squares)
    if n < 2:
        return [], float("-inf")

    x = squares[:, 0]
    y = squares[:, 1]
    theta = squares[:, 2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    half_ext = 0.5 * (np.abs(cos_t) + np.abs(sin_t))

    xmin = x - half_ext
    xmax = x + half_ext
    ymin = y - half_ext
    ymax = y + half_ext

    order = np.argsort(xmin, kind="mergesort")
    overlaps: list[tuple[int, int, float]] = []
    max_pen = float("-inf")

    for pos_i in range(n):
        i = int(order[pos_i])
        xi_max = xmax[i]
        yi_min = ymin[i]
        yi_max = ymax[i]
        xi, yi, ti = x[i], y[i], theta[i]
        ci, si = cos_t[i], sin_t[i]

        for pos_j in range(pos_i + 1, n):
            j = int(order[pos_j])
            if xmin[j] >= xi_max - tol:
                break
            if ymax[j] <= yi_min + tol or ymin[j] >= yi_max - tol:
                continue

            dx = x[j] - xi
            dy = y[j] - yi
            # Circumradius quick reject: sqrt(2) * 2 / 2 = sqrt(2)
            if dx * dx + dy * dy >= 2.0 + 1e-9:
                continue

            cj, sj = cos_t[j], sin_t[j]
            d_theta = ti - theta[j]
            r_sum = 0.5 + 0.5 * (abs(np.cos(d_theta)) + abs(np.sin(d_theta)))

            proj_max = max(
                abs(dx * ci + dy * si),
                abs(-dx * si + dy * ci),
                abs(dx * cj + dy * sj),
                abs(-dx * sj + dy * cj),
            )
            pen = float(r_sum - proj_max)
            if pen > max_pen:
                max_pen = pen
            if pen > tol:
                a, b = (i, j) if i < j else (j, i)
                overlaps.append((a, b, pen))

    return overlaps, max_pen


def compute_side_length_mpmath(
    squares_raw: Sequence[tuple[str | float, str | float, str | float]],
    dps: int = 60,
) -> mp.mpf:
    """Compute exact bounding square side length L_n using mpmath at `dps` digits."""
    with mp.workdps(dps):
        half = mp.mpf("0.5")
        xmin = mp.inf
        xmax = -mp.inf
        ymin = mp.inf
        ymax = -mp.inf

        for x_raw, y_raw, t_raw in squares_raw:
            x = mp.mpf(str(x_raw))
            y = mp.mpf(str(y_raw))
            t = mp.mpf(str(t_raw))
            h = half * (abs(mp.cos(t)) + abs(mp.sin(t)))
            if x - h < xmin:
                xmin = x - h
            if x + h > xmax:
                xmax = x + h
            if y - h < ymin:
                ymin = y - h
            if y + h > ymax:
                ymax = y + h

        return max(xmax - xmin, ymax - ymin)


def find_overlaps_mpmath(
    squares_raw: Sequence[tuple[str | float, str | float, str | float]],
    dps: int = 60,
    tol: str = "0",
) -> list[tuple[int, int, str]]:
    """Verify overlaps using mpmath at `dps` digits, using AABB sweep-and-prune first."""
    n = len(squares_raw)
    if n < 2:
        return []

    with mp.workdps(dps):
        half = mp.mpf("0.5")
        tol_mp = mp.mpf(tol)
        parsed = []
        for idx, (x_raw, y_raw, t_raw) in enumerate(squares_raw):
            x = mp.mpf(str(x_raw))
            y = mp.mpf(str(y_raw))
            t = mp.mpf(str(t_raw))
            c = mp.cos(t)
            s = mp.sin(t)
            h = half * (abs(c) + abs(s))
            parsed.append((idx, x, y, t, c, s, x - h, x + h, y - h, y + h))

        # Sort by xmin
        parsed.sort(key=lambda item: item[6])
        overlaps: list[tuple[int, int, str]] = []

        for p_i in range(n):
            i, xi, yi, ti, ci, si, _, xi_max, yi_min, yi_max = parsed[p_i]
            for p_j in range(p_i + 1, n):
                j, xj, yj, tj, cj, sj, xj_min, _, yj_min, yj_max = parsed[p_j]
                if xj_min >= xi_max - tol_mp:
                    break
                if yj_max <= yi_min + tol_mp or yj_min >= yi_max - tol_mp:
                    continue

                dx = xj - xi
                dy = yj - yi
                d_theta = ti - tj
                r_sum = half + half * (abs(mp.cos(d_theta)) + abs(mp.sin(d_theta)))
                proj_max = max(
                    abs(dx * ci + dy * si),
                    abs(-dx * si + dy * ci),
                    abs(dx * cj + dy * sj),
                    abs(-dx * sj + dy * cj),
                )
                pen = r_sum - proj_max
                if pen > tol_mp:
                    a, b = (i, j) if i < j else (j, i)
                    overlaps.append((a, b, mp.nstr(pen, 12)))

        return overlaps
