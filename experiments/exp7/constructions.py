"""Experiment 7: Exact geometric packing constructions for Squares in Squares."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from squares.geometry import compute_bounding_box, find_overlaps_float64

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class PackingCandidate:
    """Metadata describing a constructive packing of N squares in side L."""

    family: str
    n_squares: int
    side_length: float
    params: tuple[Any, ...]


def tri_rect_count(M: int, R: int, C: int) -> int:
    """Count integer pairs (r, c) with 0 <= r < R, 0 <= c < C, r + c <= M in O(1)."""
    if M < 0 or R <= 0 or C <= 0:
        return 0
    if M >= R + C - 2:
        return R * C
    if R > C:
        R, C = C, R
    r_max = min(M, R - 1)
    r_full = min(r_max, M - C + 1)
    cnt = 0
    if r_full >= 0:
        cnt += (r_full + 1) * C
        r_start = r_full + 1
    else:
        r_start = 0
    if r_start <= r_max:
        k = r_max - r_start + 1
        cnt += k * (2 * M - r_start - r_max + 2) // 2
    return cnt


# ---------------------------------------------------------------------------
# Family 1: Off-Center Clipped 4-Corner 45-deg Octagon (with sliding-band mode)
# ---------------------------------------------------------------------------

def _count_clipped_45_core(L: float, u0: float, v0: float, w: int, d: int, free_slide: bool) -> int:
    """Count 45-deg squares inside [0, L]^2 and within the w x d band/rectangle."""
    base_u = u0 - 0.5 * (d - 1) * SQRT2
    total = 0
    for j in range(w):
        vj = v0 + (j - 0.5 * (w - 1)) * SQRT2
        lo_u = SQRT2 + abs(vj)
        hi_u = 2.0 * L - SQRT2 - abs(vj)
        if hi_u < lo_u - 1e-12:
            continue
        if free_slide:
            # Rows slide freely along diagonal up to [0, L]^2 wall
            total += int(math.floor((hi_u - lo_u + 1e-12) / SQRT2)) + 1
        else:
            i_lo = max(0, int(math.ceil((lo_u - base_u - 1e-12) / SQRT2)))
            i_hi = min(d - 1, int(math.floor((hi_u - base_u + 1e-12) / SQRT2)))
            if i_hi >= i_lo:
                total += i_hi - i_lo + 1
    return total


def _best_4_corner_splits(
    Lf: int, M_ul: int, M_lr: int, M_bl: int, M_tr: int
) -> tuple[int, tuple[int, int, int, int]]:
    """Find optimal disjoint wall splits (b_l, b_r, a_t, a_b) for 4 axis-aligned corners."""
    c_ul = max(0, min(Lf, M_ul + 1))
    c_lr = max(0, min(Lf, M_lr + 1))
    c_bl = max(0, min(Lf, M_bl + 1))
    c_tr = max(0, min(Lf, M_tr + 1))

    if c_ul + c_tr <= Lf and c_bl + c_lr <= Lf and c_bl + c_ul <= Lf and c_lr + c_tr <= Lf:
        tot = (
            tri_rect_count(M_ul, Lf, Lf)
            + tri_rect_count(M_lr, Lf, Lf)
            + tri_rect_count(M_bl, Lf, Lf)
            + tri_rect_count(M_tr, Lf, Lf)
        )
        return tot, (c_bl, c_tr, c_ul, c_bl)

    bl_lo, bl_hi = sorted((max(0, Lf - c_ul), min(Lf, c_bl)))
    br_lo, br_hi = sorted((max(0, Lf - c_lr), min(Lf, c_tr)))
    at_lo, at_hi = sorted((max(0, Lf - c_tr), min(Lf, c_ul)))
    ab_lo, ab_hi = sorted((max(0, Lf - c_lr), min(Lf, c_bl)))

    def _sample(lo: int, hi: int) -> list[int]:
        if hi - lo <= 12:
            return list(range(lo, hi + 1))
        step = max(1, (hi - lo) // 8)
        pts = set(range(lo, hi + 1, step))
        pts.add(hi)
        pts.add((lo + hi) // 2)
        return sorted(pts)

    bl_rng = _sample(bl_lo, bl_hi)
    br_rng = _sample(br_lo, br_hi)
    at_rng = _sample(at_lo, at_hi)
    ab_rng = _sample(ab_lo, ab_hi)

    best_val = -1
    best_split = (0, 0, 0, 0)
    for b_l in bl_rng:
        for b_r in br_rng:
            best_at_val = -1
            best_at = at_rng[0]
            for a_t in at_rng:
                v = tri_rect_count(M_ul, Lf - b_l, a_t) + tri_rect_count(M_tr, b_r, Lf - a_t)
                if v > best_at_val:
                    best_at_val = v
                    best_at = a_t
            best_ab_val = -1
            best_ab = ab_rng[0]
            for a_b in ab_rng:
                v = tri_rect_count(M_bl, b_l, a_b) + tri_rect_count(M_lr, Lf - b_r, Lf - a_b)
                if v > best_ab_val:
                    best_ab_val = v
                    best_ab = a_b
            if best_at_val + best_ab_val > best_val:
                best_val = best_at_val + best_ab_val
                best_split = (b_l, b_r, best_at, best_ab)
    return best_val, best_split


def build_octagon_45(
    L: float,
    u0: float,
    v0: float,
    w: int,
    d: int,
    free_slide: bool,
    splits: tuple[int, int, int, int],
    eps: float = 1e-10,
) -> np.ndarray:
    """Build exact coordinates for Off-Center Clipped 4-Corner 45-deg Octagon."""
    Lf = int(math.floor(L + 1e-12))
    b_l, b_r, a_t, a_b = splits
    M_ul = int(math.floor(L - 2.0 - v0 - w / SQRT2 + 1e-12))
    M_lr = int(math.floor(L - 2.0 + v0 - w / SQRT2 + 1e-12))
    if free_slide:
        M_bl = -1
        M_tr = -1
    else:
        M_bl = int(math.floor(u0 - 2.0 - d / SQRT2 + 1e-12))
        M_tr = int(math.floor(2.0 * L - 2.0 - u0 - d / SQRT2 + 1e-12))

    # Inflate L slightly by eps so boundary inequalities are strictly satisfied in 60-digit mpmath
    L_eff = L + 2.0 * eps
    u0_eff = u0 + 2.0 * eps
    sqs: list[tuple[float, float, float]] = []

    # 1. UL corner: r in [0, Lf - b_l), c in [0, a_t), r + c <= M_ul
    for r in range(min(Lf - b_l, M_ul + 1)):
        for c in range(min(a_t, M_ul - r + 1)):
            sqs.append((c + 0.5, L_eff - r - 0.5, 0.0))

    # 2. LR corner: r in [0, Lf - b_r), c in [0, Lf - a_b), r + c <= M_lr
    for r in range(min(Lf - b_r, M_lr + 1)):
        for c in range(min(Lf - a_b, M_lr - r + 1)):
            sqs.append((L_eff - c - 0.5, r + 0.5, 0.0))

    # 3. BL corner: r in [0, b_l), c in [0, a_b), r + c <= M_bl
    for r in range(min(b_l, M_bl + 1)):
        for c in range(min(a_b, M_bl - r + 1)):
            sqs.append((c + 0.5, r + 0.5, 0.0))

    # 4. TR corner: r in [0, b_r), c in [0, Lf - a_t), r + c <= M_tr
    for r in range(min(b_r, M_tr + 1)):
        for c in range(min(Lf - a_t, M_tr - r + 1)):
            sqs.append((L_eff - c - 0.5, L_eff - r - 0.5, 0.0))

    # 5. Rotated 45-deg core
    base_u = u0_eff - 0.5 * (d - 1) * SQRT2
    for j in range(w):
        vj = v0 + (j - 0.5 * (w - 1)) * SQRT2
        lo_u = SQRT2 + abs(vj)
        hi_u = 2.0 * L_eff - SQRT2 - abs(vj)
        if hi_u < lo_u - 1e-12:
            continue
        if free_slide:
            cnt = int(math.floor((hi_u - lo_u + 1e-12) / SQRT2)) + 1
            slack = (hi_u - lo_u) - (cnt - 1) * SQRT2
            u_start = lo_u + 0.5 * max(0.0, slack)
            for k in range(cnt):
                ui = u_start + k * SQRT2
                sqs.append((0.5 * (ui - vj), 0.5 * (ui + vj), math.pi / 4.0))
        else:
            i_lo = max(0, int(math.ceil((lo_u - base_u - 1e-12) / SQRT2)))
            i_hi = min(d - 1, int(math.floor((hi_u - base_u + 1e-12) / SQRT2)))
            for i in range(i_lo, i_hi + 1):
                ui = base_u + i * SQRT2
                sqs.append((0.5 * (ui - vj), 0.5 * (ui + vj), math.pi / 4.0))

    # Scale centers slightly around centroid by (1 + eps) so touching squares have strict
    # positive separation > 0 under 60-digit mpmath despite float64 pi/4 rounding (~1e-16).
    arr = np.array(sqs, dtype=np.float64)
    if len(arr) > 1 and eps > 0.0:
        cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
        scale = 1.0 + 10.0 * eps
        arr[:, 0] = cx + scale * (arr[:, 0] - cx)
        arr[:, 1] = cy + scale * (arr[:, 1] - cy)
    bbox = compute_bounding_box(arr)
    arr[:, 0] -= bbox.xmin
    arr[:, 1] -= bbox.ymin
    return arr


# ---------------------------------------------------------------------------
# Family 2: General-Angle Rotated w x d Block + 4 Axis-Aligned Corners
# ---------------------------------------------------------------------------

def _count_corner_general(A: float, B: float, RHS: float, R_max: int, C_max: int) -> int:
    if RHS < -1e-12 or R_max <= 0 or C_max <= 0:
        return 0
    cnt = 0
    r_lim = min(R_max, int(math.floor(RHS / A + 1e-12)) + 1)
    for r in range(r_lim):
        c_max = int(math.floor((RHS - A * r) / B + 1e-12)) + 1
        if c_max > 0:
            cnt += min(c_max, C_max)
    return cnt


def _count_clipped_rect_general(
    L: float, x0: float, y0: float, ct: float, st: float, w: int, d: int
) -> int:
    h = 0.5 * (ct + st)
    lo, hi = h - 1e-12, L - h + 1e-12
    if hi < lo:
        return 0
    base_u = -0.5 * (d - 1)
    total = 0
    for j in range(w):
        vj = j - 0.5 * (w - 1)
        u_min = max((lo - x0 + vj * st) / ct, (lo - y0 - vj * ct) / st)
        u_max = min((hi - x0 + vj * st) / ct, (hi - y0 - vj * ct) / st)
        if u_max >= u_min - 1e-12:
            i_lo = max(0, int(math.ceil(u_min - base_u - 1e-12)))
            i_hi = min(d - 1, int(math.floor(u_max - base_u + 1e-12)))
            if i_hi >= i_lo:
                total += i_hi - i_lo + 1
    return total


def build_octagon_general(
    L: float,
    x0: float,
    y0: float,
    theta: float,
    w: int,
    d: int,
    splits: tuple[int, int, int, int],
    eps: float = 1e-10,
) -> np.ndarray:
    """Build exact coordinates for General-Angle Clipped 4-Corner Rotated Block."""
    ct, st = math.cos(theta), math.sin(theta)
    Lf = int(math.floor(L + 1e-12))
    b_l, b_r, a_t, a_b = splits

    rhs_ul = (L - 1.0 - y0) * ct - (1.0 - x0) * st - 0.5 * w
    rhs_lr = (L - 1.0 - x0) * st - (1.0 - y0) * ct - 0.5 * w
    rhs_bl = (x0 - 1.0) * ct + (y0 - 1.0) * st - 0.5 * d
    rhs_tr = (L - 1.0 - x0) * ct + (L - 1.0 - y0) * st - 0.5 * d

    L_eff = L + 2.0 * eps
    x0_eff = x0 + eps
    y0_eff = y0 + eps
    sqs: list[tuple[float, float, float]] = []

    # UL: ct*r + st*c <= rhs_ul
    if rhs_ul >= -1e-12:
        for r in range(min(Lf - b_l, int(math.floor(rhs_ul / ct + 1e-12)) + 1)):
            c_lim = min(a_t, int(math.floor((rhs_ul - ct * r) / st + 1e-12)) + 1)
            for c in range(max(0, c_lim)):
                sqs.append((c + 0.5, L_eff - r - 0.5, 0.0))

    # LR: ct*r + st*c <= rhs_lr
    if rhs_lr >= -1e-12:
        for r in range(min(Lf - b_r, int(math.floor(rhs_lr / ct + 1e-12)) + 1)):
            c_lim = min(Lf - a_b, int(math.floor((rhs_lr - ct * r) / st + 1e-12)) + 1)
            for c in range(max(0, c_lim)):
                sqs.append((L_eff - c - 0.5, r + 0.5, 0.0))

    # BL: st*r + ct*c <= rhs_bl
    if rhs_bl >= -1e-12:
        for r in range(min(b_l, int(math.floor(rhs_bl / st + 1e-12)) + 1)):
            c_lim = min(a_b, int(math.floor((rhs_bl - st * r) / ct + 1e-12)) + 1)
            for c in range(max(0, c_lim)):
                sqs.append((c + 0.5, r + 0.5, 0.0))

    # TR: st*r + ct*c <= rhs_tr
    if rhs_tr >= -1e-12:
        for r in range(min(b_r, int(math.floor(rhs_tr / st + 1e-12)) + 1)):
            c_lim = min(Lf - a_t, int(math.floor((rhs_tr - st * r) / ct + 1e-12)) + 1)
            for c in range(max(0, c_lim)):
                sqs.append((L_eff - c - 0.5, L_eff - r - 0.5, 0.0))

    # Rotated w x d block clipped to [0, L]^2
    h = 0.5 * (ct + st)
    lo, hi = h - 1e-12, L - h + 1e-12
    base_u = -0.5 * (d - 1)
    for j in range(w):
        vj = j - 0.5 * (w - 1)
        u_min = max((lo - x0 + vj * st) / ct, (lo - y0 - vj * ct) / st)
        u_max = min((hi - x0 + vj * st) / ct, (hi - y0 - vj * ct) / st)
        if u_max >= u_min - 1e-12:
            i_lo = max(0, int(math.ceil(u_min - base_u - 1e-12)))
            i_hi = min(d - 1, int(math.floor(u_max - base_u + 1e-12)))
            for i in range(i_lo, i_hi + 1):
                ui = base_u + i
                x = x0_eff + ui * ct - vj * st
                y = y0_eff + ui * st + vj * ct
                sqs.append((x, y, theta))

    # Scale centers slightly around centroid by (1 + eps) so touching squares have strict
    # positive separation > 0 under 60-digit mpmath despite float64 pi/4 rounding (~1e-16).
    arr = np.array(sqs, dtype=np.float64)
    if len(arr) > 1 and eps > 0.0:
        cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
        scale = 1.0 + 10.0 * eps
        arr[:, 0] = cx + scale * (arr[:, 0] - cx)
        arr[:, 1] = cy + scale * (arr[:, 1] - cy)
    bbox = compute_bounding_box(arr)
    arr[:, 0] -= bbox.xmin
    arr[:, 1] -= bbox.ymin
    return arr


# ---------------------------------------------------------------------------
# Family 3: L-Shaped Tilted Strips + Axis-Aligned Bulk
# ---------------------------------------------------------------------------

def best_tilted_strip_params(W: float, H: float) -> tuple[int, int, float]:
    """Return (best_count, k_bar, theta) for a rectangular strip W x H."""
    best_n = int(math.floor(W + 1e-12)) * int(math.floor(H + 1e-12))
    best_k = 0
    best_t = 0.0
    if H < 1.0 or W < 1.0:
        return best_n, best_k, best_t

    k_min = max(2, int(math.floor(H)) + 1)
    for k in range(k_min, min(int(math.floor(H)) + 12, 36)):
        R = math.hypot(k, 1.0)
        if H > R:
            continue
        phi = math.atan2(1.0, k)
        t = math.asin(min(1.0, H / R)) - phi
        if t <= 1e-6 or t >= math.pi / 2.0 - 1e-6:
            continue
        st, ct = math.sin(t), math.cos(t)
        bar_w = k * ct + st
        if bar_w > W + 1e-12:
            continue
        dx = 1.0 / st
        n_bars = int(math.floor((W - bar_w + 1e-12) / dx)) + 1
        cnt = n_bars * k
        m = -1
        while True:
            j_min = max(0, int(math.ceil((-m * dx - 1e-12) / ct)))
            if j_min >= k:
                break
            cnt += k - j_min
            m -= 1
        m = n_bars
        while True:
            j_max_p1 = min(k, int(math.floor((W - st - m * dx + 1e-12) / ct)))
            if j_max_p1 <= 0:
                break
            cnt += j_max_p1
            m += 1
        if cnt > best_n:
            best_n = cnt
            best_k = k
            best_t = t
    return best_n, best_k, best_t


def _build_strip_squares(
    W: float, H: float, k: int, t: float, x_off: float, y_off: float, transpose: bool = False
) -> list[tuple[float, float, float]]:
    """Build exact unit squares inside [0, W] x [0, H] (or transposed)."""
    out: list[tuple[float, float, float]] = []
    if k == 0:
        for r in range(int(math.floor(H + 1e-12))):
            for c in range(int(math.floor(W + 1e-12))):
                lx, ly, la = c + 0.5, r + 0.5, 0.0
                if transpose:
                    out.append((x_off + ly, y_off + lx, 0.0))
                else:
                    out.append((x_off + lx, y_off + ly, la))
        return out

    st, ct = math.sin(t), math.cos(t)
    dx = 1.0 / st
    bar_w = k * ct + st
    n_bars = int(math.floor((W - bar_w + 1e-12) / dx)) + 1

    # Bar m has bottom-left corner of square j at (st + m*dx + j*ct, j*st)
    # Center of square j rotated CCW by t is bottom-left + 0.5*(ct - st, st + ct)
    cx_off = 0.5 * (ct - st)
    cy_off = 0.5 * (st + ct)

    def add_bar(m_idx: int, j_lo: int, j_hi: int) -> None:
        bx = st + m_idx * dx
        for j in range(j_lo, j_hi):
            lx = bx + j * ct + cx_off
            ly = j * st + cy_off
            if transpose:
                # Mirror (x, y) -> (y, x), angle -> pi/2 - t
                out.append((x_off + ly, y_off + lx, math.pi / 2.0 - t))
            else:
                out.append((x_off + lx, y_off + ly, t))

    for m_idx in range(n_bars):
        add_bar(m_idx, 0, k)

    m = -1
    while True:
        j_min = max(0, int(math.ceil((-m * dx - 1e-12) / ct)))
        if j_min >= k:
            break
        add_bar(m, j_min, k)
        m -= 1

    m = n_bars
    while True:
        j_max_p1 = min(k, int(math.floor((W - st - m * dx + 1e-12) / ct)))
        if j_max_p1 <= 0:
            break
        add_bar(m, 0, j_max_p1)
        m += 1

    return out


def build_l_shaped_tilted(
    L: float, A: int, B: int, k_top: int, t_top: float, k_right: int, t_right: float, eps: float = 1e-10
) -> np.ndarray:
    """Build A x B axis-aligned bulk + top tilted strip + right tilted strip."""
    sqs: list[tuple[float, float, float]] = []
    for r in range(B):
        for c in range(A):
            sqs.append((c + 0.5, r + 0.5, 0.0))
    # Top strip: [0, L] x [B + eps, L + eps]
    sqs.extend(_build_strip_squares(L, L - B, k_top, t_top, 0.0, B + eps, transpose=False))
    # Right strip: [A + eps, L + eps] x [0, B] (strip of length B, height L - A, transposed!)
    sqs.extend(_build_strip_squares(float(B), L - A, k_right, t_right, A + eps, 0.0, transpose=True))
    # Scale centers slightly around centroid by (1 + eps) so touching squares have strict
    # positive separation > 0 under 60-digit mpmath despite float64 pi/4 rounding (~1e-16).
    arr = np.array(sqs, dtype=np.float64)
    if len(arr) > 1 and eps > 0.0:
        cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
        scale = 1.0 + 10.0 * eps
        arr[:, 0] = cx + scale * (arr[:, 0] - cx)
        arr[:, 1] = cy + scale * (arr[:, 1] - cy)
    bbox = compute_bounding_box(arr)
    arr[:, 0] -= bbox.xmin
    arr[:, 1] -= bbox.ymin
    return arr


# ---------------------------------------------------------------------------
# Family 4: L-Frame Composition (add p axis-aligned rows & cols to any packing)
# ---------------------------------------------------------------------------

def apply_l_frame(inner_squares: np.ndarray, p: int, eps: float = 1e-10) -> np.ndarray:
    """Wrap an existing valid packing of side L0 with an axis-aligned L-frame of thickness p."""
    if p <= 0:
        return inner_squares.copy()
    bbox = compute_bounding_box(inner_squares)
    L0 = bbox.side_length
    Lf0 = int(math.floor(L0 + 1e-12))

    # Shift inner packing to [p + eps, p + eps + L0] x [p + eps, p + eps + L0]
    inner = inner_squares.copy()
    inner[:, 0] = inner[:, 0] - bbox.xmin + p + eps
    inner[:, 1] = inner[:, 1] - bbox.ymin + p + eps

    frame: list[tuple[float, float, float]] = []
    # Bottom band: width p + Lf0, height p
    for r in range(p):
        for c in range(p + Lf0):
            frame.append((c + 0.5, r + 0.5, 0.0))
    # Left band above bottom band: width p, height Lf0
    for r in range(Lf0):
        for c in range(p):
            frame.append((c + 0.5, p + r + 0.5, 0.0))

    return np.vstack([np.array(frame, dtype=np.float64), inner])


def trim_squares_tight(squares: np.ndarray, target_n: int) -> np.ndarray:
    """Drop squares greedily from the outermost boundary to get target_n squares with minimal L."""
    W = squares.copy()
    if len(W) <= target_n:
        return W

    # Precompute half-extents once
    h = 0.5 * (np.abs(np.cos(W[:, 2])) + np.abs(np.sin(W[:, 2])))
    alive = np.ones(len(W), dtype=bool)
    remaining = len(W)

    while remaining > target_n:
        idx_alive = np.flatnonzero(alive)
        xs = W[idx_alive, 0]
        ys = W[idx_alive, 1]
        hs = h[idx_alive]
        x_lo = xs - hs
        x_hi = xs + hs
        y_lo = ys - hs
        y_hi = ys + hs

        width = float(np.max(x_hi) - np.min(x_lo))
        height = float(np.max(y_hi) - np.min(y_lo))

        # Remove a square touching the widest extreme edge
        if width >= height:
            pos = int(np.argmax(x_hi))
        else:
            pos = int(np.argmax(y_hi))
        alive[idx_alive[pos]] = False
        remaining -= 1

    out = W[alive]
    bbox = compute_bounding_box(out)
    out[:, 0] -= bbox.xmin
    out[:, 1] -= bbox.ymin
    return out


def build_tilted_l_frame(
    inner_squares: np.ndarray,
    L: float,
    k_top: int,
    t_top: float,
    k_right: int,
    t_right: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """Wrap an inner packing of side L0 inside [0, L]^2 with tilted top & right strips."""
    bbox = compute_bounding_box(inner_squares)
    L0 = bbox.side_length
    inner = inner_squares.copy()
    inner[:, 0] -= bbox.xmin
    inner[:, 1] -= bbox.ymin

    top_sqs = _build_strip_squares(L, L - L0, k_top, t_top, 0.0, L0 + eps, transpose=False)
    right_sqs = _build_strip_squares(L0, L - L0, k_right, t_right, L0 + eps, 0.0, transpose=True)

    parts = [inner]
    if top_sqs:
        parts.append(np.array(top_sqs, dtype=np.float64))
    if right_sqs:
        parts.append(np.array(right_sqs, dtype=np.float64))
    arr = np.vstack(parts)

    if len(arr) > 1 and eps > 0.0:
        cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
        scale = 1.0 + 10.0 * eps
        arr[:, 0] = cx + scale * (arr[:, 0] - cx)
        arr[:, 1] = cy + scale * (arr[:, 1] - cy)
    bb = compute_bounding_box(arr)
    arr[:, 0] -= bb.xmin
    arr[:, 1] -= bb.ymin
    return arr


def build_quad_composite(
    bl_squares: np.ndarray,
    tr_squares: np.ndarray,
    L: float,
    L1: float,
    k_tl: int,
    t_tl: float,
    k_br: int,
    t_br: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """2x2 composite: BL [0, L1]^2 + TR [L1, L]^2 + TL & BR tilted strips."""
    bb1 = compute_bounding_box(bl_squares)
    bl = bl_squares.copy()
    bl[:, 0] -= bb1.xmin
    bl[:, 1] -= bb1.ymin

    bb2 = compute_bounding_box(tr_squares)
    tr = tr_squares.copy()
    tr[:, 0] = tr[:, 0] - bb2.xmin + L1 + eps
    tr[:, 1] = tr[:, 1] - bb2.ymin + L1 + eps

    L2 = L - L1
    # Top-left: [0, L1] x [L1 + eps, L + eps]
    tl_sqs = _build_strip_squares(L1, L2, k_tl, t_tl, 0.0, L1 + eps, transpose=False)
    # Bottom-right: [L1 + eps, L + eps] x [0, L1] (length L1, height L2, transposed)
    br_sqs = _build_strip_squares(L1, L2, k_br, t_br, L1 + eps, 0.0, transpose=True)

    parts = [bl, tr]
    if tl_sqs:
        parts.append(np.array(tl_sqs, dtype=np.float64))
    if br_sqs:
        parts.append(np.array(br_sqs, dtype=np.float64))
    arr = np.vstack(parts)

    if len(arr) > 1 and eps > 0.0:
        cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
        scale = 1.0 + 10.0 * eps
        arr[:, 0] = cx + scale * (arr[:, 0] - cx)
        arr[:, 1] = cy + scale * (arr[:, 1] - cy)
    bb = compute_bounding_box(arr)
    arr[:, 0] -= bb.xmin
    arr[:, 1] -= bb.ymin
    return arr
