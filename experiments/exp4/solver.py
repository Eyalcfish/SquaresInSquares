"""Experiment 4: Scalable Fixed-Box L-BFGS-B Compression for n=1..1000."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from squares.geometry import compute_bounding_box, find_overlaps_float64


@dataclass
class Exp4Config:
    """Hyperparameters for scalable Fixed-Box L-BFGS-B Compression."""

    binary_search_steps: int = 6
    restarts_initial: int = 3
    restarts_refine: int = 1
    lbfgs_maxiter: int = 140
    safety_margin: float = 1e-8
    seed: int = 4040


@dataclass
class Exp4Result:
    """Result of running exp4 on a single puzzle n."""

    n: int
    squares: np.ndarray
    side_length: float
    score: float
    baseline_side: float
    baseline_score: float
    valid: bool
    elapsed_sec: float


def _candidate_pairs_spatial(x: np.ndarray, y: np.ndarray, cutoff: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate pairs (iu, ju) with |x_i - x_j| < cutoff and |y_i - y_j| < cutoff."""
    n = len(x)
    if n <= 80:
        return np.triu_indices(n, k=1)

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]

    iu_list: list[int] = []
    ju_list: list[int] = []

    for p_i in range(n):
        xi = xs[p_i]
        yi = ys[p_i]
        oi = int(order[p_i])
        for p_j in range(p_i + 1, n):
            if xs[p_j] - xi >= cutoff:
                break
            if abs(ys[p_j] - yi) < cutoff:
                oj = int(order[p_j])
                if oi < oj:
                    iu_list.append(oi)
                    ju_list.append(oj)
                else:
                    iu_list.append(oj)
                    ju_list.append(oi)

    return np.array(iu_list, dtype=np.int64), np.array(ju_list, dtype=np.int64)


def fixed_box_energy_and_grad_fast(
    W: np.ndarray,
    L: float,
    margin: float = 1e-7,
) -> tuple[float, np.ndarray, float, float]:
    """Compute boundary + spatially-pruned SAT overlap energy and gradient in [0, L]^2."""
    n = len(W)
    x = W[:, 0]
    y = W[:, 1]
    theta = W[:, 2]

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    eps_abs = 1e-9
    abs_cos = np.sqrt(cos_t * cos_t + eps_abs)
    abs_sin = np.sqrt(sin_t * sin_t + eps_abs)
    h = 0.5 * (abs_cos + abs_sin)
    dh_dtheta = 0.5 * (-(cos_t / abs_cos) * sin_t + (sin_t / abs_sin) * cos_t)

    grad = np.zeros_like(W)
    energy = 0.0

    # 1. Boundary violations against [0, L]^2
    vx_lo = np.maximum(0.0, h + margin - x)
    vx_hi = np.maximum(0.0, x + h + margin - L)
    vy_lo = np.maximum(0.0, h + margin - y)
    vy_hi = np.maximum(0.0, y + h + margin - L)

    max_bviol = float(max(np.max(vx_lo), np.max(vx_hi), np.max(vy_lo), np.max(vy_hi)))
    if max_bviol > 0.0:
        energy += float(np.sum(vx_lo * vx_lo + vx_hi * vx_hi + vy_lo * vy_lo + vy_hi * vy_hi))
        grad[:, 0] += 2.0 * (vx_hi - vx_lo)
        grad[:, 1] += 2.0 * (vy_hi - vy_lo)
        grad[:, 2] += 2.0 * (vx_lo + vx_hi + vy_lo + vy_hi) * dh_dtheta

    if n < 2:
        return energy, grad, max_bviol, float("-inf")

    # 2. Spatially-pruned SAT overlap violations
    iu, ju = _candidate_pairs_spatial(x, y, cutoff=1.45)
    if len(iu) == 0:
        return energy, grad, max_bviol, float("-inf")

    dx = x[ju] - x[iu]
    dy = y[ju] - y[iu]
    d_theta = theta[iu] - theta[ju]

    c_dt = np.cos(d_theta)
    s_dt = np.sin(d_theta)
    abs_cdt = np.sqrt(c_dt * c_dt + eps_abs)
    abs_sdt = np.sqrt(s_dt * s_dt + eps_abs)
    r_sum = 0.5 + 0.5 * (abs_cdt + abs_sdt)

    ci, si = cos_t[iu], sin_t[iu]
    cj, sj = cos_t[ju], sin_t[ju]

    p0 = dx * ci + dy * si
    p1 = -dx * si + dy * ci
    p2 = dx * cj + dy * sj
    p3 = -dx * sj + dy * cj

    projs = np.stack([p0, p1, p2, p3], axis=-1)
    abs_projs = np.abs(projs)
    ks = np.argmax(abs_projs, axis=-1)
    proj_max = np.take_along_axis(abs_projs, ks[..., None], axis=-1)[..., 0]
    pen = r_sum - proj_max
    max_pen = float(np.max(pen))

    viol = pen + margin
    mask = viol > 0.0
    if not np.any(mask):
        return energy, grad, max_bviol, max_pen

    iu_m = iu[mask]
    ju_m = ju[mask]
    viol_m = viol[mask]
    ks_m = ks[mask]
    p_sel = projs[mask, ks_m]
    sgn = np.where(p_sel >= 0.0, 1.0, -1.0)

    energy += float(np.sum(viol_m * viol_m))
    d_e_d_pen = 2.0 * viol_m

    dr_ddt = 0.5 * (
        -(c_dt[mask] / abs_cdt[mask]) * s_dt[mask]
        + (s_dt[mask] / abs_sdt[mask]) * c_dt[mask]
    )

    m_cnt = len(iu_m)
    ax_x = np.empty(m_cnt, dtype=np.float64)
    ax_y = np.empty(m_cnt, dtype=np.float64)
    dax_dti_x = np.zeros(m_cnt, dtype=np.float64)
    dax_dti_y = np.zeros(m_cnt, dtype=np.float64)
    dax_dtj_x = np.zeros(m_cnt, dtype=np.float64)
    dax_dtj_y = np.zeros(m_cnt, dtype=np.float64)

    m0 = ks_m == 0
    m1 = ks_m == 1
    m2 = ks_m == 2
    m3 = ks_m == 3

    ax_x[m0], ax_y[m0] = cos_t[iu_m[m0]], sin_t[iu_m[m0]]
    dax_dti_x[m0], dax_dti_y[m0] = -sin_t[iu_m[m0]], cos_t[iu_m[m0]]

    ax_x[m1], ax_y[m1] = -sin_t[iu_m[m1]], cos_t[iu_m[m1]]
    dax_dti_x[m1], dax_dti_y[m1] = -cos_t[iu_m[m1]], -sin_t[iu_m[m1]]

    ax_x[m2], ax_y[m2] = cos_t[ju_m[m2]], sin_t[ju_m[m2]]
    dax_dtj_x[m2], dax_dtj_y[m2] = -sin_t[ju_m[m2]], cos_t[ju_m[m2]]

    ax_x[m3], ax_y[m3] = -sin_t[ju_m[m3]], cos_t[ju_m[m3]]
    dax_dtj_x[m3], dax_dtj_y[m3] = -cos_t[ju_m[m3]], -sin_t[ju_m[m3]]

    dx_m = dx[mask]
    dy_m = dy[mask]

    gx = d_e_d_pen * (-sgn * ax_x)
    gy = d_e_d_pen * (-sgn * ax_y)
    gti = d_e_d_pen * (dr_ddt - sgn * (dx_m * dax_dti_x + dy_m * dax_dti_y))
    gtj = d_e_d_pen * (-dr_ddt - sgn * (dx_m * dax_dtj_x + dy_m * dax_dtj_y))

    np.add.at(grad[:, 0], ju_m, gx)
    np.add.at(grad[:, 0], iu_m, -gx)
    np.add.at(grad[:, 1], ju_m, gy)
    np.add.at(grad[:, 1], iu_m, -gy)
    np.add.at(grad[:, 2], iu_m, gti)
    np.add.at(grad[:, 2], iu_m, gtj)

    return energy, grad, max_bviol, max_pen


def try_pack_in_box_fast(
    init_W: np.ndarray,
    L: float,
    maxiter: int = 140,
    safety_margin: float = 1e-8,
) -> tuple[bool, np.ndarray, float]:
    """Pack squares inside [0, L]^2 using L-BFGS-B on spatially-pruned energy."""
    shape = init_W.shape

    def obj(flat: np.ndarray) -> tuple[float, np.ndarray]:
        e, g, _, _ = fixed_box_energy_and_grad_fast(flat.reshape(shape), L, margin=1e-7)
        return e, g.ravel()

    res = minimize(
        obj,
        init_W.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9},
    )
    W = res.x.reshape(shape)

    overlaps, max_pen = find_overlaps_float64(W, tol=1e-12)
    if overlaps and max_pen < 1e-4:
        scale = 1.0 + max_pen + safety_margin
        cx, cy = np.mean(W[:, 0]), np.mean(W[:, 1])
        W[:, 0] = cx + scale * (W[:, 0] - cx)
        W[:, 1] = cy + scale * (W[:, 1] - cy)
        overlaps, _ = find_overlaps_float64(W, tol=1e-12)

    if overlaps:
        return False, W, float("inf")

    bbox = compute_bounding_box(W)
    W[:, 0] -= bbox.xmin
    W[:, 1] -= bbox.ymin
    return True, W, bbox.side_length


def _generate_seed_fast(
    n: int,
    L: float,
    baseline_W: np.ndarray,
    seed_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate structured and random initializations inside [0, L]^2."""
    W = np.empty((n, 3), dtype=np.float64)
    if seed_idx == 0:
        # Compressed grid + tilted top/right excess rows
        base_L = compute_bounding_box(baseline_W).side_length
        scale = min(1.0, (L - 0.08) / max(base_L, 1e-6))
        W[:, :2] = baseline_W[:, :2] * scale + 0.04
        # Tilt the outer/last ~20% of squares toward 45 deg
        W[:, 2] = rng.normal(0.0, 0.08, size=n)
        k_tilt = max(1, n // 5)
        W[-k_tilt:, 2] = math.pi / 4.0 + rng.normal(0.0, 0.1, size=k_tilt)
    elif seed_idx == 1:
        # 45-degree rotated core lattice + axis-aligned outer
        W[:, 0] = rng.uniform(0.55, max(0.6, L - 0.55), size=n)
        W[:, 1] = rng.uniform(0.55, max(0.6, L - 0.55), size=n)
        W[:, 2] = rng.choice([0.0, math.pi / 4.0], size=n) + rng.normal(0.0, 0.08, size=n)
    else:
        lo = min(0.5, L * 0.25)
        hi = max(lo + 0.01, L - lo)
        W[:, 0] = rng.uniform(lo, hi, size=n)
        W[:, 1] = rng.uniform(lo, hi, size=n)
        W[:, 2] = rng.uniform(0.0, math.pi / 2.0, size=n)
    return W


def _config_for_n(n: int, base_cfg: Exp4Config) -> Exp4Config:
    if n <= 60:
        return Exp4Config(
            binary_search_steps=8,
            restarts_initial=4,
            restarts_refine=2,
            lbfgs_maxiter=180,
            safety_margin=base_cfg.safety_margin,
            seed=base_cfg.seed,
        )
    if n <= 250:
        return Exp4Config(
            binary_search_steps=6,
            restarts_initial=3,
            restarts_refine=1,
            lbfgs_maxiter=120,
            safety_margin=base_cfg.safety_margin,
            seed=base_cfg.seed,
        )
    return Exp4Config(
        binary_search_steps=5,
        restarts_initial=2,
        restarts_refine=1,
        lbfgs_maxiter=90,
        safety_margin=base_cfg.safety_margin,
        seed=base_cfg.seed,
    )


def solve_puzzle_exp4(
    n: int,
    init_squares: np.ndarray,
    baseline_side: float,
    cfg: Exp4Config | None = None,
) -> Exp4Result:
    """Solve puzzle n via scalable Fixed-Box L-BFGS-B Compression + Binary Search."""
    t0 = time.perf_counter()
    if cfg is None:
        cfg = Exp4Config()
    c = _config_for_n(n, cfg)

    rng = np.random.default_rng(c.seed + n * 1009)
    baseline_score = baseline_side / math.sqrt(n) - 1.0

    best_W = init_squares.copy()
    best_L = compute_bounding_box(best_W).side_length

    root = math.isqrt(n)
    if root * root == n:
        return Exp4Result(
            n=n,
            squares=best_W,
            side_length=best_L,
            score=best_L / math.sqrt(n) - 1.0,
            baseline_side=baseline_side,
            baseline_score=baseline_score,
            valid=True,
            elapsed_sec=time.perf_counter() - t0,
        )

    low = math.sqrt(n)
    high = best_L

    for step in range(c.binary_search_steps):
        if high - low < 1e-4:
            break
        # Probe ~0.25 below current best on first step, then binary search
        target_L = 0.5 * (low + high) if step > 0 else max(low + 0.01, best_L - 0.25)
        if target_L >= best_L:
            target_L = 0.5 * (low + best_L)

        found = False
        if best_L < baseline_side or step > 0:
            warm = best_W.copy()
            warm[:, :2] *= (target_L / best_L)
            ok, w_cand, l_cand = try_pack_in_box_fast(
                warm, target_L, maxiter=c.lbfgs_maxiter, safety_margin=c.safety_margin
            )
            if ok and l_cand < best_L:
                best_W, best_L = w_cand, l_cand
                high = min(high, best_L)
                found = True

        if not found:
            n_restarts = c.restarts_initial if best_L >= baseline_side - 1e-6 else c.restarts_refine
            for r in range(n_restarts):
                seed_W = _generate_seed_fast(n, target_L, init_squares, r, rng)
                ok, w_cand, l_cand = try_pack_in_box_fast(
                    seed_W, target_L, maxiter=c.lbfgs_maxiter, safety_margin=c.safety_margin
                )
                if ok and l_cand < best_L:
                    best_W, best_L = w_cand, l_cand
                    high = min(high, best_L)
                    found = True
                    break

        if not found:
            low = target_L

    overlaps, _ = find_overlaps_float64(best_W, tol=1e-12)
    valid = len(overlaps) == 0
    final_L = compute_bounding_box(best_W).side_length
    final_score = final_L / math.sqrt(n) - 1.0

    return Exp4Result(
        n=n,
        squares=best_W,
        side_length=final_L,
        score=final_score,
        baseline_side=baseline_side,
        baseline_score=baseline_score,
        valid=valid,
        elapsed_sec=time.perf_counter() - t0,
    )
