"""Experiment 3: Fixed-Box L-BFGS-B Compression + Binary Search & Structured Seeds."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from squares.geometry import compute_bounding_box, find_overlaps_float64


@dataclass
class Exp3Config:
    """Hyperparameters for Fixed-Box L-BFGS-B Compression."""

    binary_search_steps: int = 10
    restarts_initial: int = 6
    restarts_refine: int = 2
    lbfgs_maxiter: int = 220
    safety_margin: float = 1e-8
    seed: int = 2026


@dataclass
class Exp3Result:
    """Result of running exp3 on a single puzzle n."""

    n: int
    squares: np.ndarray
    side_length: float
    score: float
    baseline_side: float
    baseline_score: float
    valid: bool
    elapsed_sec: float


def fixed_box_energy_and_grad(
    W: np.ndarray,
    L: float,
    margin: float = 1e-7,
) -> tuple[float, np.ndarray, float, float]:
    """Compute boundary + SAT overlap penalty energy and analytical gradient in [0, L]^2.

    Returns:
        (energy, grad_W, max_boundary_viol, max_penetration)
    """
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

    # 2. Pairwise SAT overlap violations
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    d_theta = theta[:, None] - theta[None, :]

    c_dt = np.cos(d_theta)
    s_dt = np.sin(d_theta)
    abs_cdt = np.sqrt(c_dt * c_dt + eps_abs)
    abs_sdt = np.sqrt(s_dt * s_dt + eps_abs)
    r_sum = 0.5 + 0.5 * (abs_cdt + abs_sdt)

    p0 = dx * cos_t[:, None] + dy * sin_t[:, None]
    p1 = -dx * sin_t[:, None] + dy * cos_t[:, None]
    p2 = dx * cos_t[None, :] + dy * sin_t[None, :]
    p3 = -dx * sin_t[None, :] + dy * cos_t[None, :]

    projs = np.stack([p0, p1, p2, p3], axis=-1)
    abs_projs = np.abs(projs)
    k_star = np.argmax(abs_projs, axis=-1)
    proj_max = np.take_along_axis(abs_projs, k_star[..., None], axis=-1)[..., 0]
    pen = r_sum - proj_max

    iu, ju = np.triu_indices(n, k=1)
    pen_pairs = pen[iu, ju]
    max_pen = float(np.max(pen_pairs))

    viol = pen_pairs + margin
    mask = viol > 0.0
    if not np.any(mask):
        return energy, grad, max_bviol, max_pen

    iu_m = iu[mask]
    ju_m = ju[mask]
    viol_m = viol[mask]
    energy += float(np.sum(viol_m * viol_m))
    d_e_d_pen = 2.0 * viol_m

    dr_ddt = 0.5 * (
        -(c_dt[iu_m, ju_m] / abs_cdt[iu_m, ju_m]) * s_dt[iu_m, ju_m]
        + (s_dt[iu_m, ju_m] / abs_sdt[iu_m, ju_m]) * c_dt[iu_m, ju_m]
    )

    ks = k_star[iu_m, ju_m]
    p_sel = projs[iu_m, ju_m, ks]
    sgn = np.where(p_sel >= 0.0, 1.0, -1.0)

    m_cnt = len(iu_m)
    ax_x = np.empty(m_cnt, dtype=np.float64)
    ax_y = np.empty(m_cnt, dtype=np.float64)
    dax_dti_x = np.zeros(m_cnt, dtype=np.float64)
    dax_dti_y = np.zeros(m_cnt, dtype=np.float64)
    dax_dtj_x = np.zeros(m_cnt, dtype=np.float64)
    dax_dtj_y = np.zeros(m_cnt, dtype=np.float64)

    m0 = ks == 0
    m1 = ks == 1
    m2 = ks == 2
    m3 = ks == 3

    ax_x[m0], ax_y[m0] = cos_t[iu_m[m0]], sin_t[iu_m[m0]]
    dax_dti_x[m0], dax_dti_y[m0] = -sin_t[iu_m[m0]], cos_t[iu_m[m0]]

    ax_x[m1], ax_y[m1] = -sin_t[iu_m[m1]], cos_t[iu_m[m1]]
    dax_dti_x[m1], dax_dti_y[m1] = -cos_t[iu_m[m1]], -sin_t[iu_m[m1]]

    ax_x[m2], ax_y[m2] = cos_t[ju_m[m2]], sin_t[ju_m[m2]]
    dax_dtj_x[m2], dax_dtj_y[m2] = -sin_t[ju_m[m2]], cos_t[ju_m[m2]]

    ax_x[m3], ax_y[m3] = -sin_t[ju_m[m3]], cos_t[ju_m[m3]]
    dax_dtj_x[m3], dax_dtj_y[m3] = -cos_t[ju_m[m3]], -sin_t[ju_m[m3]]

    dx_m = dx[iu_m, ju_m]
    dy_m = dy[iu_m, ju_m]

    gx = d_e_d_pen * (-sgn * ax_x)
    gy = d_e_d_pen * (-sgn * ax_y)
    gti = d_e_d_pen * (dr_ddt - sgn * (dx_m * dax_dti_x + dy_m * dax_dti_y))
    gtj = d_e_d_pen * (-dr_ddt - sgn * (dx_m * dax_dtj_x + dy_m * dax_dtj_y))

    np.add.at(grad[:, 0], ju_m, gx)
    np.add.at(grad[:, 0], iu_m, -gx)
    np.add.at(grad[:, 1], ju_m, gy)
    np.add.at(grad[:, 1], iu_m, -gy)
    np.add.at(grad[:, 2], iu_m, gti)
    np.add.at(grad[:, 2], ju_m, gtj)

    return energy, grad, max_bviol, max_pen


def try_pack_in_box(
    init_W: np.ndarray,
    L: float,
    maxiter: int = 220,
    safety_margin: float = 1e-8,
) -> tuple[bool, np.ndarray, float]:
    """Attempt to pack squares inside [0, L]^2 using L-BFGS-B on fixed_box_energy_and_grad."""
    shape = init_W.shape

    def obj(flat: np.ndarray) -> tuple[float, np.ndarray]:
        e, g, _, _ = fixed_box_energy_and_grad(flat.reshape(shape), L, margin=1e-7)
        return e, g.ravel()

    res = minimize(
        obj,
        init_W.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
    )
    W = res.x.reshape(shape)

    # Check exact overlaps; if tiny micro-overlap (< 1e-6), try tiny uniform inflation
    overlaps, max_pen = find_overlaps_float64(W, tol=1e-12)
    if overlaps and max_pen < 1e-5:
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


def _generate_seed(
    n: int,
    L: float,
    baseline_W: np.ndarray,
    seed_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate diverse initializations inside [0, L]^2."""
    W = np.empty((n, 3), dtype=np.float64)
    if seed_idx == 0:
        # Compressed grid with small random rotations
        base_L = compute_bounding_box(baseline_W).side_length
        scale = min(1.0, (L - 0.1) / max(base_L, 1e-6))
        W[:, :2] = baseline_W[:, :2] * scale + 0.05
        W[:, 2] = rng.uniform(-0.35, 0.35, size=n)
    elif seed_idx == 1:
        # 45-degree rotated interior + random outer
        W[:, 0] = rng.uniform(0.6, max(0.7, L - 0.6), size=n)
        W[:, 1] = rng.uniform(0.6, max(0.7, L - 0.6), size=n)
        W[:, 2] = rng.choice([0.0, math.pi / 4.0], size=n) + rng.normal(0.0, 0.1, size=n)
    else:
        # Uniform random centers & angles inside [0.5, L - 0.5]^2
        lo = min(0.5, L * 0.25)
        hi = max(lo + 0.01, L - lo)
        W[:, 0] = rng.uniform(lo, hi, size=n)
        W[:, 1] = rng.uniform(lo, hi, size=n)
        W[:, 2] = rng.uniform(0.0, math.pi / 2.0, size=n)
    return W


def solve_puzzle_exp3(
    n: int,
    baseline_squares: np.ndarray,
    cfg: Exp3Config | None = None,
) -> Exp3Result:
    """Solve puzzle n via Fixed-Box L-BFGS-B Compression + Binary Search."""
    t0 = time.perf_counter()
    if cfg is None:
        cfg = Exp3Config()

    rng = np.random.default_rng(cfg.seed + n * 1009)
    baseline_L = compute_bounding_box(baseline_squares).side_length
    baseline_score = baseline_L / math.sqrt(n) - 1.0

    best_W = baseline_squares.copy()
    best_L = baseline_L

    # Trivial optimal cases (perfect squares)
    root = math.isqrt(n)
    if root * root == n:
        return Exp3Result(
            n=n,
            squares=best_W,
            side_length=best_L,
            score=baseline_score,
            baseline_side=baseline_L,
            baseline_score=baseline_score,
            valid=True,
            elapsed_sec=time.perf_counter() - t0,
        )

    low = math.sqrt(n)
    high = baseline_L

    # First probe an aggressive target (e.g. midpoint or baseline - 0.35)
    for step in range(cfg.binary_search_steps):
        if high - low < 1e-4:
            break
        target_L = 0.5 * (low + high) if step > 0 else max(low + 0.01, baseline_L - 0.35)
        if target_L >= best_L:
            target_L = 0.5 * (low + best_L)

        found = False
        # First try warm-starting from current best_W scaled down to target_L
        if best_L < baseline_L:
            warm = best_W.copy()
            warm[:, :2] *= (target_L / best_L)
            ok, w_cand, l_cand = try_pack_in_box(
                warm, target_L, maxiter=cfg.lbfgs_maxiter, safety_margin=cfg.safety_margin
            )
            if ok and l_cand < best_L:
                best_W, best_L = w_cand, l_cand
                high = min(high, best_L)
                found = True

        if not found:
            n_restarts = cfg.restarts_initial if best_L >= baseline_L - 1e-6 else cfg.restarts_refine
            for r in range(n_restarts):
                seed_W = _generate_seed(n, target_L, baseline_squares, r, rng)
                ok, w_cand, l_cand = try_pack_in_box(
                    seed_W, target_L, maxiter=cfg.lbfgs_maxiter, safety_margin=cfg.safety_margin
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

    return Exp3Result(
        n=n,
        squares=best_W,
        side_length=final_L,
        score=final_score,
        baseline_side=baseline_L,
        baseline_score=baseline_score,
        valid=valid,
        elapsed_sec=time.perf_counter() - t0,
    )
