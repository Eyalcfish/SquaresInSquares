"""Experiment 1: Simulated Annealing + Gradient Descent Hybrid Solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from squares.geometry import compute_bounding_box, find_overlaps_float64


@dataclass
class Exp1Config:
    """Hyperparameters for the SA + GD hybrid optimizer."""

    sa_steps: int = 2500
    sa_t_start: float = 0.25
    sa_t_end: float = 0.002
    sa_pos_scale: float = 0.35
    sa_ang_scale: float = 0.45
    gd_steps: int = 600
    gd_lr: float = 0.02
    gd_overlap_weight: float = 50.0
    gd_margin: float = 1e-6
    smooth_tau: float = 0.015
    gd_triggers: int = 4
    seed: int = 42


@dataclass
class Exp1Result:
    """Result of running exp1 on a single puzzle n."""

    n: int
    squares: np.ndarray
    side_length: float
    score: float
    baseline_side: float
    baseline_score: float
    valid: bool
    sa_valid_found: int
    gd_runs: int
    elapsed_sec: float


def _softmax(v: np.ndarray, tau: float) -> np.ndarray:
    z = (v - np.max(v)) / tau
    e = np.exp(z)
    return e / np.sum(e)


def _compute_loss_and_grad(
    W: np.ndarray,
    overlap_weight: float,
    margin: float,
    tau: float,
) -> tuple[float, np.ndarray, float, float]:
    """Compute smooth bounding-square + overlap penalty loss and analytical gradient.

    Returns:
        (total_loss, grad_W, exact_side_length, max_penetration)
    """
    n = len(W)
    x = W[:, 0]
    y = W[:, 1]
    theta = W[:, 2]

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    eps_abs = 1e-8
    abs_cos = np.sqrt(cos_t * cos_t + eps_abs)
    abs_sin = np.sqrt(sin_t * sin_t + eps_abs)
    h = 0.5 * (abs_cos + abs_sin)
    dh_dtheta = 0.5 * (-(cos_t / abs_cos) * sin_t + (sin_t / abs_sin) * cos_t)

    # Smooth width and height via LogSumExp / softmax
    wx_max = _softmax(x + h, tau)
    wx_min = _softmax(-(x - h), tau)
    wy_max = _softmax(y + h, tau)
    wy_min = _softmax(-(y - h), tau)

    width_s = float(np.sum(wx_max * (x + h)) - np.sum(wx_min * (x - h)))
    height_s = float(np.sum(wy_max * (y + h)) - np.sum(wy_min * (y - h)))

    wh_w = _softmax(np.array([width_s, height_s]), tau)
    L_smooth = float(wh_w[0] * width_s + wh_w[1] * height_s)

    grad = np.zeros_like(W)
    grad[:, 0] = wh_w[0] * (wx_max - wx_min)
    grad[:, 1] = wh_w[1] * (wy_max - wy_min)
    grad[:, 2] = (
        wh_w[0] * (wx_max + wx_min) + wh_w[1] * (wy_max + wy_min)
    ) * dh_dtheta

    exact_width = float(np.max(x + h) - np.min(x - h))
    exact_height = float(np.max(y + h) - np.min(y - h))
    exact_L = max(exact_width, exact_height)

    # Vectorized pairwise SAT overlap penalty
    if n < 2:
        return L_smooth, grad, exact_L, float("-inf")

    dx = x[None, :] - x[:, None]  # (n, n): x_j - x_i
    dy = y[None, :] - y[:, None]  # (n, n): y_j - y_i
    d_theta = theta[:, None] - theta[None, :]

    c_dt = np.cos(d_theta)
    s_dt = np.sin(d_theta)
    abs_cdt = np.sqrt(c_dt * c_dt + eps_abs)
    abs_sdt = np.sqrt(s_dt * s_dt + eps_abs)
    r_sum = 0.5 + 0.5 * (abs_cdt + abs_sdt)

    # Projections onto 4 axes: u1_i, u2_i, u1_j, u2_j
    p0 = dx * cos_t[:, None] + dy * sin_t[:, None]
    p1 = -dx * sin_t[:, None] + dy * cos_t[:, None]
    p2 = dx * cos_t[None, :] + dy * sin_t[None, :]
    p3 = -dx * sin_t[None, :] + dy * cos_t[None, :]

    projs = np.stack([p0, p1, p2, p3], axis=-1)  # (n, n, 4)
    abs_projs = np.abs(projs)
    k_star = np.argmax(abs_projs, axis=-1)  # (n, n)
    proj_max = np.take_along_axis(abs_projs, k_star[..., None], axis=-1)[..., 0]
    pen = r_sum - proj_max  # (n, n)

    iu, ju = np.triu_indices(n, k=1)
    pen_pairs = pen[iu, ju]
    max_pen = float(np.max(pen_pairs))

    viol = pen_pairs + margin
    mask = viol > 0.0
    if not np.any(mask):
        return L_smooth, grad, exact_L, max_pen

    iu_m = iu[mask]
    ju_m = ju[mask]
    viol_m = viol[mask]
    ov_loss = float(overlap_weight * np.sum(viol_m * viol_m))
    d_loss_d_pen = 2.0 * overlap_weight * viol_m  # (m,)

    # Derivative of r_sum wrt theta_i and theta_j
    dr_ddt = 0.5 * (
        -(c_dt[iu_m, ju_m] / abs_cdt[iu_m, ju_m]) * s_dt[iu_m, ju_m]
        + (s_dt[iu_m, ju_m] / abs_sdt[iu_m, ju_m]) * c_dt[iu_m, ju_m]
    )

    ks = k_star[iu_m, ju_m]
    p_sel = projs[iu_m, ju_m, ks]
    sgn = np.where(p_sel >= 0.0, 1.0, -1.0)

    # Axis vectors for each active pair
    ax_x = np.empty(len(iu_m), dtype=np.float64)
    ax_y = np.empty(len(iu_m), dtype=np.float64)
    dax_dti_x = np.zeros(len(iu_m), dtype=np.float64)
    dax_dti_y = np.zeros(len(iu_m), dtype=np.float64)
    dax_dtj_x = np.zeros(len(iu_m), dtype=np.float64)
    dax_dtj_y = np.zeros(len(iu_m), dtype=np.float64)

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

    # pen = r_sum - sgn * (dx * ax_x + dy * ax_y)
    # dpen / dx_j = -sgn * ax_x, dpen / dx_i = +sgn * ax_x
    gx = d_loss_d_pen * (-sgn * ax_x)
    gy = d_loss_d_pen * (-sgn * ax_y)
    gti = d_loss_d_pen * (dr_ddt - sgn * (dx_m * dax_dti_x + dy_m * dax_dti_y))
    gtj = d_loss_d_pen * (-dr_ddt - sgn * (dx_m * dax_dtj_x + dy_m * dax_dtj_y))

    np.add.at(grad[:, 0], ju_m, gx)
    np.add.at(grad[:, 0], iu_m, -gx)
    np.add.at(grad[:, 1], ju_m, gy)
    np.add.at(grad[:, 1], iu_m, -gy)
    np.add.at(grad[:, 2], iu_m, gti)
    np.add.at(grad[:, 2], ju_m, gtj)

    return L_smooth + ov_loss, grad, exact_L, max_pen


def gradient_descent_compact(
    init_squares: np.ndarray,
    cfg: Exp1Config,
) -> tuple[np.ndarray, float, bool]:
    """Run Adam gradient descent from a valid configuration to compact L_n."""
    W = init_squares.copy()
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_W = W.copy()
    best_L = compute_bounding_box(W).side_length
    overlaps, _ = find_overlaps_float64(W, tol=1e-12)
    best_valid = len(overlaps) == 0

    for step in range(1, cfg.gd_steps + 1):
        # Cosine learning rate schedule
        lr = cfg.gd_lr * 0.5 * (1.0 + math.cos(math.pi * step / cfg.gd_steps))
        _, grad, _, _ = _compute_loss_and_grad(
            W,
            overlap_weight=cfg.gd_overlap_weight,
            margin=cfg.gd_margin,
            tau=cfg.smooth_tau,
        )
        # Clip gradient norm to stabilize tight contacts
        gnorm = float(np.linalg.norm(grad))
        if gnorm > 10.0:
            grad = grad * (10.0 / gnorm)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        W = W - lr * m_hat / (np.sqrt(v_hat) + eps)

        if step % 10 == 0 or step == cfg.gd_steps:
            overlaps, _ = find_overlaps_float64(W, tol=1e-12)
            if not overlaps:
                L_cur = compute_bounding_box(W).side_length
                if not best_valid or L_cur < best_L:
                    best_L = L_cur
                    best_W = W.copy()
                    best_valid = True

    return best_W, best_L, best_valid


def _sa_energy(squares: np.ndarray, penalty_weight: float = 20.0) -> tuple[float, float, int]:
    bbox = compute_bounding_box(squares)
    overlaps, _ = find_overlaps_float64(squares, tol=1e-12)
    ov_sum = sum(p for _, _, p in overlaps)
    return bbox.side_length + penalty_weight * ov_sum, bbox.side_length, len(overlaps)


def solve_puzzle_exp1(
    n: int,
    baseline_squares: np.ndarray,
    cfg: Exp1Config | None = None,
) -> Exp1Result:
    """Solve puzzle n using SA exploration + GD compaction hybrid."""
    import time

    t0 = time.perf_counter()
    if cfg is None:
        cfg = Exp1Config()

    rng = np.random.default_rng(cfg.seed + n * 1009)
    baseline_L = compute_bounding_box(baseline_squares).side_length
    baseline_score = baseline_L / math.sqrt(n) - 1.0

    best_valid_W = baseline_squares.copy()
    best_valid_L = baseline_L

    # Run GD directly on baseline + small random rotation seed first
    w_gd, l_gd, ok_gd = gradient_descent_compact(baseline_squares, cfg)
    gd_runs = 1
    sa_valid_found = 0
    if ok_gd and l_gd < best_valid_L:
        best_valid_L = l_gd
        best_valid_W = w_gd

    # Start SA from slightly perturbed baseline (and random angles)
    cur_W = baseline_squares.copy()
    cur_W[:, 2] = rng.uniform(-0.3, 0.3, size=n)
    # Expand slightly so SA starts feasible or near-feasible
    cur_W[:, :2] *= 1.05
    cur_E, cur_L, cur_ov = _sa_energy(cur_W)

    gd_interval = max(1, cfg.sa_steps // max(cfg.gd_triggers, 1))
    last_gd_step = 0

    for step in range(1, cfg.sa_steps + 1):
        frac = step / cfg.sa_steps
        temp = cfg.sa_t_start * ((cfg.sa_t_end / cfg.sa_t_start) ** frac)
        pos_scale = cfg.sa_pos_scale * (1.0 - 0.7 * frac)
        ang_scale = cfg.sa_ang_scale * (1.0 - 0.6 * frac)

        cand_W = cur_W.copy()
        idx = int(rng.integers(0, n))
        move_type = float(rng.random())
        if move_type < 0.65:
            # Translate + rotate single square
            cand_W[idx, 0] += rng.normal(0.0, pos_scale)
            cand_W[idx, 1] += rng.normal(0.0, pos_scale)
            cand_W[idx, 2] += rng.normal(0.0, ang_scale)
        elif move_type < 0.85:
            # Pull square slightly toward centroid
            cx, cy = np.mean(cand_W[:, 0]), np.mean(cand_W[:, 1])
            cand_W[idx, 0] += 0.15 * (cx - cand_W[idx, 0]) + rng.normal(0.0, 0.1 * pos_scale)
            cand_W[idx, 1] += 0.15 * (cy - cand_W[idx, 1]) + rng.normal(0.0, 0.1 * pos_scale)
            cand_W[idx, 2] += rng.normal(0.0, ang_scale)
        else:
            # Global shrink toward centroid
            cx, cy = np.mean(cand_W[:, 0]), np.mean(cand_W[:, 1])
            scale = 1.0 - rng.uniform(0.005, 0.03)
            cand_W[:, 0] = cx + scale * (cand_W[:, 0] - cx)
            cand_W[:, 1] = cy + scale * (cand_W[:, 1] - cy)

        cand_E, cand_L, cand_ov = _sa_energy(cand_W)
        delta_E = cand_E - cur_E
        if delta_E <= 0.0 or rng.random() < math.exp(-delta_E / max(temp, 1e-9)):
            cur_W, cur_E, cur_L, cur_ov = cand_W, cand_E, cand_L, cand_ov
            if cur_ov == 0:
                sa_valid_found += 1
                if cur_L < best_valid_L:
                    best_valid_L = cur_L
                    best_valid_W = cur_W.copy()

                # Trigger GD compaction when SA finds a valid state
                if step - last_gd_step >= gd_interval or cur_L <= best_valid_L + 0.05:
                    last_gd_step = step
                    gd_runs += 1
                    w_gd, l_gd, ok_gd = gradient_descent_compact(cur_W, cfg)
                    if ok_gd and l_gd < best_valid_L:
                        best_valid_L = l_gd
                        best_valid_W = w_gd
                        # Re-seed SA from improved valid state
                        cur_W = w_gd.copy()
                        cur_E, cur_L, cur_ov = _sa_energy(cur_W)

    # Final GD polish on best valid state
    gd_runs += 1
    w_gd, l_gd, ok_gd = gradient_descent_compact(best_valid_W, cfg)
    if ok_gd and l_gd < best_valid_L:
        best_valid_L = l_gd
        best_valid_W = w_gd

    # Shift to [0, L] origin cleanly
    bbox = compute_bounding_box(best_valid_W)
    best_valid_W[:, 0] -= bbox.xmin
    best_valid_W[:, 1] -= bbox.ymin
    overlaps, _ = find_overlaps_float64(best_valid_W, tol=1e-12)
    valid = len(overlaps) == 0
    final_L = compute_bounding_box(best_valid_W).side_length
    final_score = final_L / math.sqrt(n) - 1.0

    return Exp1Result(
        n=n,
        squares=best_valid_W,
        side_length=final_L,
        score=final_score,
        baseline_side=baseline_L,
        baseline_score=baseline_score,
        valid=valid,
        sa_valid_found=sa_valid_found,
        gd_runs=gd_runs,
        elapsed_sec=time.perf_counter() - t0,
    )
