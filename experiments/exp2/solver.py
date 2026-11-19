"""Experiment 2: Adaptive-Expansion SA + Adam + L-BFGS-B Hybrid Solver."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from experiments.exp1.solver import _compute_loss_and_grad, _sa_energy
from squares.geometry import compute_bounding_box, find_overlaps_float64


@dataclass
class Exp2Config:
    """Hyperparameters for Adaptive-Expansion SA + Adam + L-BFGS-B."""

    expansion_schedule: tuple[float, ...] = (1.08, 1.22, 1.40, 1.65, 1.95)
    sa_steps: int = 2000
    sa_t_start: float = 0.35
    sa_t_end: float = 0.002
    sa_pos_scale: float = 0.40
    sa_ang_scale: float = 0.55
    gd_steps: int = 450
    gd_lr: float = 0.025
    lbfgs_maxiter: int = 180
    gd_triggers: int = 3
    seed: int = 101


@dataclass
class Exp2Result:
    """Result of running exp2 on a single puzzle n."""

    n: int
    squares: np.ndarray
    side_length: float
    score: float
    baseline_side: float
    baseline_score: float
    valid: bool
    attempts_used: int
    winning_expansion: float
    sa_valid_found: int
    gd_runs: int
    elapsed_sec: float


def compact_adam_lbfgs(
    init_squares: np.ndarray,
    cfg: Exp2Config,
) -> tuple[np.ndarray, float, bool]:
    """Two-stage compaction: annealed Adam followed by L-BFGS-B polish."""
    W = init_squares.copy()
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_W = W.copy()
    best_L = compute_bounding_box(W).side_length
    overlaps, _ = find_overlaps_float64(W, tol=1e-12)
    best_valid = len(overlaps) == 0

    # Stage 2a: Annealed Adam
    for step in range(1, cfg.gd_steps + 1):
        frac = step / cfg.gd_steps
        lr = cfg.gd_lr * 0.5 * (1.0 + math.cos(math.pi * frac))
        tau = 0.02 * ((0.003 / 0.02) ** frac)
        ov_w = 50.0 * ((400.0 / 50.0) ** frac)

        _, grad, _, _ = _compute_loss_and_grad(W, overlap_weight=ov_w, margin=1e-6, tau=tau)
        gnorm = float(np.linalg.norm(grad))
        if gnorm > 10.0:
            grad = grad * (10.0 / gnorm)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        W = W - lr * m_hat / (np.sqrt(v_hat) + eps)

        if step % 15 == 0 or step == cfg.gd_steps:
            overlaps, _ = find_overlaps_float64(W, tol=1e-12)
            if not overlaps:
                L_cur = compute_bounding_box(W).side_length
                if not best_valid or L_cur < best_L:
                    best_L = L_cur
                    best_W = W.copy()
                    best_valid = True

    # Stage 2b: L-BFGS-B polish starting from best valid (or current W)
    start_W = best_W if best_valid else W
    shape = start_W.shape

    def obj_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
        w_cur = flat.reshape(shape)
        loss, grad, _, _ = _compute_loss_and_grad(
            w_cur,
            overlap_weight=2500.0,
            margin=1e-6,
            tau=0.001,
        )
        return loss, grad.ravel()

    res = minimize(
        obj_and_grad,
        start_W.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": cfg.lbfgs_maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    W_pol = res.x.reshape(shape)
    overlaps, _ = find_overlaps_float64(W_pol, tol=1e-12)
    if not overlaps:
        L_pol = compute_bounding_box(W_pol).side_length
        if not best_valid or L_pol < best_L:
            best_L = L_pol
            best_W = W_pol.copy()
            best_valid = True

    return best_W, best_L, best_valid


def solve_puzzle_exp2(
    n: int,
    baseline_squares: np.ndarray,
    cfg: Exp2Config | None = None,
) -> Exp2Result:
    """Solve puzzle n with adaptive initial expansion on failed attempts."""
    t0 = time.perf_counter()
    if cfg is None:
        cfg = Exp2Config()

    baseline_L = compute_bounding_box(baseline_squares).side_length
    baseline_score = baseline_L / math.sqrt(n) - 1.0

    best_valid_W = baseline_squares.copy()
    best_valid_L = baseline_L
    winning_expansion = 1.0
    total_sa_valid = 0
    total_gd_runs = 0
    attempts_used = 0

    for attempt_idx, exp_factor in enumerate(cfg.expansion_schedule):
        attempts_used = attempt_idx + 1
        rng = np.random.default_rng(cfg.seed + n * 1009 + attempt_idx * 9173)

        # Build expanded starting state around centroid with randomized angles
        cur_W = baseline_squares.copy()
        cx = float(np.mean(cur_W[:, 0]))
        cy = float(np.mean(cur_W[:, 1]))
        cur_W[:, 0] = cx + exp_factor * (cur_W[:, 0] - cx)
        cur_W[:, 1] = cy + exp_factor * (cur_W[:, 1] - cy)
        cur_W[:, 2] = rng.uniform(-math.pi / 4.0, math.pi / 4.0, size=n)

        # Add small spatial jitter proportional to extra room
        jitter = 0.25 * (exp_factor - 1.0) * baseline_L
        cur_W[:, 0] += rng.normal(0.0, jitter, size=n)
        cur_W[:, 1] += rng.normal(0.0, jitter, size=n)

        cur_E, cur_L, cur_ov = _sa_energy(cur_W)
        attempt_improved = False
        gd_interval = max(1, cfg.sa_steps // max(cfg.gd_triggers, 1))
        last_gd_step = 0

        for step in range(1, cfg.sa_steps + 1):
            frac = step / cfg.sa_steps
            temp = cfg.sa_t_start * ((cfg.sa_t_end / cfg.sa_t_start) ** frac)
            pos_scale = cfg.sa_pos_scale * exp_factor * (1.0 - 0.7 * frac)
            ang_scale = cfg.sa_ang_scale * (1.0 - 0.6 * frac)

            cand_W = cur_W.copy()
            idx = int(rng.integers(0, n))
            mv = float(rng.random())
            if mv < 0.60:
                cand_W[idx, 0] += rng.normal(0.0, pos_scale)
                cand_W[idx, 1] += rng.normal(0.0, pos_scale)
                cand_W[idx, 2] += rng.normal(0.0, ang_scale)
            elif mv < 0.85:
                mx, my = np.mean(cand_W[:, 0]), np.mean(cand_W[:, 1])
                cand_W[idx, 0] += 0.18 * (mx - cand_W[idx, 0]) + rng.normal(0.0, 0.1 * pos_scale)
                cand_W[idx, 1] += 0.18 * (my - cand_W[idx, 1]) + rng.normal(0.0, 0.1 * pos_scale)
                cand_W[idx, 2] += rng.normal(0.0, ang_scale)
            else:
                mx, my = np.mean(cand_W[:, 0]), np.mean(cand_W[:, 1])
                scale = 1.0 - rng.uniform(0.01, 0.04)
                cand_W[:, 0] = mx + scale * (cand_W[:, 0] - mx)
                cand_W[:, 1] = my + scale * (cand_W[:, 1] - my)

            cand_E, cand_L, cand_ov = _sa_energy(cand_W)
            delta_E = cand_E - cur_E
            if delta_E <= 0.0 or rng.random() < math.exp(-delta_E / max(temp, 1e-9)):
                cur_W, cur_E, cur_L, cur_ov = cand_W, cand_E, cand_L, cand_ov
                if cur_ov == 0:
                    total_sa_valid += 1
                    if cur_L < best_valid_L - 1e-6:
                        best_valid_L = cur_L
                        best_valid_W = cur_W.copy()
                        winning_expansion = exp_factor
                        attempt_improved = True

                    if step - last_gd_step >= gd_interval or cur_L <= best_valid_L + 0.15:
                        last_gd_step = step
                        total_gd_runs += 1
                        w_gd, l_gd, ok_gd = compact_adam_lbfgs(cur_W, cfg)
                        if ok_gd and l_gd < best_valid_L - 1e-6:
                            best_valid_L = l_gd
                            best_valid_W = w_gd
                            winning_expansion = exp_factor
                            attempt_improved = True
                            cur_W = w_gd.copy()
                            cur_E, cur_L, cur_ov = _sa_energy(cur_W)

        # If this expansion attempt already found a real improvement, stop early
        if attempt_improved and best_valid_L < baseline_L - 1e-4:
            break

    bbox = compute_bounding_box(best_valid_W)
    best_valid_W[:, 0] -= bbox.xmin
    best_valid_W[:, 1] -= bbox.ymin
    overlaps, _ = find_overlaps_float64(best_valid_W, tol=1e-12)
    valid = len(overlaps) == 0
    final_L = compute_bounding_box(best_valid_W).side_length
    final_score = final_L / math.sqrt(n) - 1.0

    return Exp2Result(
        n=n,
        squares=best_valid_W,
        side_length=final_L,
        score=final_score,
        baseline_side=baseline_L,
        baseline_score=baseline_score,
        valid=valid,
        attempts_used=attempts_used,
        winning_expansion=winning_expansion,
        sa_valid_found=total_sa_valid,
        gd_runs=total_gd_runs,
        elapsed_sec=time.perf_counter() - t0,
    )
