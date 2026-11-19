"""Experiment 5: Big Bang Bounded Simulated Annealing (+ L-BFGS-B Polish)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from experiments.exp4.solver import try_pack_in_box_fast
from squares.geometry import compute_bounding_box, find_overlaps_float64


@dataclass
class Exp5Config:
    """Hyperparameters for Big Bang Bounded Simulated Annealing."""

    sa_steps: int = 8000
    t_start: float = 0.30
    t_end: float = 0.0005
    binary_search_steps: int = 6
    restarts: int = 2
    polish_lbfgs: bool = True
    seed: int = 5050


@dataclass
class Exp5Result:
    """Result of running exp5 on a single puzzle n."""

    n: int
    squares: np.ndarray
    side_length: float
    score: float
    baseline_side: float
    baseline_score: float
    valid: bool
    sa_success_steps: int
    elapsed_sec: float


def _pair_penetrations_row(W: np.ndarray, idx: int) -> np.ndarray:
    """Compute SAT penetration of square `idx` against all other squares (shape (n,))."""
    x = W[:, 0]
    y = W[:, 1]
    theta = W[:, 2]

    dx = x - x[idx]
    dy = y - y[idx]
    ti = theta[idx]
    ci, si = math.cos(ti), math.sin(ti)
    cj, sj = np.cos(theta), np.sin(theta)

    d_theta = ti - theta
    r_sum = 0.5 + 0.5 * (np.abs(np.cos(d_theta)) + np.abs(np.sin(d_theta)))

    p0 = np.abs(dx * ci + dy * si)
    p1 = np.abs(-dx * si + dy * ci)
    p2 = np.abs(dx * cj + dy * sj)
    p3 = np.abs(-dx * sj + dy * cj)

    proj_max = np.maximum(np.maximum(p0, p1), np.maximum(p2, p3))
    pen = np.maximum(0.0, r_sum - proj_max)
    pen[idx] = 0.0
    return pen


def _clamp_to_box(x: float, y: float, theta: float, L: float) -> tuple[float, float]:
    """Clamp (x, y) so a unit square at angle theta stays strictly inside [0, L]^2."""
    h = 0.5 * (abs(math.cos(theta)) + abs(math.sin(theta)))
    lo = h + 1e-9
    hi = max(lo, L - h - 1e-9)
    return min(max(x, lo), hi), min(max(y, lo), hi)


def big_bang_sa_in_box(
    n: int,
    L: float,
    cfg: Exp5Config,
    rng: np.random.Generator,
) -> tuple[bool, np.ndarray, int]:
    """Start all squares at the center (L/2, L/2) and run bounded SA inside [0, L]^2."""
    # Big Bang initialization: all squares clustered at the center
    W = np.empty((n, 3), dtype=np.float64)
    W[:, 0] = 0.5 * L + rng.normal(0.0, 1e-3, size=n)
    W[:, 1] = 0.5 * L + rng.normal(0.0, 1e-3, size=n)
    W[:, 2] = rng.uniform(0.0, math.pi / 2.0, size=n)

    # Maintain symmetric pairwise overlap matrix O(n) per move
    pen_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = _pair_penetrations_row(W, i)
        pen_mat[i, :] = row
        pen_mat[:, i] = row

    cur_energy = 0.5 * float(np.sum(pen_mat * pen_mat))
    pos_scale_init = 0.45 * L
    ang_scale_init = 0.65

    for step in range(1, cfg.sa_steps + 1):
        if cur_energy <= 1e-18:
            overlaps, _ = find_overlaps_float64(W, tol=1e-12)
            if not overlaps:
                return True, W, step

        frac = step / cfg.sa_steps
        temp = cfg.t_start * ((cfg.t_end / cfg.t_start) ** frac)
        pos_scale = pos_scale_init * (1.0 - 0.85 * frac) + 0.02
        ang_scale = ang_scale_init * (1.0 - 0.75 * frac) + 0.03

        idx = int(rng.integers(0, n))
        old_x, old_y, old_t = float(W[idx, 0]), float(W[idx, 1]), float(W[idx, 2])

        if rng.random() < 0.15:
            # Occasional long jump inside the bounded box to escape jams
            new_t = float(rng.uniform(0.0, math.pi / 2.0))
            h = 0.5 * (abs(math.cos(new_t)) + abs(math.sin(new_t)))
            lo, hi = h + 1e-9, max(h + 1e-9, L - h - 1e-9)
            new_x = float(rng.uniform(lo, hi))
            new_y = float(rng.uniform(lo, hi))
        else:
            new_t = old_t + float(rng.normal(0.0, ang_scale))
            new_x, new_y = _clamp_to_box(
                old_x + float(rng.normal(0.0, pos_scale)),
                old_y + float(rng.normal(0.0, pos_scale)),
                new_t,
                L,
            )

        W[idx, 0], W[idx, 1], W[idx, 2] = new_x, new_y, new_t
        new_row = _pair_penetrations_row(W, idx)
        old_row = pen_mat[idx, :]

        delta_e = float(np.sum(new_row * new_row) - np.sum(old_row * old_row))
        if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / max(temp, 1e-12)):
            pen_mat[idx, :] = new_row
            pen_mat[:, idx] = new_row
            cur_energy = max(0.0, cur_energy + delta_e)
        else:
            W[idx, 0], W[idx, 1], W[idx, 2] = old_x, old_y, old_t

    overlaps, _ = find_overlaps_float64(W, tol=1e-12)
    return len(overlaps) == 0, W, cfg.sa_steps


def solve_puzzle_exp5(
    n: int,
    baseline_squares: np.ndarray,
    cfg: Exp5Config | None = None,
) -> Exp5Result:
    """Solve puzzle n via Big Bang Bounded SA + Binary Search (+ optional L-BFGS-B polish)."""
    t0 = time.perf_counter()
    if cfg is None:
        cfg = Exp5Config()

    rng = np.random.default_rng(cfg.seed + n * 1009)
    baseline_L = compute_bounding_box(baseline_squares).side_length
    baseline_score = baseline_L / math.sqrt(n) - 1.0

    best_W = baseline_squares.copy()
    best_L = baseline_L
    first_success_step = 0

    root = math.isqrt(n)
    if root * root == n:
        return Exp5Result(
            n=n,
            squares=best_W,
            side_length=best_L,
            score=baseline_score,
            baseline_side=baseline_L,
            baseline_score=baseline_score,
            valid=True,
            sa_success_steps=0,
            elapsed_sec=time.perf_counter() - t0,
        )

    low = math.sqrt(n)
    high = baseline_L

    for step in range(cfg.binary_search_steps):
        if high - low < 1e-4:
            break
        target_L = 0.5 * (low + high) if step > 0 else max(low + 0.01, baseline_L - 0.15)
        if target_L >= best_L:
            target_L = 0.5 * (low + best_L)

        found = False
        for _ in range(cfg.restarts):
            ok, w_cand, s_step = big_bang_sa_in_box(n, target_L, cfg, rng)
            if ok:
                bbox = compute_bounding_box(w_cand)
                w_cand[:, 0] -= bbox.xmin
                w_cand[:, 1] -= bbox.ymin
                l_cand = bbox.side_length
                if cfg.polish_lbfgs:
                    # Try tightening slightly further with L-BFGS-B
                    ok_pol, w_pol, l_pol = try_pack_in_box_fast(w_cand, l_cand * 0.99, maxiter=100)
                    if ok_pol and l_pol < l_cand:
                        w_cand, l_cand = w_pol, l_pol
                if l_cand < best_L:
                    best_W, best_L = w_cand, l_cand
                    high = min(high, best_L)
                    if first_success_step == 0:
                        first_success_step = s_step
                    found = True
                    break

        if not found:
            low = target_L

    overlaps, _ = find_overlaps_float64(best_W, tol=1e-12)
    valid = len(overlaps) == 0
    final_L = compute_bounding_box(best_W).side_length
    final_score = final_L / math.sqrt(n) - 1.0

    return Exp5Result(
        n=n,
        squares=best_W,
        side_length=final_L,
        score=final_score,
        baseline_side=baseline_L,
        baseline_score=baseline_score,
        valid=valid,
        sa_success_steps=first_success_step,
        elapsed_sec=time.perf_counter() - t0,
    )
