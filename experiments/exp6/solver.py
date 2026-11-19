"""Experiment 6: Expanding-Bucket Circle Drop + Bounded Simulated Annealing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from experiments.exp4.solver import try_pack_in_box_fast
from experiments.exp5.solver import _clamp_to_box, _pair_penetrations_row
from squares.geometry import compute_bounding_box, find_overlaps_float64


@dataclass
class Exp6Config:
    """Hyperparameters for Expanding-Bucket Circle Drop + Bounded SA."""

    circle_radius: float = 1.0 / math.sqrt(math.pi)  # Area-matched (~0.5642)
    bucket_expand_factor: float = 1.03
    bucket_relax_steps: int = 80
    sa_steps: int = 8000
    t_start: float = 0.28
    t_end: float = 0.0005
    binary_search_steps: int = 6
    restarts: int = 2
    polish_lbfgs: bool = True
    seed: int = 6060


@dataclass
class Exp6Result:
    """Result of running exp6 on a single puzzle n."""

    n: int
    squares: np.ndarray
    side_length: float
    score: float
    bucket_side: float
    baseline_side: float
    baseline_score: float
    valid: bool
    elapsed_sec: float


def drop_circles_expanding_bucket(
    n: int,
    radius: float,
    expand_factor: float,
    relax_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Drop n circles of given radius into an expanding square bucket until they fit."""
    L = max(2.0 * radius + 0.01, math.sqrt(n) * 1.02)
    pos = np.empty((n, 2), dtype=np.float64)
    pos[:, 0] = rng.uniform(radius, max(radius + 1e-4, L - radius), size=n)
    pos[:, 1] = rng.uniform(radius, max(radius + 1e-4, L - radius), size=n)

    min_dist = 2.0 * radius
    for _ in range(80):  # Up to 80 bucket expansions
        for _ in range(relax_steps):
            dx = pos[None, :, 0] - pos[:, None, 0]
            dy = pos[None, :, 1] - pos[:, None, 1]
            dist2 = dx * dx + dy * dy
            np.fill_diagonal(dist2, min_dist * min_dist + 1.0)
            dist = np.sqrt(dist2)

            overlap = np.maximum(0.0, min_dist - dist)
            max_ov = float(np.max(overlap))
            if max_ov <= 1e-6:
                return pos, L

            # Push overlapping circles apart symmetrically
            inv_d = np.where(dist > 1e-9, overlap / dist, 0.0)
            fx = -0.45 * np.sum(dx * inv_d, axis=1)
            fy = -0.45 * np.sum(dy * inv_d, axis=1)
            pos[:, 0] = np.clip(pos[:, 0] + fx, radius, L - radius)
            pos[:, 1] = np.clip(pos[:, 1] + fy, radius, L - radius)

        # Bucket too tight — expand and scale positions slightly
        L_new = L * expand_factor
        pos *= L_new / L
        L = L_new

    return pos, L


def _sa_from_bucket_in_box(
    n: int,
    bucket_pos: np.ndarray,
    bucket_L: float,
    target_L: float,
    cfg: Exp6Config,
    rng: np.random.Generator,
) -> tuple[bool, np.ndarray]:
    """Run Bounded SA inside [0, target_L]^2 starting from bucket-dropped circle centers."""
    W = np.empty((n, 3), dtype=np.float64)
    scale = target_L / max(bucket_L, 1e-6)
    W[:, :2] = bucket_pos * scale
    W[:, 2] = rng.uniform(0.0, math.pi / 2.0, size=n)
    for i in range(n):
        W[i, 0], W[i, 1] = _clamp_to_box(float(W[i, 0]), float(W[i, 1]), float(W[i, 2]), target_L)

    pen_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = _pair_penetrations_row(W, i)
        pen_mat[i, :] = row
        pen_mat[:, i] = row

    cur_energy = 0.5 * float(np.sum(pen_mat * pen_mat))
    pos_scale_init = 0.35 * target_L
    ang_scale_init = 0.60

    for step in range(1, cfg.sa_steps + 1):
        if cur_energy <= 1e-18:
            overlaps, _ = find_overlaps_float64(W, tol=1e-12)
            if not overlaps:
                return True, W

        frac = step / cfg.sa_steps
        temp = cfg.t_start * ((cfg.t_end / cfg.t_start) ** frac)
        pos_scale = pos_scale_init * (1.0 - 0.85 * frac) + 0.02
        ang_scale = ang_scale_init * (1.0 - 0.75 * frac) + 0.03

        idx = int(rng.integers(0, n))
        old_x, old_y, old_t = float(W[idx, 0]), float(W[idx, 1]), float(W[idx, 2])

        new_t = old_t + float(rng.normal(0.0, ang_scale))
        new_x, new_y = _clamp_to_box(
            old_x + float(rng.normal(0.0, pos_scale)),
            old_y + float(rng.normal(0.0, pos_scale)),
            new_t,
            target_L,
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
    return len(overlaps) == 0, W


def solve_puzzle_exp6(
    n: int,
    baseline_squares: np.ndarray,
    cfg: Exp6Config | None = None,
) -> Exp6Result:
    """Solve puzzle n via Expanding-Bucket Circle Drop + Bounded SA."""
    t0 = time.perf_counter()
    if cfg is None:
        cfg = Exp6Config()

    rng = np.random.default_rng(cfg.seed + n * 1009)
    baseline_L = compute_bounding_box(baseline_squares).side_length
    baseline_score = baseline_L / math.sqrt(n) - 1.0

    best_W = baseline_squares.copy()
    best_L = baseline_L

    bucket_pos, bucket_L = drop_circles_expanding_bucket(
        n,
        radius=cfg.circle_radius,
        expand_factor=cfg.bucket_expand_factor,
        relax_steps=cfg.bucket_relax_steps,
        rng=rng,
    )

    root = math.isqrt(n)
    if root * root == n:
        return Exp6Result(
            n=n,
            squares=best_W,
            side_length=best_L,
            score=baseline_score,
            bucket_side=bucket_L,
            baseline_side=baseline_L,
            baseline_score=baseline_score,
            valid=True,
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
        for r in range(cfg.restarts):
            if r > 0:
                bucket_pos, bucket_L = drop_circles_expanding_bucket(
                    n,
                    radius=cfg.circle_radius,
                    expand_factor=cfg.bucket_expand_factor,
                    relax_steps=cfg.bucket_relax_steps,
                    rng=rng,
                )
            ok, w_cand = _sa_from_bucket_in_box(n, bucket_pos, bucket_L, target_L, cfg, rng)
            if ok:
                bbox = compute_bounding_box(w_cand)
                w_cand[:, 0] -= bbox.xmin
                w_cand[:, 1] -= bbox.ymin
                l_cand = bbox.side_length
                if cfg.polish_lbfgs:
                    ok_pol, w_pol, l_pol = try_pack_in_box_fast(w_cand, l_cand * 0.99, maxiter=100)
                    if ok_pol and l_pol < l_cand:
                        w_cand, l_cand = w_pol, l_pol
                if l_cand < best_L:
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

    return Exp6Result(
        n=n,
        squares=best_W,
        side_length=final_L,
        score=final_score,
        bucket_side=bucket_L,
        baseline_side=baseline_L,
        baseline_score=baseline_score,
        valid=valid,
        elapsed_sec=time.perf_counter() - t0,
    )
