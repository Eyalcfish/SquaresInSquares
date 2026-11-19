"""Experiment 7: Parallel multi-family constructive search, composites, L-BFGS-B polish, and mpmath certification."""

from __future__ import annotations

import concurrent.futures
import math
from typing import Any

import numpy as np

from experiments.exp4.solver import try_pack_in_box_fast
from experiments.exp7.constructions import (
    PackingCandidate,
    SQRT2,
    _best_4_corner_splits,
    _count_clipped_45_core,
    _count_clipped_rect_general,
    _count_corner_general,
    apply_l_frame,
    best_tilted_strip_params,
    build_l_shaped_tilted,
    build_octagon_45,
    build_octagon_general,
    build_quad_composite,
    trim_squares_tight,
)
from squares.geometry import compute_bounding_box, find_overlaps_float64
from squares.io import trivial_grid_side
from squares.scoring import verify_puzzle_mpmath


def _update_best_map(
    best: dict[int, PackingCandidate], cand: PackingCandidate, n_max: int
) -> None:
    cap = min(cand.n_squares, n_max)
    if cap <= 0:
        return
    if cand.side_length >= best[cap].side_length - 1e-12:
        return
    for n in range(cap, 0, -1):
        if cand.side_length < best[n].side_length - 1e-12:
            best[n] = cand
        else:
            break


def _init_grid_best(n_max: int) -> dict[int, PackingCandidate]:
    return {
        n: PackingCandidate(
            family="grid",
            n_squares=n,
            side_length=float(trivial_grid_side(n)),
            params=(n,),
        )
        for n in range(1, n_max + 1)
    }


# ---------------------------------------------------------------------------
# Family 1 Worker: Off-Center Clipped 4-Corner 45-deg Octagon + Sliding Band
# ---------------------------------------------------------------------------

def _search_oct45_for_w(args: tuple[int, int]) -> dict[int, PackingCandidate]:
    w, n_max = args
    local_best = _init_grid_best(n_max)

    for half_m in range(2, 66):
        L0 = 0.5 * half_m + w / SQRT2
        if L0 > 32.15 or L0 < 1.5:
            continue

        Lf = int(math.floor(L0 + 1e-12))
        v_cands = (0.0, 1.0, 2.0, 3.0) if (half_m % 2 == 0) else (0.5, 1.5, 2.5, 3.5)

        min_useful = Lf * Lf + 1
        while min_useful <= n_max and local_best[min_useful].side_length <= L0:
            min_useful += 1
        if min_useful > n_max:
            continue

        # Mode A: Free-sliding diagonal band (BL/TR empty) + exact critical L jumps
        for v0 in v_cands:
            if abs(v0) > L0 * 0.6:
                continue
            cand_Ls = [L0]
            for j in range(w):
                vj = v0 + (j - 0.5 * (w - 1)) * SQRT2
                span0 = L0 - SQRT2 - abs(vj)
                if span0 >= -0.99:
                    q_next = int(math.floor(SQRT2 * span0 + 1e-12)) + 2
                    L_crit = (q_next + 1) / SQRT2 + abs(vj)
                    if L0 < L_crit <= min(L0 + 0.45, 32.15):
                        cand_Ls.append(L_crit)
            for L in cand_Ls:
                Lf_cur = int(math.floor(L + 1e-12))
                M_ul = int(math.floor(L - 2.0 - v0 - w / SQRT2 + 1e-12))
                M_lr = int(math.floor(L - 2.0 + v0 - w / SQRT2 + 1e-12))
                n_cor, splits = _best_4_corner_splits(Lf_cur, M_ul, M_lr, -1, -1)
                n_core = _count_clipped_45_core(L, L, v0, w, 1, free_slide=True)
                N = n_cor + n_core
                if N > Lf_cur * Lf_cur:
                    cand = PackingCandidate(
                        family="oct45",
                        n_squares=N,
                        side_length=L,
                        params=(L, L, v0, w, 1, True, splits),
                    )
                    _update_best_map(local_best, cand, n_max)

        # Mode B: Clipped w x d rectangle + 4 axis-aligned corners
        d_max = min(int(math.ceil(SQRT2 * L0)) + 2, 46)
        for v0 in v_cands:
            if abs(v0) > L0 * 0.55:
                continue
            M_ul = int(math.floor(L0 - 2.0 - v0 - w / SQRT2 + 1e-12))
            M_lr = int(math.floor(L0 - 2.0 + v0 - w / SQRT2 + 1e-12))
            for d in range(1, d_max + 1):
                u_align1 = math.floor(L0 - d / SQRT2) + d / SQRT2
                u_align2 = u_align1 + 1.0
                u_cands = (
                    L0,
                    u_align1,
                    u_align2,
                    0.5 * (u_align1 + u_align2),
                    u_align1 - 1.0,
                    u_align2 + 1.0,
                )
                for u0 in u_cands:
                    if u0 < 0.5 or u0 > 2.0 * L0 - 0.5:
                        continue
                    n_core = _count_clipped_45_core(L0, u0, v0, w, d, free_slide=False)
                    if n_core <= 0:
                        continue
                    M_bl = int(math.floor(u0 - 2.0 - d / SQRT2 + 1e-12))
                    M_tr = int(math.floor(2.0 * L0 - 2.0 - u0 - d / SQRT2 + 1e-12))
                    # Quick area upper bound
                    if n_core + Lf * Lf < min_useful:
                        continue
                    n_cor, splits = _best_4_corner_splits(Lf, M_ul, M_lr, M_bl, M_tr)
                    N = n_core + n_cor
                    if N >= min_useful:
                        cand = PackingCandidate(
                            family="oct45",
                            n_squares=N,
                            side_length=L0,
                            params=(L0, u0, v0, w, d, False, splits),
                        )
                        _update_best_map(local_best, cand, n_max)
                        while min_useful <= n_max and local_best[min_useful].side_length <= L0:
                            min_useful += 1

    # Also check exact tight L = (w + d) / SQRT2 (symmetric unclipped 45-deg rect touching walls)
    for d in range(w, 46):
        L = (w + d) / SQRT2
        if L > 32.15 or L < 1.5:
            continue
        Lf = int(math.floor(L + 1e-12))
        M_ul = int(math.floor(L - 2.0 - w / SQRT2 + 1e-12))
        M_bl = int(math.floor(L - 2.0 - d / SQRT2 + 1e-12))
        n_cor, splits = _best_4_corner_splits(Lf, M_ul, M_ul, M_bl, M_bl)
        N = w * d + n_cor
        if N > Lf * Lf:
            cand = PackingCandidate(
                family="oct45",
                n_squares=N,
                side_length=L,
                params=(L, L, 0.0, w, d, False, splits),
            )
            _update_best_map(local_best, cand, n_max)

    return local_best


def search_octagon_45_all(n_max: int = 1000, workers: int = 8) -> dict[int, PackingCandidate]:
    """Parallelized full search of Family 1 (Off-Center Clipped 4-Corner 45-deg Octagon)."""
    best = _init_grid_best(n_max)
    tasks = [(w, n_max) for w in range(1, 33)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for local_best in ex.map(_search_oct45_for_w, tasks):
            for n, cand in local_best.items():
                if cand.side_length < best[n].side_length - 1e-12:
                    best[n] = cand
    return best


# ---------------------------------------------------------------------------
# Family 2 Worker: General-Angle Clipped 4-Corner Rotated Block
# ---------------------------------------------------------------------------

def _search_octgen_for_angle(args: tuple[float, int, dict[int, float]]) -> dict[int, PackingCandidate]:
    theta, n_max, cur_best_L = args
    ct, st = math.cos(theta), math.sin(theta)
    local_best = _init_grid_best(n_max)
    for n in range(1, n_max + 1):
        local_best[n] = PackingCandidate("grid", n, cur_best_L[n], (n,))

    # Sweep L in steps of 0.03 (plus tighter 0.015 for L <= 12)
    L_vals = [x / 100.0 for x in range(250, 1200, 2)] + [x / 100.0 for x in range(1200, 3210, 4)]
    for L in L_vals:
        Lf = int(math.floor(L + 1e-12))
        min_useful = Lf * Lf + 1
        while min_useful <= n_max and local_best[min_useful].side_length <= L:
            min_useful += 1
        if min_useful > n_max:
            continue

        x0 = 0.5 * L
        y0 = 0.5 * L
        base_rhs = (0.5 * L - 1.0) * (ct + st)
        w_lim = min(int(L * 1.42), 34)

        for w in range(1, w_lim + 1):
            rhs_w = base_rhs - 0.5 * w
            for d in range(w, w_lim + 1):
                rhs_d = base_rhs - 0.5 * d
                n_rect = _count_clipped_rect_general(L, x0, y0, ct, st, w, d)
                if n_rect <= 0 or n_rect + Lf * Lf < min_useful:
                    continue
                # Symmetric splits (a, Lf-a) for 4 corners
                a_mid = min(Lf, max(0, int(rhs_w / max(ct, 1e-6)) + 1))
                cands_a = set([0, Lf // 2, (Lf + 1) // 2, Lf, a_mid, max(0, a_mid - 1), min(Lf, a_mid + 1)])
                best_c = -1
                best_a = 0
                for a in cands_a:
                    b = Lf - a
                    val = 2 * _count_corner_general(ct, st, rhs_w, a, a) + 2 * _count_corner_general(
                        st, ct, rhs_d, b, b
                    )
                    if val > best_c:
                        best_c = val
                        best_a = a
                N = n_rect + best_c
                if N >= min_useful:
                    b = Lf - best_a
                    splits = (b, b, best_a, b)
                    cand = PackingCandidate(
                        family="octgen",
                        n_squares=N,
                        side_length=L,
                        params=(L, x0, y0, theta, w, d, splits),
                    )
                    _update_best_map(local_best, cand, n_max)
                    while min_useful <= n_max and local_best[min_useful].side_length <= L:
                        min_useful += 1
    return local_best


def search_octagon_general_all(
    cur_best: dict[int, PackingCandidate], n_max: int = 1000, workers: int = 8
) -> dict[int, PackingCandidate]:
    """Parallelized search of Family 2 (General-Angle Clipped 4-Corner Rotated Block)."""
    angles_deg = [
        15.0, 18.4349, 21.8014, 24.5, 26.5651, 28.0725, 30.0,
        32.0, 33.6901, 35.5, 36.8699, 38.6598, 40.18, 42.0, 43.5,
    ]
    cur_L_map = {n: cur_best[n].side_length for n in range(1, n_max + 1)}
    tasks = [(math.radians(deg), n_max, cur_L_map) for deg in angles_deg]
    best = dict(cur_best)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for local_best in ex.map(_search_octgen_for_angle, tasks):
            for n, cand in local_best.items():
                if cand.family != "grid" and cand.side_length < best[n].side_length - 1e-12:
                    best[n] = cand
    return best


# ---------------------------------------------------------------------------
# Family 3 Worker: L-Shaped Tilted Strips + Axis-Aligned Bulk
# ---------------------------------------------------------------------------

def _search_l_strips_chunk(args: tuple[list[float], int]) -> dict[int, PackingCandidate]:
    L_list, n_max = args
    local_best = _init_grid_best(n_max)
    for L in L_list:
        Wf = int(math.floor(L + 1e-12))
        best_N = Wf * Wf
        best_params = None
        for B in range(0, Wf + 1):
            n_top, k_top, t_top = best_tilted_strip_params(L, L - B)
            for A in range(0, Wf + 1):
                n_r, k_r, t_r = best_tilted_strip_params(float(B), L - A)
                N = A * B + n_top + n_r
                if N > best_N:
                    best_N = N
                    best_params = (L, A, B, k_top, t_top, k_r, t_r)
        if best_params is not None and best_N > Wf * Wf:
            cand = PackingCandidate(
                family="l_strip",
                n_squares=best_N,
                side_length=L,
                params=best_params,
            )
            _update_best_map(local_best, cand, n_max)
    return local_best


def search_l_shaped_tilted_all(
    cur_best: dict[int, PackingCandidate], n_max: int = 1000, workers: int = 8
) -> dict[int, PackingCandidate]:
    """Parallelized sweep of Family 3 (L-Shaped Tilted Strips)."""
    all_L = [x / 100.0 for x in range(250, 3210, 2)]
    chunks = [all_L[i::workers] for i in range(workers)]
    best = dict(cur_best)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for local_best in ex.map(_search_l_strips_chunk, [(c, n_max) for c in chunks]):
            for n, cand in local_best.items():
                if cand.family != "grid" and cand.side_length < best[n].side_length - 1e-12:
                    best[n] = cand
    return best


def materialize_candidate(cand: PackingCandidate, target_n: int, baseline_squares: np.ndarray) -> np.ndarray:
    """Build concrete (target_n, 3) coordinates from a PackingCandidate."""
    if cand.family == "grid":
        return baseline_squares.copy()
    if cand.family == "oct45":
        L, u0, v0, w, d, fs, splits = cand.params
        W = build_octagon_45(L, u0, v0, w, d, fs, splits)
    elif cand.family == "octgen":
        L, x0, y0, theta, w, d, splits = cand.params
        W = build_octagon_general(L, x0, y0, theta, w, d, splits)
    elif cand.family == "l_strip":
        L, A, B, k_top, t_top, k_r, t_r = cand.params
        W = build_l_shaped_tilted(L, A, B, k_top, t_top, k_r, t_r)
    else:
        raise ValueError(f"Unknown family: {cand.family}")

    if len(W) > target_n:
        W = trim_squares_tight(W, target_n)
    return W


# ---------------------------------------------------------------------------
# Composite Propagation: L-Frames + 2x2 Quad Composites + Monotonic Prefix
# ---------------------------------------------------------------------------

def propagate_composites(
    puzzles: dict[int, np.ndarray],
    n_max: int = 1000,
) -> tuple[dict[int, np.ndarray], int]:
    """Propagate record packings via integer L-Frames, 2x2 Quad Composites, and Prefix Trimming."""
    puzzles = dict(puzzles)
    side_L = {n: compute_bounding_box(puzzles[n]).side_length for n in range(1, n_max + 1)}
    improvements = 0

    # 1. Integer L-Frames: wrap any packing n0 with thickness p >= 1
    for n0 in range(2, n_max):
        L0 = side_L[n0]
        Lf0 = int(math.floor(L0 + 1e-12))
        if L0 >= trivial_grid_side(n0) - 1e-6:
            continue
        for p in range(1, 32):
            L_new = L0 + p
            if L_new > 32.1:
                break
            N_new = n0 + p * p + 2 * p * Lf0
            cap = min(N_new, n_max)
            if L_new + 1e-7 < side_L[cap]:
                W_frame = apply_l_frame(puzzles[n0], p)
                ov, _ = find_overlaps_float64(W_frame, tol=1e-12)
                if ov:
                    continue
                actual_L = compute_bounding_box(W_frame).side_length
                for n in range(cap, 0, -1):
                    if actual_L + 1e-7 < side_L[n]:
                        W_trim = trim_squares_tight(W_frame, n)
                        ov_t, _ = find_overlaps_float64(W_trim, tol=1e-12)
                        if not ov_t:
                            t_L = compute_bounding_box(W_trim).side_length
                            if t_L + 1e-7 < side_L[n]:
                                puzzles[n] = W_trim
                                side_L[n] = t_L
                                improvements += 1
                    else:
                        break

    # 2. 2x2 Quad Composites: pick top non-trivial packings (n1, n2)
    good_ns = [n for n in range(5, min(400, n_max + 1)) if side_L[n] < trivial_grid_side(n) - 0.05]
    # Keep Pareto-best by side length bucket (every 0.1) to stay fast
    buckets: dict[int, int] = {}
    for n in good_ns:
        b = int(round(side_L[n] * 10.0))
        if b not in buckets or n > buckets[b]:
            buckets[b] = n
    rep_ns = sorted(buckets.values())

    for idx1, n1 in enumerate(rep_ns):
        L1 = side_L[n1]
        for n2 in rep_ns[idx1:]:
            L2 = side_L[n2]
            L_tot = L1 + L2
            if L_tot > 32.1:
                continue
            n_tl, k_tl, t_tl = best_tilted_strip_params(L1, L2)
            n_br, k_br, t_br = best_tilted_strip_params(L1, L2)
            N_tot = n1 + n2 + n_tl + n_br
            cap = min(N_tot, n_max)
            if cap <= 0 or L_tot + 1e-6 >= side_L[cap]:
                continue
            W_q = build_quad_composite(puzzles[n1], puzzles[n2], L_tot, L1, k_tl, t_tl, k_br, t_br)
            ov, _ = find_overlaps_float64(W_q, tol=1e-12)
            if ov:
                continue
            actual_L = compute_bounding_box(W_q).side_length
            for n in range(cap, 0, -1):
                if actual_L + 1e-7 < side_L[n]:
                    W_trim = trim_squares_tight(W_q, n)
                    ov_t, _ = find_overlaps_float64(W_trim, tol=1e-12)
                    if not ov_t:
                        t_L = compute_bounding_box(W_trim).side_length
                        if t_L + 1e-7 < side_L[n]:
                            puzzles[n] = W_trim
                            side_L[n] = t_L
                            improvements += 1
                else:
                    break

    # 3. Monotonic Prefix Trimming pass (from n_max down to 2)
    for n in range(n_max - 1, 1, -1):
        if side_L[n + 1] + 1e-7 < side_L[n]:
            W_trim = trim_squares_tight(puzzles[n + 1], n)
            ov, _ = find_overlaps_float64(W_trim, tol=1e-12)
            if not ov:
                t_L = compute_bounding_box(W_trim).side_length
                if t_L + 1e-7 < side_L[n]:
                    puzzles[n] = W_trim
                    side_L[n] = t_L
                    improvements += 1

    return puzzles, improvements


# ---------------------------------------------------------------------------
# Layer 2: Multi-Start + Symmetry-Breaking Fixed-Box L-BFGS-B Polish
# ---------------------------------------------------------------------------

def _polish_one_puzzle(args: tuple[int, np.ndarray, int, int, int]) -> tuple[int, np.ndarray, float]:
    n, start_W, n_warm_pert, n_fresh, maxiter = args
    rng = np.random.default_rng(7070 + n * 1009)
    best_W = start_W.copy()
    best_L = compute_bounding_box(best_W).side_length

    root = math.isqrt(n)
    if root * root == n or n <= 3:
        return n, best_W, best_L

    low = math.sqrt(n)
    high = best_L

    # Phase A: Symmetry-breaking warm-start polish (shrinks constructive packings!)
    for step in range(9):
        if high - low < 2e-4:
            break
        target_L = high - 0.08 if step == 0 else 0.5 * (low + high)
        target_L = max(low + 1e-4, min(target_L, best_L - 1e-4))
        found = False
        for p_idx in range(n_warm_pert):
            warm = best_W.copy()
            warm[:, :2] *= target_L / best_L
            if p_idx > 0:
                ang_sig = 0.03 if p_idx <= 2 else 0.10
                pos_sig = 0.015 if p_idx <= 2 else 0.04
                warm[:, 2] += rng.normal(0.0, ang_sig, size=n)
                warm[:, 0] += rng.normal(0.0, pos_sig, size=n)
                warm[:, 1] += rng.normal(0.0, pos_sig, size=n)
            ok, w_c, l_c = try_pack_in_box_fast(warm, target_L, maxiter=maxiter)
            if ok and l_c < best_L - 1e-6:
                best_W, best_L = w_c, l_c
                high = min(high, best_L)
                found = True
                break
        if not found:
            low = target_L

    # Phase B: Fresh random/mixed starts to discover novel non-45-deg topologies (for small/medium n)
    if n_fresh > 0:
        target_probe = max(math.sqrt(n) + 0.02, best_L - 0.04)
        for r in range(n_fresh):
            lo_c = min(0.5, target_probe * 0.25)
            hi_c = max(lo_c + 0.01, target_probe - lo_c)
            seed_W = np.empty((n, 3), dtype=np.float64)
            seed_W[:, 0] = rng.uniform(lo_c, hi_c, size=n)
            seed_W[:, 1] = rng.uniform(lo_c, hi_c, size=n)
            if r % 3 == 0:
                seed_W[:, 2] = rng.choice([0.0, math.pi / 4.0], size=n) + rng.normal(0.0, 0.12, size=n)
            elif r % 3 == 1:
                seed_W[:, 2] = rng.choice([0.0, math.radians(28.07)], size=n) + rng.normal(0.0, 0.12, size=n)
            else:
                seed_W[:, 2] = rng.uniform(0.0, math.pi / 2.0, size=n)

            ok, w_c, l_c = try_pack_in_box_fast(seed_W, target_probe, maxiter=maxiter + 60)
            if ok and l_c < best_L - 1e-6:
                best_W, best_L = w_c, l_c
                # Tighten the newly discovered topology via binary search
                lo2, hi2 = math.sqrt(n), best_L
                for _ in range(7):
                    if hi2 - lo2 < 2e-4:
                        break
                    mid = 0.5 * (lo2 + hi2)
                    warm = best_W.copy()
                    warm[:, :2] *= mid / best_L
                    ok2, w2, l2 = try_pack_in_box_fast(warm, mid, maxiter=maxiter)
                    if ok2 and l2 < best_L - 1e-6:
                        best_W, best_L = w2, l2
                        hi2 = best_L
                    else:
                        lo2 = mid
                target_probe = max(math.sqrt(n) + 0.02, best_L - 0.02)

    return n, best_W, best_L


def polish_puzzles_parallel(
    puzzles: dict[int, np.ndarray],
    target_ns: list[int],
    workers: int = 8,
) -> tuple[dict[int, np.ndarray], int]:
    """Run symmetry-breaking + multi-start Fixed-Box L-BFGS-B polish across target_ns."""
    tasks = []
    for n in target_ns:
        if n <= 25:
            tasks.append((n, puzzles[n], 4, 12, 220))
        elif n <= 45:
            tasks.append((n, puzzles[n], 3, 6, 180))
        elif n <= 85:
            tasks.append((n, puzzles[n], 3, 2, 150))
        else:
            tasks.append((n, puzzles[n], 2, 0, 120))

    updated = dict(puzzles)
    improved = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for n, w_pol, l_pol in ex.map(_polish_one_puzzle, tasks, chunksize=1):
            old_L = compute_bounding_box(updated[n]).side_length
            if l_pol < old_L - 1e-6:
                ov, _ = find_overlaps_float64(w_pol, tol=1e-12)
                if not ov:
                    updated[n] = w_pol
                    improved += 1
    return updated, improved


# ---------------------------------------------------------------------------
# Layer 4: Full 1,000-Puzzle 60-Digit mpmath Certification
# ---------------------------------------------------------------------------

def _certify_one_mpmath(args: tuple[int, np.ndarray, np.ndarray, int]) -> tuple[int, np.ndarray, bool]:
    n, W, baseline_W, dps = args
    cur = W.copy()
    # Skip expensive mpmath on untouched integer baseline grids (axis-aligned integers are exact)
    if np.array_equal(cur, baseline_W):
        return n, cur, True

    for scale_eps in (0.0, 1e-11, 1e-9, 1e-7, 1e-5):
        if scale_eps > 0.0:
            cx, cy = float(np.mean(cur[:, 0])), float(np.mean(cur[:, 1]))
            s = 1.0 + scale_eps
            cur[:, 0] = cx + s * (cur[:, 0] - cx)
            cur[:, 1] = cy + s * (cur[:, 1] - cy)
            bb = compute_bounding_box(cur)
            cur[:, 0] -= bb.xmin
            cur[:, 1] -= bb.ymin
        raw = [(f"{cur[i, 0]:.17g}", f"{cur[i, 1]:.17g}", f"{cur[i, 2]:.17g}") for i in range(n)]
        ok, _, _, _ = verify_puzzle_mpmath(n, raw, dps=dps)
        if ok:
            return n, cur, True

    # Fallback to exact grid if repair somehow fails
    return n, baseline_W.copy(), False


def certify_all_mpmath(
    puzzles: dict[int, np.ndarray],
    baseline_puzzles: dict[int, np.ndarray],
    n_max: int = 1000,
    workers: int = 8,
    dps: int = 60,
) -> tuple[dict[int, np.ndarray], int, int]:
    """Verify and certify all n=1..n_max puzzles under 60-digit mpmath."""
    tasks = [(n, puzzles[n], baseline_puzzles[n], dps) for n in range(1, n_max + 1)]
    certified = dict(puzzles)
    passed = 0
    fallbacks = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for n, w_cert, ok in ex.map(_certify_one_mpmath, tasks, chunksize=8):
            certified[n] = w_cert
            if ok:
                passed += 1
            else:
                fallbacks += 1
    return certified, passed, fallbacks
