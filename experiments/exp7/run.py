"""Run Experiment 7: Multi-Family Geometric Constructions + Composites + L-BFGS-B Polish + 60-Digit mpmath."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.exp7.search import (
    certify_all_mpmath,
    materialize_candidate,
    polish_puzzles_parallel,
    propagate_composites,
    search_l_shaped_tilted_all,
    search_octagon_45_all,
    search_octagon_general_all,
)
from squares.geometry import compute_bounding_box, find_overlaps_float64
from squares.io import load_submission, save_submission
from squares.scoring import score_submission
from squares.visualize import plot_puzzle_grid, plot_score_profile

KAGGLE_TOP1_REF = 22.03995
KAGGLE_TOP5_REF = 22.05809


def _cand_score(cand_map: dict, n_max: int) -> float:
    return sum(cand_map[n].side_length / math.sqrt(n) - 1.0 for n in range(1, n_max + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp7: Full 1,000-puzzle portfolio solver.")
    parser.add_argument("--baseline-csv", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="experiments/exp7/outputs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--skip-polish", action="store_true")
    parser.add_argument("--skip-mpmath", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    baseline_puzzles = load_submission(args.baseline_csv)
    baseline_report = score_submission(baseline_puzzles, n_max=args.n_max)
    print(f"[Stage 0] Baseline Grid Score             : {baseline_report.total_score:.5f}", flush=True)

    # Stage 1: Off-Center Clipped 4-Corner 45-deg Octagon + Sliding Bands
    t0 = time.perf_counter()
    best_cands = search_octagon_45_all(n_max=args.n_max, workers=args.workers)
    print(
        f"[Stage 1] + 45-deg Octagon & Bands        : {_cand_score(best_cands, args.n_max):.5f}  "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    # Stage 2: General-Angle Clipped 4-Corner Rotated Blocks
    t0 = time.perf_counter()
    best_cands = search_octagon_general_all(best_cands, n_max=args.n_max, workers=args.workers)
    print(
        f"[Stage 2] + General-Angle Rotated Blocks  : {_cand_score(best_cands, args.n_max):.5f}  "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    # Stage 3: L-Shaped Tilted Strips + Axis-Aligned Bulk
    t0 = time.perf_counter()
    best_cands = search_l_shaped_tilted_all(best_cands, n_max=args.n_max, workers=args.workers)
    print(
        f"[Stage 3] + L-Shaped Tilted Strips        : {_cand_score(best_cands, args.n_max):.5f}  "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    # Stage 4: Materialize exact coordinates & verify float64 SAT
    t0 = time.perf_counter()
    puzzles: dict[int, any] = {}
    family_counts: dict[str, int] = {}
    for n in range(1, args.n_max + 1):
        cand = best_cands[n]
        W = materialize_candidate(cand, n, baseline_puzzles[n])
        ov, _ = find_overlaps_float64(W, tol=1e-12)
        if ov or compute_bounding_box(W).side_length > compute_bounding_box(baseline_puzzles[n]).side_length + 1e-6:
            W = baseline_puzzles[n].copy()
            fam = "grid_fallback"
        else:
            fam = cand.family
        puzzles[n] = W
        family_counts[fam] = family_counts.get(fam, 0) + 1

    rep_mat = score_submission(puzzles, n_max=args.n_max)
    print(
        f"[Stage 4a] Materialized (float64 valid={rep_mat.valid_count}/{args.n_max}): "
        f"{rep_mat.total_score:.5f}  families={family_counts} ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    # Stage 4b: Composite Propagation (L-Frames + 2x2 Quads + Monotonic Prefix Trimming)
    t0 = time.perf_counter()
    puzzles, comp_imps = propagate_composites(puzzles, n_max=args.n_max)
    rep_comp = score_submission(puzzles, n_max=args.n_max)
    print(
        f"[Stage 4b] + Composites & Prefix Trim     : {rep_comp.total_score:.5f}  "
        f"(improved={comp_imps}, {time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    # Stage 5: High-Effort Multi-Start & Symmetry-Breaking L-BFGS-B Polish on n=5..85 + Re-propagate
    if not args.skip_polish:
        t0 = time.perf_counter()
        small_targets = [n for n in range(5, min(86, args.n_max + 1)) if math.isqrt(n) ** 2 != n]
        puzzles, pol_imps = polish_puzzles_parallel(puzzles, small_targets, workers=args.workers)
        puzzles, comp2 = propagate_composites(puzzles, n_max=args.n_max)
        rep_pol1 = score_submission(puzzles, n_max=args.n_max)
        print(
            f"[Stage 5] + Small-n L-BFGS-B + Re-prop    : {rep_pol1.total_score:.5f}  "
            f"(polished={pol_imps}, prop={comp2}, {time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

        # Stage 6: Polish Top-35 Remaining Highest-Cost Medium Puzzles (86 <= n <= 180)
        t0 = time.perf_counter()
        med_cands = [
            r.n for r in rep_pol1.worst_puzzles(200)
            if 86 <= r.n <= 180 and math.isqrt(r.n) ** 2 != r.n
        ][:35]
        if med_cands:
            puzzles, pol_med = polish_puzzles_parallel(puzzles, med_cands, workers=args.workers)
            puzzles, comp3 = propagate_composites(puzzles, n_max=args.n_max)
            rep_pol2 = score_submission(puzzles, n_max=args.n_max)
            print(
                f"[Stage 6] + Medium-n L-BFGS-B Polish      : {rep_pol2.total_score:.5f}  "
                f"(polished={pol_med}, prop={comp3}, {time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    # Stage 7: Full 1,000-Puzzle 60-Digit mpmath Certification
    if not args.skip_mpmath:
        t0 = time.perf_counter()
        puzzles, mp_pass, mp_fb = certify_all_mpmath(
            puzzles, baseline_puzzles, n_max=args.n_max, workers=args.workers, dps=60
        )
        print(
            f"[Stage 7] 60-digit mpmath certification   : {mp_pass}/{args.n_max} passed "
            f"(fallbacks={mp_fb}, {time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    final_report = score_submission(puzzles, n_max=args.n_max)
    sub_path = out_dir / "exp7_submission.csv"
    save_submission(puzzles, sub_path)

    improved_total = sum(
        1 for n in range(1, args.n_max + 1)
        if final_report.results[n].gain_over_baseline > 1e-6
    )

    print("\n" + "=" * 64)
    print("  EXP7 FINAL FULL 1,000-PUZZLE KAGGLE LEADERBOARD REPORT")
    print("=" * 64)
    print(f"Valid puzzles            : {final_report.valid_count} / {args.n_max}")
    print(f"Puzzles improved vs Grid : {improved_total} / {args.n_max}")
    print(f"Kaggle Baseline (Grid)   : {baseline_report.total_score:.5f}")
    print(f"Kaggle Exp7 Final Score  : {final_report.total_score:.5f}  "
          f"(delta: {final_report.total_score - baseline_report.total_score:+.5f})")
    print(f"Kaggle Top-5 Leaderboard : ~{KAGGLE_TOP5_REF:.5f}")
    print(f"Kaggle Top-1 Leaderboard : ~{KAGGLE_TOP1_REF:.5f}")
    print(f"Gap to World #1          : {final_report.total_score - KAGGLE_TOP1_REF:.5f}")
    print(f"Total runtime            : {time.perf_counter() - t_all:.1f}s")
    print(f"Saved submission         : {sub_path}")

    print("\nBreakdown by n range:")
    for lo, hi in [(1, 20), (21, 50), (51, 100), (101, 250), (251, 500), (501, 1000)]:
        b_s = sum(baseline_report.results[n].score for n in range(lo, min(hi, args.n_max) + 1))
        e_s = sum(final_report.results[n].score for n in range(lo, min(hi, args.n_max) + 1))
        print(f"  n={lo:4d}..{hi:4d}: base={b_s:8.5f} -> exp7={e_s:8.5f} (delta={e_s - b_s:+8.5f})")

    print("\nSelected classic small/medium puzzles:")
    for n in [5, 10, 11, 17, 18, 19, 26, 27, 28, 29, 37, 38, 50, 101, 257, 626, 962]:
        if n <= args.n_max:
            r = final_report.results[n]
            print(
                f"  n={n:4d}: L={r.side_length:8.5f} (grid={baseline_report.results[n].side_length:5.1f}, "
                f"sqrt={math.sqrt(n):7.4f}) score={r.score:.5f} (gain={r.gain_over_baseline:+.5f})"
            )

    profile_path = out_dir / "exp7_score_profile.png"
    plot_score_profile(final_report, baseline_report=baseline_report, save_path=profile_path)
    print(f"\nSaved score profile      : {profile_path}")

    gallery_ns = [n for n in [5, 10, 11, 17, 18, 26, 27, 50] if n <= args.n_max]
    gallery_path = out_dir / "exp7_gallery.png"
    plot_puzzle_grid(
        puzzles,
        gallery_ns,
        cols=4,
        show_indices=False,
        suptitle=f"Exp7 Portfolio Solver — Kaggle Score: {final_report.total_score:.5f} (Baseline: {baseline_report.total_score:.5f})",
        save_path=gallery_path,
    )
    print(f"Saved gallery            : {gallery_path}")


if __name__ == "__main__":
    main()
