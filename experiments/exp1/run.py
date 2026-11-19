"""Run Experiment 1 (SA + GD Hybrid) and generate a numerical report."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.exp1.solver import Exp1Config, Exp1Result, solve_puzzle_exp1
from squares.io import load_submission, save_submission
from squares.scoring import score_submission, verify_puzzle_mpmath
from squares.visualize import plot_puzzle_grid


DEFAULT_TARGET_NS = [
    2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 37, 38
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp1: SA + GD hybrid optimizer.")
    parser.add_argument("--csv", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="experiments/exp1/outputs")
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--sa-steps", type=int, default=2500)
    parser.add_argument("--gd-steps", type=int, default=600)
    parser.add_argument(
        "--ns",
        type=int,
        nargs="*",
        default=DEFAULT_TARGET_NS,
        help="List of n values to optimize",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    puzzles = load_submission(args.csv)
    baseline_report = score_submission(puzzles)

    target_ns = sorted(set(args.ns))
    results: list[Exp1Result] = []
    updated_puzzles = dict(puzzles)

    t_all = time.perf_counter()
    print(f"Running Exp1 (SA={args.sa_steps} steps, GD={args.gd_steps} steps, restarts={args.restarts}) "
          f"on {len(target_ns)} puzzles...\n")
    print(
        f"{'n':>4}  {'base_L':>10}  {'exp1_L':>10}  {'base_sc':>10}  {'exp1_sc':>10}  "
        f"{'delta_sc':>10}  {'mp60':>5}  {'sa_ok':>5}  {'gd':>3}  {'time(s)':>7}"
    )

    for n in target_ns:
        best_res: Exp1Result | None = None
        for r in range(args.restarts):
            cfg = Exp1Config(
                sa_steps=args.sa_steps,
                gd_steps=args.gd_steps,
                seed=42 + r * 7919,
            )
            res = solve_puzzle_exp1(n, puzzles[n], cfg=cfg)
            if best_res is None or (res.valid and res.side_length < best_res.side_length):
                best_res = res

        assert best_res is not None
        # Verify with 60-digit mpmath
        raw_sq = [
            (f"{best_res.squares[i, 0]:.17g}", f"{best_res.squares[i, 1]:.17g}", f"{best_res.squares[i, 2]:.17g}")
            for i in range(n)
        ]
        mp_valid, mp_L, mp_sc, _ = verify_puzzle_mpmath(n, raw_sq, dps=60)
        if mp_valid and mp_L < best_res.baseline_side:
            updated_puzzles[n] = best_res.squares

        results.append(best_res)
        delta = best_res.baseline_score - best_res.score
        print(
            f"{n:4d}  {best_res.baseline_side:10.6f}  {best_res.side_length:10.6f}  "
            f"{best_res.baseline_score:10.6f}  {best_res.score:10.6f}  "
            f"{delta:+10.6f}  {str(mp_valid):>5}  {best_res.sa_valid_found:5d}  "
            f"{best_res.gd_runs:3d}  {best_res.elapsed_sec:7.2f}"
        )

    total_elapsed = time.perf_counter() - t_all
    sub_path = out_dir / "exp1_submission.csv"
    save_submission(updated_puzzles, sub_path)
    exp1_report = score_submission(updated_puzzles)

    subset_base = sum(r.baseline_score for r in results)
    subset_exp1 = sum(r.score for r in results)
    improved_count = sum(1 for r in results if r.baseline_score - r.score > 1e-6)

    print("\n=== Exp1 Numerical Summary ===")
    print(f"Puzzles tested           : {len(results)}")
    print(f"Puzzles improved (>1e-6) : {improved_count} / {len(results)}")
    print(f"Subset baseline score    : {subset_base:.8f}")
    print(f"Subset exp1 score        : {subset_exp1:.8f}")
    print(f"Subset improvement       : {subset_base - subset_exp1:+.8f}")
    print(f"Full 1000 baseline score : {baseline_report.total_score:.8f}")
    print(f"Full 1000 exp1 score     : {exp1_report.total_score:.8f}")
    print(f"Full 1000 improvement    : {baseline_report.total_score - exp1_report.total_score:+.8f}")
    print(f"Total runtime            : {total_elapsed:.2f}s")
    print(f"Saved submission         : {sub_path}")

    # Plot top-8 most improved puzzles
    top_improved = sorted(
        results, key=lambda r: r.baseline_score - r.score, reverse=True
    )[:8]
    plot_ns = [r.n for r in top_improved]
    plot_path = out_dir / "exp1_top_improved.png"
    plot_puzzle_grid(
        updated_puzzles,
        plot_ns,
        cols=4,
        show_indices=True,
        suptitle="Exp1 (SA + GD Hybrid) — Top 8 Most Improved Packings",
        save_path=plot_path,
    )
    print(f"Saved visualization      : {plot_path}")


if __name__ == "__main__":
    main()
