"""Run Experiment 3 (Fixed-Box L-BFGS-B Compression) and report Kaggle score."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.exp3.solver import Exp3Config, solve_puzzle_exp3
from squares.io import load_submission, save_submission
from squares.scoring import score_submission, verify_puzzle_mpmath
from squares.visualize import plot_puzzle_grid


DEFAULT_TARGET_NS = [
    2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 37, 38
]

KAGGLE_TOP1_REF = 22.03995
KAGGLE_TOP5_REF = 22.05809


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp3: Fixed-Box L-BFGS-B Compression.")
    parser.add_argument("--csv", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="experiments/exp3/outputs")
    parser.add_argument("--ns", type=int, nargs="*", default=DEFAULT_TARGET_NS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    puzzles = load_submission(args.csv)
    baseline_report = score_submission(puzzles)
    updated_puzzles = dict(puzzles)

    target_ns = sorted(set(args.ns))
    results = []
    t_all = time.perf_counter()

    cfg = Exp3Config()
    print(f"Running Exp3 (Fixed-Box L-BFGS-B + Binary Search) on {len(target_ns)} puzzles...\n")
    print(
        f"{'n':>4}  {'base_L':>10}  {'exp3_L':>10}  {'base_sc':>10}  {'exp3_sc':>10}  "
        f"{'delta_sc':>10}  {'mp60':>5}  {'time(s)':>7}"
    )

    for n in target_ns:
        res = solve_puzzle_exp3(n, puzzles[n], cfg=cfg)
        raw_sq = [
            (f"{res.squares[i, 0]:.17g}", f"{res.squares[i, 1]:.17g}", f"{res.squares[i, 2]:.17g}")
            for i in range(n)
        ]
        mp_valid, mp_L, _, _ = verify_puzzle_mpmath(n, raw_sq, dps=60)
        if mp_valid and mp_L < res.baseline_side:
            updated_puzzles[n] = res.squares

        results.append(res)
        delta = res.baseline_score - res.score
        print(
            f"{n:4d}  {res.baseline_side:10.6f}  {res.side_length:10.6f}  "
            f"{res.baseline_score:10.6f}  {res.score:10.6f}  "
            f"{delta:+10.6f}  {str(mp_valid):>5}  {res.elapsed_sec:7.2f}"
        )

    total_elapsed = time.perf_counter() - t_all
    sub_path = out_dir / "exp3_submission.csv"
    save_submission(updated_puzzles, sub_path)
    exp3_report = score_submission(updated_puzzles)

    subset_base = sum(r.baseline_score for r in results)
    subset_exp3 = sum(r.score for r in results)
    improved_count = sum(1 for r in results if r.baseline_score - r.score > 1e-6)

    print("\n=== Exp3 Numerical Summary & Kaggle Leaderboard Comparison ===")
    print(f"Puzzles tested           : {len(results)}")
    print(f"Puzzles improved (>1e-6) : {improved_count} / {len(results)}")
    print(f"Subset baseline score    : {subset_base:.8f}")
    print(f"Subset exp3 score        : {subset_exp3:.8f}")
    print(f"Subset improvement       : {subset_base - subset_exp3:+.8f}")
    print("-" * 58)
    print(f"Kaggle Baseline (Grid)   : {baseline_report.total_score:.5f}")
    print(f"Kaggle Exp3 Score        : {exp3_report.total_score:.5f}  "
          f"(delta vs baseline: {exp3_report.total_score - baseline_report.total_score:+.5f})")
    print(f"Kaggle Top-5 Leaderboard : ~{KAGGLE_TOP5_REF:.5f}")
    print(f"Kaggle Top-1 Leaderboard : ~{KAGGLE_TOP1_REF:.5f}")
    print(f"Gap to Top-1             : {exp3_report.total_score - KAGGLE_TOP1_REF:.5f}")
    print(f"Total runtime            : {total_elapsed:.2f}s")
    print(f"Saved submission         : {sub_path}")

    top_improved = sorted(
        results, key=lambda r: r.baseline_score - r.score, reverse=True
    )[:8]
    plot_ns = [r.n for r in top_improved]
    plot_path = out_dir / "exp3_top_improved.png"
    plot_puzzle_grid(
        updated_puzzles,
        plot_ns,
        cols=4,
        show_indices=True,
        suptitle="Exp3 (Fixed-Box L-BFGS-B Compression) — Top 8 Most Improved Packings",
        save_path=plot_path,
    )
    print(f"Saved visualization      : {plot_path}")


if __name__ == "__main__":
    main()
