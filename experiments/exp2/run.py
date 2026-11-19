"""Run Experiment 2 (Adaptive-Expansion SA + Adam + L-BFGS-B) and report Kaggle score."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.exp2.solver import Exp2Config, solve_puzzle_exp2
from squares.io import load_submission, save_submission
from squares.scoring import score_submission, verify_puzzle_mpmath
from squares.visualize import plot_puzzle_grid


DEFAULT_TARGET_NS = [
    2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 37, 38
]

KAGGLE_TOP1_REF = 22.03995
KAGGLE_TOP5_REF = 22.05809


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp2: Adaptive-Expansion SA + Adam + L-BFGS-B.")
    parser.add_argument("--csv", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="experiments/exp2/outputs")
    parser.add_argument("--sa-steps", type=int, default=2000)
    parser.add_argument("--gd-steps", type=int, default=450)
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

    print(
        f"Running Exp2 (adaptive expansion {Exp2Config().expansion_schedule}, "
        f"SA={args.sa_steps}, Adam={args.gd_steps}+L-BFGS-B) on {len(target_ns)} puzzles...\n"
    )
    print(
        f"{'n':>4}  {'base_L':>10}  {'exp2_L':>10}  {'base_sc':>10}  {'exp2_sc':>10}  "
        f"{'delta_sc':>10}  {'exp':>5}  {'att':>3}  {'mp60':>5}  {'time(s)':>7}"
    )

    cfg = Exp2Config(sa_steps=args.sa_steps, gd_steps=args.gd_steps)
    for n in target_ns:
        res = solve_puzzle_exp2(n, puzzles[n], cfg=cfg)
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
            f"{delta:+10.6f}  {res.winning_expansion:5.2f}  {res.attempts_used:3d}  "
            f"{str(mp_valid):>5}  {res.elapsed_sec:7.2f}"
        )

    total_elapsed = time.perf_counter() - t_all
    sub_path = out_dir / "exp2_submission.csv"
    save_submission(updated_puzzles, sub_path)
    exp2_report = score_submission(updated_puzzles)

    subset_base = sum(r.baseline_score for r in results)
    subset_exp2 = sum(r.score for r in results)
    improved_count = sum(1 for r in results if r.baseline_score - r.score > 1e-6)

    print("\n=== Exp2 Numerical Summary & Kaggle Leaderboard Comparison ===")
    print(f"Puzzles tested           : {len(results)}")
    print(f"Puzzles improved (>1e-6) : {improved_count} / {len(results)}")
    print(f"Subset baseline score    : {subset_base:.8f}")
    print(f"Subset exp2 score        : {subset_exp2:.8f}")
    print(f"Subset improvement       : {subset_base - subset_exp2:+.8f}")
    print("-" * 58)
    print(f"Kaggle Baseline (Grid)   : {baseline_report.total_score:.5f}")
    print(f"Kaggle Exp2 Score        : {exp2_report.total_score:.5f}  "
          f"(delta vs baseline: {exp2_report.total_score - baseline_report.total_score:+.5f})")
    print(f"Kaggle Top-5 Leaderboard : ~{KAGGLE_TOP5_REF:.5f}")
    print(f"Kaggle Top-1 Leaderboard : ~{KAGGLE_TOP1_REF:.5f}")
    print(f"Gap to Top-1             : {exp2_report.total_score - KAGGLE_TOP1_REF:.5f} "
          f"(across remaining 1000 puzzles)")
    print(f"Total runtime            : {total_elapsed:.2f}s")
    print(f"Saved submission         : {sub_path}")

    top_improved = sorted(
        results, key=lambda r: r.baseline_score - r.score, reverse=True
    )[:8]
    plot_ns = [r.n for r in top_improved]
    plot_path = out_dir / "exp2_top_improved.png"
    plot_puzzle_grid(
        updated_puzzles,
        plot_ns,
        cols=4,
        show_indices=True,
        suptitle="Exp2 (Adaptive Expansion + SA + Adam + L-BFGS-B) — Top 8 Packings",
        save_path=plot_path,
    )
    print(f"Saved visualization      : {plot_path}")


if __name__ == "__main__":
    main()
