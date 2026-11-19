"""Run Experiment 4 across ALL 1,000 puzzles in parallel and report Kaggle score."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from experiments.exp4.solver import Exp4Config, Exp4Result, solve_puzzle_exp4
from squares.io import load_submission, save_submission, trivial_grid_side
from squares.scoring import score_submission, verify_puzzle_mpmath
from squares.visualize import plot_puzzle_grid, plot_score_profile


KAGGLE_TOP1_REF = 22.03995
KAGGLE_TOP5_REF = 22.05809


def _worker(args: tuple[int, np.ndarray, float, Exp4Config]) -> Exp4Result:
    n, init_sq, base_side, cfg = args
    return solve_puzzle_exp4(n, init_sq, base_side, cfg=cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp4 across all 1,000 puzzles.")
    parser.add_argument(
        "--csv",
        default="experiments/exp3/outputs/exp3_submission.csv",
        help="Starting submission CSV (warm-started from exp3)",
    )
    parser.add_argument("--baseline-csv", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="experiments/exp4/outputs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_puzzles = load_submission(args.baseline_csv)
    baseline_report = score_submission(baseline_puzzles, n_max=args.n_max)

    start_csv = Path(args.csv)
    puzzles = load_submission(start_csv) if start_csv.exists() else dict(baseline_puzzles)
    updated_puzzles = dict(puzzles)

    cfg = Exp4Config()
    tasks = [
        (n, puzzles[n], float(trivial_grid_side(n)), cfg)
        for n in range(1, args.n_max + 1)
    ]

    t_all = time.perf_counter()
    print(f"Running Exp4 across n=1..{args.n_max} using {args.workers} workers...")

    results: dict[int, Exp4Result] = {}
    done_cnt = 0
    improved_cnt = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_worker, tasks, chunksize=4):
            results[res.n] = res
            done_cnt += 1
            if res.valid and res.side_length < res.baseline_side - 1e-6:
                updated_puzzles[res.n] = res.squares
                improved_cnt += 1
            if done_cnt % 100 == 0 or done_cnt == args.n_max:
                elapsed = time.perf_counter() - t_all
                print(
                    f"  Progress: {done_cnt:4d}/{args.n_max} puzzles | "
                    f"improved={improved_cnt:3d} | elapsed={elapsed:6.1f}s",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - t_all

    # Spot-verify top-20 most improved puzzles with 60-digit mpmath
    top_improved = sorted(
        results.values(),
        key=lambda r: r.baseline_score - r.score,
        reverse=True,
    )
    print("\nTop 20 Most Improved Puzzles (with 60-digit mpmath verification):")
    print(f"{'n':>5}  {'base_L':>10}  {'exp4_L':>10}  {'base_sc':>10}  {'exp4_sc':>10}  {'delta_sc':>10}  {'mp60':>5}")
    for r in top_improved[:20]:
        raw_sq = [
            (f"{r.squares[i, 0]:.17g}", f"{r.squares[i, 1]:.17g}", f"{r.squares[i, 2]:.17g}")
            for i in range(r.n)
        ]
        mp_valid, _, _, _ = verify_puzzle_mpmath(r.n, raw_sq, dps=60)
        delta = r.baseline_score - r.score
        print(
            f"{r.n:5d}  {r.baseline_side:10.6f}  {r.side_length:10.6f}  "
            f"{r.baseline_score:10.6f}  {r.score:10.6f}  {delta:+10.6f}  {str(mp_valid):>5}"
        )

    sub_path = out_dir / "exp4_submission.csv"
    save_submission(updated_puzzles, sub_path)
    exp4_report = score_submission(updated_puzzles, n_max=args.n_max)

    print("\n=== Exp4 Full 1,000-Puzzle Kaggle Leaderboard Report ===")
    print(f"Total puzzles evaluated  : {args.n_max}")
    print(f"Valid puzzles            : {exp4_report.valid_count} / {args.n_max}")
    print(f"Puzzles improved vs Grid : {improved_cnt} / {args.n_max}")
    print("-" * 58)
    print(f"Kaggle Baseline (Grid)   : {baseline_report.total_score:.5f}")
    print(f"Kaggle Exp4 Score        : {exp4_report.total_score:.5f}  "
          f"(delta vs baseline: {exp4_report.total_score - baseline_report.total_score:+.5f})")
    print(f"Kaggle Top-5 Leaderboard : ~{KAGGLE_TOP5_REF:.5f}")
    print(f"Kaggle Top-1 Leaderboard : ~{KAGGLE_TOP1_REF:.5f}")
    print(f"Gap to Top-1             : {exp4_report.total_score - KAGGLE_TOP1_REF:.5f}")
    print(f"Total runtime            : {total_elapsed:.2f}s")
    print(f"Saved submission         : {sub_path}")

    profile_path = out_dir / "exp4_score_profile.png"
    plot_score_profile(exp4_report, baseline_report=baseline_report, save_path=profile_path)
    print(f"Saved score profile      : {profile_path}")

    plot_ns = [r.n for r in top_improved[:8]]
    grid_path = out_dir / "exp4_top_improved.png"
    plot_puzzle_grid(
        updated_puzzles,
        plot_ns,
        cols=4,
        show_indices=False,
        suptitle="Exp4 (Full 1,000-Puzzle Fixed-Box L-BFGS-B) — Top 8 Most Improved",
        save_path=grid_path,
    )
    print(f"Saved puzzle grid        : {grid_path}")


if __name__ == "__main__":
    main()
