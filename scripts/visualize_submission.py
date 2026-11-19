"""CLI script to generate diagnostic visualizations for a submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from squares.io import load_submission
from squares.scoring import score_submission
from squares.visualize import plot_puzzle_grid, plot_score_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Squares in Squares submission.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/sample_submission.csv",
        help="Path to submission CSV (default: data/sample_submission.csv)",
    )
    parser.add_argument("--out-dir", default="outputs", help="Directory for output PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    puzzles = load_submission(args.csv_path)
    report = score_submission(puzzles)

    profile_path = out_dir / "score_profile.png"
    plot_score_profile(report, save_path=profile_path)
    print(f"Saved {profile_path}")

    worst_ns = [r.n for r in report.worst_puzzles(8)]
    worst_path = out_dir / "worst_puzzles.png"
    plot_puzzle_grid(
        puzzles,
        worst_ns,
        cols=4,
        show_indices=True,
        suptitle="Top 8 Highest-Cost Puzzles (Biggest Optimization Targets)",
        save_path=worst_path,
    )
    print(f"Saved {worst_path}")

    small_ns = [n for n in [2, 3, 5, 6, 7, 8, 10, 11] if n in puzzles]
    if small_ns:
        small_path = out_dir / "small_puzzles.png"
        plot_puzzle_grid(
            puzzles,
            small_ns,
            cols=4,
            show_indices=True,
            suptitle="Selected Small Puzzles",
            save_path=small_path,
        )
        print(f"Saved {small_path}")


if __name__ == "__main__":
    main()
