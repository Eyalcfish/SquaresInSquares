"""CLI script to evaluate a Squares in Squares submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from squares.io import load_submission, load_submission_raw
from squares.scoring import score_submission, verify_puzzle_mpmath


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a Squares in Squares submission CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/sample_submission.csv",
        help="Path to submission CSV (default: data/sample_submission.csv)",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Show top-k highest-cost puzzles")
    parser.add_argument(
        "--mpmath-ns",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of puzzle n values to verify with 60-digit mpmath",
    )
    parser.add_argument("--dps", type=int, default=60, help="Decimal precision for mpmath verification")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    puzzles = load_submission(csv_path)
    report = score_submission(puzzles)
    print(report.summary_text(top_k=args.top_k))

    if args.mpmath_ns:
        target_ns = set(args.mpmath_ns)
        raw = load_submission_raw(csv_path, only_n=target_ns)
        print(f"\n=== Exact mpmath verification (dps={args.dps}) ===")
        for n in sorted(target_ns):
            valid, side, sc, overlaps = verify_puzzle_mpmath(n, raw[n], dps=args.dps)
            print(f"n={n:4d}: valid={valid}, L={side:.12f}, score={sc:.12f}, overlaps={len(overlaps)}")


if __name__ == "__main__":
    main()
