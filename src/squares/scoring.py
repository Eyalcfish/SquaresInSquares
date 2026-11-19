"""Kaggle Squares in Squares scoring and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import mpmath as mp
import numpy as np

from squares.geometry import (
    BoundingBox,
    compute_bounding_box,
    compute_side_length_mpmath,
    find_overlaps_float64,
    find_overlaps_mpmath,
)
from squares.io import trivial_grid_score, trivial_grid_side


@dataclass
class PuzzleResult:
    """Evaluation result for a single puzzle n."""

    n: int
    side_length: float
    score: float
    baseline_score: float
    valid: bool
    overlap_count: int
    overlaps: list[tuple[int, int, float]]
    max_penetration: float
    packing_density: float
    bbox: BoundingBox | None = None

    @property
    def gain_over_baseline(self) -> float:
        """Positive when this packing beats the trivial grid baseline."""
        return self.baseline_score - self.score


@dataclass
class SubmissionReport:
    """Aggregated evaluation report across n=1..n_max."""

    n_max: int
    total_score: float
    baseline_total_score: float
    valid_count: int
    invalid_ns: list[int]
    missing_ns: list[int]
    results: dict[int, PuzzleResult]

    @property
    def improvement(self) -> float:
        """Total score reduction compared to the trivial grid baseline."""
        return self.baseline_total_score - self.total_score

    def worst_puzzles(self, k: int = 20) -> list[PuzzleResult]:
        """Return the k puzzles with the highest (worst) individual scores."""
        return sorted(self.results.values(), key=lambda r: r.score, reverse=True)[:k]

    def best_gains(self, k: int = 20) -> list[PuzzleResult]:
        """Return the k puzzles with the largest score reduction vs baseline."""
        return sorted(
            self.results.values(),
            key=lambda r: r.gain_over_baseline,
            reverse=True,
        )[:k]

    def summary_text(self, top_k: int = 10) -> str:
        """Return a human-readable summary of the submission score."""
        lines = [
            f"=== Squares in Squares Evaluation (n=1..{self.n_max}) ===",
            f"Total Score          : {self.total_score:.8f}",
            f"Baseline (Grid) Score: {self.baseline_total_score:.8f}",
            f"Improvement vs Grid  : {self.improvement:+.8f}",
            f"Valid Puzzles        : {self.valid_count} / {self.n_max}",
        ]
        if self.invalid_ns:
            preview = ", ".join(str(x) for x in self.invalid_ns[:10])
            suffix = "..." if len(self.invalid_ns) > 10 else ""
            lines.append(f"Invalid (Overlapping): {len(self.invalid_ns)} [{preview}{suffix}]")
        if self.missing_ns:
            preview = ", ".join(str(x) for x in self.missing_ns[:10])
            suffix = "..." if len(self.missing_ns) > 10 else ""
            lines.append(f"Missing Puzzles      : {len(self.missing_ns)} [{preview}{suffix}]")

        lines.append(f"\nTop {top_k} Highest-Cost Puzzles (biggest optimization targets):")
        lines.append(f"{'n':>5}  {'L_n':>12}  {'sqrt(n)':>10}  {'score(n)':>12}  {'density':>8}  {'valid':>5}")
        for r in self.worst_puzzles(top_k):
            lines.append(
                f"{r.n:5d}  {r.side_length:12.6f}  {math.sqrt(r.n):10.6f}  "
                f"{r.score:12.6f}  {r.packing_density:8.4f}  {str(r.valid):>5}"
            )
        return "\n".join(lines)


def score_puzzle(
    n: int,
    squares: np.ndarray,
    tol: float = 1e-12,
    fallback_on_invalid: bool = True,
) -> PuzzleResult:
    """Score a single puzzle packing of n unit squares."""
    if len(squares) != n:
        raise ValueError(f"Expected {n} squares for puzzle n={n}, got {len(squares)}")

    bbox = compute_bounding_box(squares)
    overlaps, max_pen = find_overlaps_float64(squares, tol=tol)
    valid = len(overlaps) == 0

    baseline_L = float(trivial_grid_side(n))
    baseline_s = baseline_L / math.sqrt(n) - 1.0

    if valid or not fallback_on_invalid:
        side = bbox.side_length
        sc = side / math.sqrt(n) - 1.0
    else:
        side = baseline_L
        sc = baseline_s

    density = n / (side * side)
    return PuzzleResult(
        n=n,
        side_length=side,
        score=sc,
        baseline_score=baseline_s,
        valid=valid,
        overlap_count=len(overlaps),
        overlaps=overlaps,
        max_penetration=max_pen,
        packing_density=density,
        bbox=bbox,
    )


def score_submission(
    puzzles: Mapping[int, np.ndarray],
    n_max: int = 1000,
    tol: float = 1e-12,
    fallback_on_invalid: bool = True,
) -> SubmissionReport:
    """Score an entire submission across n=1..n_max."""
    results: dict[int, PuzzleResult] = {}
    invalid_ns: list[int] = []
    missing_ns: list[int] = []

    total_score = 0.0
    baseline_total = 0.0
    valid_count = 0

    for n in range(1, n_max + 1):
        b_side = float(trivial_grid_side(n))
        b_score = b_side / math.sqrt(n) - 1.0
        baseline_total += b_score

        if n not in puzzles:
            missing_ns.append(n)
            results[n] = PuzzleResult(
                n=n,
                side_length=b_side,
                score=b_score,
                baseline_score=b_score,
                valid=False,
                overlap_count=0,
                overlaps=[],
                max_penetration=float("-inf"),
                packing_density=n / (b_side * b_side),
                bbox=None,
            )
            total_score += b_score
            continue

        res = score_puzzle(
            n=n,
            squares=puzzles[n],
            tol=tol,
            fallback_on_invalid=fallback_on_invalid,
        )
        results[n] = res
        total_score += res.score
        if res.valid:
            valid_count += 1
        else:
            invalid_ns.append(n)

    return SubmissionReport(
        n_max=n_max,
        total_score=total_score,
        baseline_total_score=baseline_total,
        valid_count=valid_count,
        invalid_ns=invalid_ns,
        missing_ns=missing_ns,
        results=results,
    )


def verify_puzzle_mpmath(
    n: int,
    squares_raw: Sequence[tuple[str | float, str | float, str | float]],
    dps: int = 60,
) -> tuple[bool, float, float, list[tuple[int, int, str]]]:
    """Verify a puzzle using exact mpmath arithmetic at `dps` digits.

    Returns:
        (valid, side_length_float, score_float, overlaps)
    """
    with mp.workdps(dps):
        side_mp = compute_side_length_mpmath(squares_raw, dps=dps)
        score_mp = side_mp / mp.sqrt(n) - 1
        overlaps = find_overlaps_mpmath(squares_raw, dps=dps, tol="0")
        return len(overlaps) == 0, float(side_mp), float(score_mp), overlaps
