"""Submission CSV I/O and validation helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EXPECTED_HEADER = ["id", "n", "square_id", "x", "y", "angle"]


def trivial_grid_side(n: int) -> int:
    """Return the side length of the trivial axis-aligned ceil(sqrt(n)) grid packing."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.isqrt(n - 1) + 1


def trivial_grid_score(n: int) -> float:
    """Return the baseline score (L_n / sqrt(n) - 1) of the trivial grid packing."""
    return trivial_grid_side(n) / math.sqrt(n) - 1.0


def load_submission(
    path: str | Path,
) -> dict[int, np.ndarray]:
    """Load a Kaggle Squares in Squares submission CSV into {n: (n, 3) float64 array}.

    Rows for each `n` are ordered by `square_id`.
    """
    path = Path(path)
    buckets: dict[int, list[tuple[int, float, float, float]]] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV file: {path}")
        if [h.strip() for h in header] != EXPECTED_HEADER:
            raise ValueError(f"Unexpected header {header}, expected {EXPECTED_HEADER}")

        for row in reader:
            if not row:
                continue
            n = int(row[1])
            sq_id = int(row[2])
            x = float(row[3])
            y = float(row[4])
            angle = float(row[5])
            buckets.setdefault(n, []).append((sq_id, x, y, angle))

    puzzles: dict[int, np.ndarray] = {}
    for n, items in buckets.items():
        if len(items) != n:
            raise ValueError(f"Puzzle n={n} has {len(items)} rows, expected {n}")
        items.sort(key=lambda t: t[0])
        expected_ids = list(range(n))
        actual_ids = [t[0] for t in items]
        if actual_ids != expected_ids:
            raise ValueError(f"Puzzle n={n} has non-contiguous square_ids")
        arr = np.array([(t[1], t[2], t[3]) for t in items], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Puzzle n={n} contains NaN or Inf values")
        puzzles[n] = arr

    return puzzles


def load_submission_raw(
    path: str | Path,
    only_n: set[int] | None = None,
) -> dict[int, list[tuple[str, str, str]]]:
    """Load raw string (x, y, angle) tuples for high-precision mpmath verification."""
    path = Path(path)
    buckets: dict[int, list[tuple[int, str, str, str]]] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != EXPECTED_HEADER:
            raise ValueError(f"Invalid header in {path}: {header}")

        for row in reader:
            if not row:
                continue
            n = int(row[1])
            if only_n is not None and n not in only_n:
                continue
            sq_id = int(row[2])
            buckets.setdefault(n, []).append((sq_id, row[3].strip(), row[4].strip(), row[5].strip()))

    result: dict[int, list[tuple[str, str, str]]] = {}
    for n, items in buckets.items():
        items.sort(key=lambda t: t[0])
        result[n] = [(t[1], t[2], t[3]) for t in items]
    return result


def save_submission(
    puzzles: Mapping[int, np.ndarray | Sequence[tuple[str | float, str | float, str | float]]],
    path: str | Path,
    float_precision: int = 17,
) -> Path:
    """Save puzzles {n: squares} to a Kaggle-formatted submission CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = f"{{:.{float_precision}g}}"
    row_id = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADER)
        for n in sorted(puzzles.keys()):
            squares = puzzles[n]
            if len(squares) != n:
                raise ValueError(f"Puzzle n={n} has {len(squares)} squares, expected {n}")
            for sq_id in range(n):
                x, y, angle = squares[sq_id]
                x_s = x if isinstance(x, str) else fmt.format(float(x))
                y_s = y if isinstance(y, str) else fmt.format(float(y))
                a_s = angle if isinstance(angle, str) else fmt.format(float(angle))
                writer.writerow([row_id, n, sq_id, x_s, y_s, a_s])
                row_id += 1

    return path
