"""Unit tests for scoring and CSV I/O."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from squares.io import load_submission, save_submission, trivial_grid_score, trivial_grid_side
from squares.scoring import score_puzzle, score_submission, verify_puzzle_mpmath


def test_trivial_grid_helpers():
    assert trivial_grid_side(1) == 1
    assert trivial_grid_side(4) == 2
    assert trivial_grid_side(5) == 3
    assert trivial_grid_score(4) == pytest.approx(0.0)
    assert trivial_grid_score(5) == pytest.approx(3.0 / math.sqrt(5) - 1.0)


def test_score_puzzle_n5_rotated_beats_grid():
    # Classic n=5 packing: 4 axis-aligned corner squares + 1 45-deg rotated center square
    # Side L = 2 + sqrt(2)/2 ~= 2.70710678
    s = math.sqrt(2.0) / 2.0
    L = 2.0 + s
    squares = np.array([
        [0.5, 0.5, 0.0],
        [L - 0.5, 0.5, 0.0],
        [0.5, L - 0.5, 0.0],
        [L - 0.5, L - 0.5, 0.0],
        [L / 2.0, L / 2.0, math.pi / 4.0],
    ])
    # Add tiny margin so float64 rounding doesn't cause micro-overlap
    squares[:4, 0] += np.array([-1e-9, 1e-9, -1e-9, 1e-9])
    squares[:4, 1] += np.array([-1e-9, -1e-9, 1e-9, 1e-9])

    res = score_puzzle(5, squares)
    assert res.valid
    assert res.side_length < 2.71
    assert res.gain_over_baseline > 0.12


def test_invalid_puzzle_falls_back_to_baseline():
    # Overlapping n=2
    squares = np.array([
        [0.5, 0.5, 0.0],
        [0.8, 0.5, 0.0],
    ])
    res = score_puzzle(2, squares, fallback_on_invalid=True)
    assert not res.valid
    assert res.overlap_count == 1
    assert res.side_length == pytest.approx(2.0)
    assert res.score == pytest.approx(trivial_grid_score(2))


def test_submission_roundtrip_and_missing_fallback(tmp_path: Path):
    puzzles = {
        1: np.array([[0.5, 0.5, 0.0]]),
        2: np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]]),
    }
    csv_path = tmp_path / "sub.csv"
    save_submission(puzzles, csv_path)
    loaded = load_submission(csv_path)
    assert set(loaded.keys()) == {1, 2}
    np.testing.assert_allclose(loaded[2], puzzles[2])

    report = score_submission(loaded, n_max=3)
    assert report.valid_count == 2
    assert report.missing_ns == [3]
    expected_total = trivial_grid_score(1) + trivial_grid_score(2) + trivial_grid_score(3)
    assert report.total_score == pytest.approx(expected_total)


def test_verify_puzzle_mpmath():
    raw = [("0.5", "0.5", "0"), ("1.5", "0.5", "0")]
    valid, side, sc, overlaps = verify_puzzle_mpmath(2, raw, dps=60)
    assert valid
    assert side == pytest.approx(2.0)
    assert sc == pytest.approx(2.0 / math.sqrt(2) - 1.0)
    assert overlaps == []
