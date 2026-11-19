"""Smoke tests for visualization functions."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from squares.scoring import score_submission
from squares.visualize import plot_puzzle, plot_puzzle_grid, plot_score_profile


def test_visualizations_render_to_png(tmp_path: Path):
    puzzles = {
        1: np.array([[0.5, 0.5, 0.0]]),
        2: np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]]),
        3: np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.25]]),
    }
    p1 = tmp_path / "p3.png"
    plot_puzzle(3, puzzles[3], show_indices=True, save_path=p1)
    assert p1.exists() and p1.stat().st_size > 0

    p2 = tmp_path / "grid.png"
    plot_puzzle_grid(puzzles, [1, 2, 3], cols=3, save_path=p2)
    assert p2.exists() and p2.stat().st_size > 0

    report = score_submission(puzzles, n_max=3)
    p3 = tmp_path / "profile.png"
    plot_score_profile(report, top_k=3, save_path=p3)
    assert p3.exists() and p3.stat().st_size > 0
