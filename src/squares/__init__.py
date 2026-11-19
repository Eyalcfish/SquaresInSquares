"""Squares in Squares scoring, geometry, I/O, and visualization package."""

from squares.geometry import (
    BoundingBox,
    compute_bounding_box,
    compute_side_length_mpmath,
    find_overlaps_float64,
    find_overlaps_mpmath,
    sat_penetration_pair,
    square_corners,
)
from squares.io import (
    load_submission,
    load_submission_raw,
    save_submission,
    trivial_grid_score,
    trivial_grid_side,
)
from squares.scoring import (
    PuzzleResult,
    SubmissionReport,
    score_puzzle,
    score_submission,
    verify_puzzle_mpmath,
)
from squares.visualize import (
    plot_puzzle,
    plot_puzzle_grid,
    plot_score_profile,
)

__all__ = [
    "BoundingBox",
    "PuzzleResult",
    "SubmissionReport",
    "compute_bounding_box",
    "compute_side_length_mpmath",
    "find_overlaps_float64",
    "find_overlaps_mpmath",
    "load_submission",
    "load_submission_raw",
    "plot_puzzle",
    "plot_puzzle_grid",
    "plot_score_profile",
    "sat_penetration_pair",
    "save_submission",
    "score_puzzle",
    "score_submission",
    "square_corners",
    "trivial_grid_score",
    "trivial_grid_side",
    "verify_puzzle_mpmath",
]
