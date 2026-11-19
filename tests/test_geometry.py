"""Unit tests for geometry and SAT overlap detection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from squares.geometry import (
    compute_bounding_box,
    compute_side_length_mpmath,
    find_overlaps_float64,
    find_overlaps_mpmath,
    sat_penetration_pair,
    square_corners,
)


def test_square_corners_rotated_45():
    corners = square_corners(0.0, 0.0, math.pi / 4)
    expected_r = math.sqrt(2.0) / 2.0
    xs = sorted(corners[:, 0])
    ys = sorted(corners[:, 1])
    assert pytest.approx(xs[0], abs=1e-12) == -expected_r
    assert pytest.approx(xs[-1], abs=1e-12) == expected_r
    assert pytest.approx(ys[0], abs=1e-12) == -expected_r
    assert pytest.approx(ys[-1], abs=1e-12) == expected_r


def test_bounding_box_rotated_single_square():
    sq = np.array([[1.0, 2.0, math.pi / 4]])
    bbox = compute_bounding_box(sq)
    assert pytest.approx(bbox.side_length, abs=1e-12) == math.sqrt(2.0)


def test_sat_touching_axis_aligned_no_overlap():
    squares = np.array([
        [0.5, 0.5, 0.0],
        [1.5, 0.5, 0.0],
        [0.5, 1.5, 0.0],
        [1.5, 1.5, 0.0],
    ])
    overlaps, max_pen = find_overlaps_float64(squares)
    assert overlaps == []
    assert max_pen <= 1e-12


def test_sat_overlapping_axis_aligned():
    squares = np.array([
        [0.5, 0.5, 0.0],
        [1.4, 0.5, 0.0],
    ])
    overlaps, max_pen = find_overlaps_float64(squares)
    assert len(overlaps) == 1
    assert overlaps[0][0] == 0 and overlaps[0][1] == 1
    assert pytest.approx(overlaps[0][2], abs=1e-12) == 0.1
    assert pytest.approx(max_pen, abs=1e-12) == 0.1


def test_sat_rotated_45_touching_and_overlapping():
    # Two 45-degree squares on x-axis touch at distance sqrt(2)
    d = math.sqrt(2.0)
    pen_touch = sat_penetration_pair((0.0, 0.0, math.pi / 4), (d, 0.0, math.pi / 4))
    assert pytest.approx(pen_touch, abs=1e-12) == 0.0

    pen_overlap = sat_penetration_pair((0.0, 0.0, math.pi / 4), (d - 0.05, 0.0, math.pi / 4))
    assert pytest.approx(pen_overlap, abs=1e-12) == 0.05 / math.sqrt(2.0)


def test_mpmath_catches_tiny_overlap():
    raw = [
        ("0.5", "0.5", "0"),
        ("1.499999999999999", "0.5", "0"),
    ]
    overlaps = find_overlaps_mpmath(raw, dps=60, tol="0")
    assert len(overlaps) == 1
    side = compute_side_length_mpmath(raw, dps=60)
    assert float(side) == pytest.approx(2.0, abs=1e-12)
