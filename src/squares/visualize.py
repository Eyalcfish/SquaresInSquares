"""Visualization utilities for Squares in Squares packings and score diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from squares.geometry import compute_bounding_box, find_overlaps_float64, square_corners
from squares.scoring import SubmissionReport


def _angle_colors(angles: np.ndarray) -> np.ndarray:
    """Map rotation angles to [0, 1] modulo pi/2 (4-fold square symmetry)."""
    half_pi = math.pi / 2.0
    mod_a = np.mod(angles, half_pi) / half_pi
    return mod_a


def plot_puzzle(
    n: int,
    squares: np.ndarray,
    ax: plt.Axes | None = None,
    show_indices: bool = False,
    highlight_overlaps: bool = True,
    color_by: str = "angle",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a single n-square packing and its minimal axis-aligned bounding square."""
    if len(squares) != n:
        raise ValueError(f"Expected {n} squares, got {len(squares)}")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5), dpi=140)
        created_fig = True

    bbox = compute_bounding_box(squares)
    overlaps, _ = find_overlaps_float64(squares) if highlight_overlaps else ([], float("-inf"))
    bad_indices = set()
    for i, j, _ in overlaps:
        bad_indices.add(i)
        bad_indices.add(j)

    patches: list[Polygon] = []
    colors: list[float] = []
    if color_by == "angle":
        norm_vals = _angle_colors(squares[:, 2])
        cmap = matplotlib.colormaps["twilight"]
    elif color_by == "index":
        norm_vals = np.linspace(0.0, 1.0, max(n, 1))
        cmap = matplotlib.colormaps["viridis"]
    else:
        raise ValueError(f"Unknown color_by={color_by!r}, expected 'angle' or 'index'")

    for idx in range(n):
        x, y, angle = squares[idx]
        corners = square_corners(float(x), float(y), float(angle))
        patches.append(Polygon(corners, closed=True))
        colors.append(float(norm_vals[idx]))

    collection = PatchCollection(
        patches,
        cmap=cmap,
        alpha=0.82,
        edgecolor="#1f2937",
        linewidth=0.7 if n <= 100 else 0.35,
    )
    collection.set_array(np.array(colors))
    collection.set_clim(0.0, 1.0)
    ax.add_collection(collection)

    if bad_indices:
        bad_patches = [
            Polygon(
                square_corners(float(squares[idx, 0]), float(squares[idx, 1]), float(squares[idx, 2])),
                closed=True,
            )
            for idx in sorted(bad_indices)
        ]
        bad_col = PatchCollection(
            bad_patches,
            facecolor="#ef4444",
            alpha=0.65,
            edgecolor="#991b1b",
            linewidth=1.2,
        )
        ax.add_collection(bad_col)

    if show_indices and n <= 200:
        fontsize = 8 if n <= 36 else 6
        for idx in range(n):
            ax.text(
                float(squares[idx, 0]),
                float(squares[idx, 1]),
                str(idx),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="#111827",
                fontweight="bold",
            )

    # Draw enclosing square of side L_n centered on the bounding box
    cx = 0.5 * (bbox.xmin + bbox.xmax)
    cy = 0.5 * (bbox.ymin + bbox.ymax)
    half_L = 0.5 * bbox.side_length
    sq_x0 = cx - half_L
    sq_y0 = cy - half_L

    container = Rectangle(
        (sq_x0, sq_y0),
        bbox.side_length,
        bbox.side_length,
        fill=False,
        edgecolor="#0f172a",
        linewidth=1.8,
        linestyle="--",
    )
    ax.add_patch(container)

    pad = max(0.05 * bbox.side_length, 0.15)
    ax.set_xlim(sq_x0 - pad, sq_x0 + bbox.side_length + pad)
    ax.set_ylim(sq_y0 - pad, sq_y0 + bbox.side_length + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)

    score_n = bbox.side_length / math.sqrt(n) - 1.0
    density = n / (bbox.side_length * bbox.side_length)
    status = "VALID" if not bad_indices else f"OVERLAP ({len(overlaps)} pairs)"
    default_title = (
        f"n={n} | L={bbox.side_length:.5f} (sqrt={math.sqrt(n):.4f})\n"
        f"score={score_n:.5f} | density={density:.3f} | {status}"
    )
    ax.set_title(title or default_title, fontsize=10)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    if created_fig and save_path is not None:
        plt.close(ax.figure)

    return ax


def plot_puzzle_grid(
    puzzles: Mapping[int, np.ndarray],
    ns: Sequence[int],
    cols: int = 4,
    show_indices: bool = False,
    color_by: str = "angle",
    suptitle: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot multiple puzzles in a grid layout."""
    count = len(ns)
    if count == 0:
        raise ValueError("ns must be non-empty")

    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), dpi=130)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, n in enumerate(ns):
        plot_puzzle(
            n=n,
            squares=puzzles[n],
            ax=axes_flat[idx],
            show_indices=show_indices,
            color_by=color_by,
        )

    for idx in range(count, len(axes_flat)):
        axes_flat[idx].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return fig


def plot_score_profile(
    report: SubmissionReport,
    baseline_report: SubmissionReport | None = None,
    top_k: int = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a 4-panel diagnostic dashboard of the submission score profile."""
    ns = np.array(sorted(report.results.keys()), dtype=np.int64)
    scores = np.array([report.results[int(n)].score for n in ns], dtype=np.float64)
    densities = np.array([report.results[int(n)].packing_density for n in ns], dtype=np.float64)
    cum_scores = np.cumsum(scores)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)

    # Panel 1: score(n) vs n
    ax1 = axes[0, 0]
    if baseline_report is not None:
        b_scores = np.array([baseline_report.results[int(n)].score for n in ns], dtype=np.float64)
        ax1.plot(ns, b_scores, color="#94a3b8", linewidth=1.0, alpha=0.8, label="Baseline (grid)")
    ax1.plot(ns, scores, color="#2563eb", linewidth=1.1, label="Submission")
    ax1.set_title("Per-Puzzle Score: L_n / sqrt(n) - 1 (lower is better)")
    ax1.set_xlabel("n (number of unit squares)")
    ax1.set_ylabel("score(n)")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend()

    # Panel 2: Cumulative score vs n
    ax2 = axes[0, 1]
    if baseline_report is not None:
        b_cum = np.cumsum([baseline_report.results[int(n)].score for n in ns])
        ax2.plot(ns, b_cum, color="#94a3b8", linewidth=1.2, linestyle="--", label="Baseline cumulative")
    ax2.plot(ns, cum_scores, color="#059669", linewidth=1.5, label=f"Total = {report.total_score:.4f}")
    ax2.set_title("Cumulative Leaderboard Score Contribution")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Cumulative score")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend()

    # Panel 3: Packing density n / L_n^2 vs n
    ax3 = axes[1, 0]
    ax3.plot(ns, densities, color="#7c3aed", linewidth=1.0)
    ax3.axhline(1.0, color="#16a34a", linestyle="--", linewidth=1.0, alpha=0.7, label="Theoretical max (1.0)")
    ax3.set_title("Packing Density: n / L_n^2 (higher is better)")
    ax3.set_xlabel("n")
    ax3.set_ylabel("Density")
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend()

    # Panel 4: Top-K worst puzzles
    ax4 = axes[1, 1]
    worst = report.worst_puzzles(top_k)[::-1]
    labels = [f"n={r.n}" for r in worst]
    vals = [r.score for r in worst]
    bars = ax4.barh(labels, vals, color="#ea580c", alpha=0.85)
    ax4.set_title(f"Top {len(worst)} Highest-Cost Puzzles (Optimization Targets)")
    ax4.set_xlabel("score(n)")
    ax4.grid(True, axis="x", linestyle=":", alpha=0.4)
    for bar, r in zip(bars, worst):
        ax4.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{r.score:.4f} (L={r.side_length:.2f})",
            va="center",
            fontsize=8,
        )
    ax4.set_xlim(0, max(vals) * 1.28 if vals else 1.0)

    fig.suptitle(
        f"Squares in Squares Score Profile — Total: {report.total_score:.6f} "
        f"(Valid: {report.valid_count}/{report.n_max})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return fig
