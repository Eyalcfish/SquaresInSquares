# Experiment 1: Simulated Annealing + Gradient Descent Hybrid

## Method
- **Parameters**: $W \in \mathbb{R}^{n \times 3}$ representing $(x_i, y_i, \theta_i)$ for each unit square.
- **Stage 1 (SA Exploration)**: Metropolis simulated annealing (`2500` steps, geometric temperature decay `0.25 -> 0.002`) proposing single-square translations/rotations, centroid pulls, and global shrinks.
- **Stage 2 (GD Compaction)**: Whenever SA discovers a valid (zero-overlap) state, Adam (`600` steps, cosine LR `0.02 -> 0`) minimizes a smooth LogSumExp bounding-square side length plus a quadratic SAT penetration penalty (`weight=50.0`, `margin=1e-6`).
- **Validation**: Every improved solution is verified with 60-digit `mpmath` (`dps=60`) SAT overlap checks.

## Results (20 Target Puzzles, 2 Restarts Each, 65.15s Total)

| Metric | Value |
| --- | ---: |
| Puzzles tested | 20 |
| Puzzles improved (`> 1e-6`) | 1 / 20 (`n=5`) |
| Subset baseline score | `3.44079354` |
| Subset Exp1 score | `3.35803264` |
| **Subset improvement** | **`+0.08276090`** |
| Full 1000 baseline score | `28.93766992` |
| Full 1000 Exp1 score | `28.85490902` |
| **Full 1000 improvement** | **`+0.08276090`** |

### Notable Puzzle Result
- **`n = 5`**: `L = 3.000000 -> 2.814941`, `score = 0.341641 -> 0.258880` (`+0.082761`, `mpmath` 60-digit valid).

## Key Takeaways for Next Experiments
1. `n=2` and `n=3` are already at their proven optimal $L=2.0$, so no further gain is possible there.
2. For `n >= 10`, starting SA from a `1.05x` expanded grid is too cramped: for `n=37, 38`, SA found `0` valid states (`sa_ok=0`). Starting from a looser box (e.g. `1.2x`–`1.4x` or random placement inside a generous box) and using **L-BFGS / multi-scale penalty continuation** during GD should unlock improvements on `n=10, 11, 17, 18, 26, 27, 37, 38`.
3. `n=5` can be pushed further toward the known optimum $L_5 = 2 + \sqrt{2}/2 \approx 2.707107$ (`score ≈ 0.210657`) with tighter `smooth_tau` annealing and L-BFGS polishing.
