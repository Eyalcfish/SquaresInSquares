# Experiment 2: Adaptive-Expansion SA + Adam + L-BFGS-B

## Method
- **Adaptive Initial Expansion**: Starts at `1.08x` and steps through `(1.08, 1.22, 1.40, 1.65, 1.95)` whenever an attempt fails to improve upon the baseline.
- **Compaction**: Annealed Adam (`450` steps) followed by `L-BFGS-B` (`180` maxiter) with `tau=0.001`, `overlap_weight=2500.0`, and `margin=1e-6`.
- **Validation**: 60-digit `mpmath` (`dps=60`) verification on every puzzle.

## Results (20 Target Puzzles, 167.38s Total)

| Metric | Value |
| --- | ---: |
| Puzzles tested | 20 |
| Puzzles improved (`> 1e-6`) | 1 / 20 (`n=5`, `exp=1.22`, `att=2`) |
| Subset baseline score | `3.44079354` |
| Subset Exp2 score | `3.36678456` |
| **Subset improvement** | **`+0.07400898`** |
| **Kaggle Baseline (Grid)** | **`28.93767`** |
| **Kaggle Exp1 Score** | **`28.85491`** |
| **Kaggle Exp2 Score** | **`28.86366`** |
| **Kaggle Top-5 Leaderboard** | **`~22.05809`** |
| **Kaggle Top-1 Leaderboard** | **`~22.03995`** |
| **Gap to Top-1** | **`6.82371`** |

## Diagnosis: Why Generic SA + GD Jams
1. **Jamming in random configurations**: When squares are expanded to `1.4x–1.95x` with random angles and compacted, they jam into disordered packings whose bounding square $L$ is *larger* than the $k \times k$ grid, rather than finding the structured tilted-band / staggered-pocket packings that beat the grid.
2. **Strict `1e-12` rejection during L-BFGS-B**: Any micro-overlap from `L-BFGS-B` causes `find_overlaps_float64` to reject the polished state instead of applying a tiny `1 + eps` inflation repair.
3. **Path to `~22.04`**: Closing the `6.82`-point gap across $n=1\dots 1000$ requires **structured geometric / MILP / tilted-grid / billiard-compression ("shrink-and-push" inside a fixed $L$)** solvers rather than unconstrained random SA from a grid.
