# Experiment 7: Multi-Family Geometric Portfolio + Composites + L-BFGS-B Polish

## Method
1. **Family 1 (`oct45`)**: Off-center clipped 4-corner $45^\circ$ octagons and free-sliding diagonal bands (`w = 1..32`, `d = 1..46`, exact critical $L$ values, optimized disjoint 4-corner wall splits).
2. **Family 2 (`octgen`)**: General-angle clipped 4-corner rotated blocks (`15°–43.5°`).
3. **Family 3 (`l_strip`)**: L-shaped tilted $1 \times k$ strips + axis-aligned bulk.
4. **Composite Propagation**: Integer L-frames, 2×2 quad composites, and greedy boundary-square prefix trimming.
5. **Continuous Polish**: Parallelized symmetry-breaking + multi-start Fixed-Box `L-BFGS-B` on `n = 5..85` and top-35 highest-cost medium puzzles (`86..180`).
6. **Certification**: Full **1,000 / 1,000** 60-digit `mpmath` (`dps=60`) overlap verification.

## Results (Full 1,000 Puzzles, 8 CPU Workers, `970.9s` Total)

| Stage / Benchmark | Kaggle Score (lower is better) | Delta vs Baseline |
| --- | ---: | ---: |
| `sample_submission.csv` (Grid Baseline) | `28.93767` | `0.00000` |
| Stage 1: + 45-deg Octagon & Sliding Bands (`151.6s`) | `23.01679` | `-5.92088` |
| Stage 3: + L-Shaped Tilted Strips (`5.0s`) | `23.00260` | `-5.93507` |
| Stage 5: + Small-$n$ L-BFGS-B Polish | `22.99707` | `-5.94060` |
| **Stage 6–7: Exp7 Final (`1000/1000` `mp60` valid)** | **`22.99696`** | **`-5.94071`** |
| Kaggle Public Top-5 | `~22.05809` | `-6.87958` |
| Kaggle Public Top-1 | `~22.03995` | `-6.89772` |

- **Puzzles improved vs Grid**: `447 / 1000` (`oct45`: 422, `l_strip`: 25, plus L-BFGS-B polish improvements)
- **60-digit `mpmath` certification**: `1000 / 1000` passed (`fallbacks = 0`)
- **Gap to World #1**: **`0.95701`** (closed **86.1%** of the entire baseline-to-#1 gap on a laptop!)
