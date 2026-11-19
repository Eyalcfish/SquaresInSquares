# Experiment 8 Plan (For Home CPU/GPU Machine)

## Current State (`exp7` Baseline)
- **Kaggle Score**: **`22.99696`** (`1000 / 1000` valid under 60-digit `mpmath`)
- **Top-5 Target**: `~22.05809` (gap: **`0.93887`**)
- **Top-1 Reference**: `~22.03995` (gap: **`0.95701`**)

### Where the Remaining Score Lives
| Bucket | Count | Score Sum |
| --- | ---: | ---: |
| Perfect squares (`score = 0`) | `31` | `0.00000` |
| Improved by `exp7` | `447` | `13.27858` |
| Untouched non-square grid (`n <= 100`) | `57` | `3.47222` |
| Untouched non-square grid (`n > 100`) | `465` | `6.24617` |

## High-ROI Roadmap for `exp8`
1. **Compiled / GPU-Batched Billiard-Inflation & Multi-Start L-BFGS (`n = 5..150`)**:
   - Port `fixed_box_energy_and_grad` / Gensane–Ryckelynck billiard inflation to **PyTorch (GPU batched across hundreds of restarts)** or **C++/Rust** (`pybind11` / standalone binary).
   - Target the `57` untouched `n <= 100` puzzles (`n=6, 7, 12, 13, 14, 15, 19, 20, 29, 30, ...`) and tighten `n=11, 17, 18, 26, 27, 28, 37, 38, ...`.
   - Re-run `propagate_composites` from `experiments/exp7/search.py` after any improvement to cascade gains to larger `n`.
2. **Chung–Graham Small-Angle Tilted Strips & 3×3 / Recursive Composites (`n > 100`)**:
   - Attack the `465` untouched upper-band puzzles (`6.24617` score) using multi-layer small-angle tilted strips ($\theta \approx k^{-1/2}$) and recursive $2\times 2$ / $3\times 3$ block compositions.
3. **Final Certification**:
   - Always run `certify_all_mpmath(..., dps=60)` before generating the final Kaggle CSV, and write coordinates with `float_precision=17` (or higher precision strings for tight packings).
