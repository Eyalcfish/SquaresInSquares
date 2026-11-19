# Experiment 6: Expanding-Bucket Circle Drop + Bounded SA

## Method
- **Stage 1 (Expanding-Bucket Circle Drop)**: Drop $n$ area-matched circles ($r = 1/\sqrt{\pi} \approx 0.5642$) into a square bucket starting at $L = 1.02\sqrt{n}$, relaxing overlaps via spring repulsion and expanding $L \leftarrow 1.03 L$ until all circles fit (`bucket_L`).
- **Stage 2 (Bounded SA + L-BFGS-B Polish)**: Initialize $(x_i, y_i)$ from the settled circle centers and run Bounded SA (`8000` steps) + binary search over $L$ + `L-BFGS-B` tightening.

## Results (20 Target Puzzles, 88.75s Total)

| Metric | Value |
| --- | ---: |
| Puzzles tested | 20 |
| Puzzles improved (`> 1e-6`) | **2 / 20** (`n=5, 10`) |
| `n=5` | `bucket_L = 2.889236`, `exp6_L = 2.803651` (`+0.087810`, `mp60=True`) |
| `n=10` | `bucket_L = 3.966987`, `exp6_L = 3.941437` (`+0.018519`, `mp60=True`) |
| Subset improvement | **`+0.10632915`** |
| **Kaggle Exp6 Score** | **`28.83134`** (`-0.10633` vs `28.93767` baseline) |
