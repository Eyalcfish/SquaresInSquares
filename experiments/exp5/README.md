# Experiment 5: Big Bang Bounded Simulated Annealing

## Method
- **Big Bang Initialization**: All $n$ squares start at the center $(L/2, L/2)$ of a bounded square $[0, L]^2$ with random orientations.
- **Bounded SA**: Single-square moves (plus 15% long jumps) are clamped strictly inside $[0, L]^2$, with $O(n)$ incremental SAT overlap-matrix updates (`8000` steps, geometric temperature `0.30 -> 0.0005`), followed by optional `L-BFGS-B` tightening and binary search over $L$.

## Results (20 Target Puzzles, 87.41s Total)

| Metric | Value |
| --- | ---: |
| Puzzles tested | 20 |
| Puzzles improved (`> 1e-6`) | **2 / 20** (`n=5, 10`) |
| Subset baseline score | `3.44079354` |
| Subset Exp5 score | `3.34136264` |
| **Subset improvement** | **`+0.09943091`** |
| **Kaggle Baseline (Grid)** | `28.93767` |
| **Kaggle Exp1 (SA + Adam)** | `28.85491` |
| **Kaggle Exp2 (Adaptive Expansion SA+GD)** | `28.86366` |
| **Kaggle Exp5 (Big Bang Bounded SA)** | **`28.83824`** (`-0.09943`) |
| **Kaggle Exp3 (Fixed-Box L-BFGS-B)** | **`28.71953`** (`-0.21814`) |
| **Kaggle Top-1 Leaderboard** | `~22.03995` |
