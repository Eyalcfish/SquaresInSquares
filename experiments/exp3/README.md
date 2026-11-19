# Experiment 3: Fixed-Box L-BFGS-B Compression + Binary Search

## Method
- Instead of minimizing $L$ directly (which only pushes outermost squares and jams the interior), `exp3` minimizes the total boundary + SAT overlap violation energy $E_L(W)$ inside a **fixed target box** $[0, L]^2$ using `L-BFGS-B` with exact analytical gradients.
- Binary search over $L \in [\sqrt{n}, L_{\text{baseline}}]$ with warm-starting from the best valid packing and multi-start seeds (compressed grid, $45^\circ$-rotated mix, uniform random).
- Verified with 60-digit `mpmath` (`dps=60`).

## Results (20 Target Puzzles, 121.94s Total)

| Metric | Value |
| --- | ---: |
| Puzzles tested | 20 |
| Puzzles improved (`> 1e-6`) | **9 / 20** (`n=5, 10, 11, 17, 18, 26, 27, 37, 38`) |
| Subset baseline score | `3.44079354` |
| Subset Exp3 score | `3.22264969` |
| **Subset improvement** | **`+0.21814385`** |
| **Kaggle Baseline (Grid)** | **`28.93767`** |
| **Kaggle Exp1 Score** | `28.85491` |
| **Kaggle Exp2 Score** | `28.86366` |
| **Kaggle Exp3 Score** | **`28.71953`** (`-0.21814`) |
| **Kaggle Top-5 Leaderboard** | `~22.05809` |
| **Kaggle Top-1 Leaderboard** | `~22.03995` |
| **Gap to Top-1** | `6.67958` |
