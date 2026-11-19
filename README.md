# Squares in Squares

Scoring, high-precision (`mpmath` 60-digit) validation, visualization, and optimization experiments for the Kaggle **Squares in Squares** competition ($n = 1 \dots 1000$).

## Setup

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python pytest
.venv/bin/pytest -v
```

## Evaluate & Visualize a Submission

```bash
.venv/bin/python scripts/evaluate.py data/sample_submission.csv --mpmath-ns 1 2 5 10 1000
.venv/bin/python scripts/visualize_submission.py data/sample_submission.csv
```

## Experiments

- `experiments/exp1`: Simulated Annealing + Adam Gradient Descent hybrid
- `experiments/exp2`: Adaptive-Expansion SA + Adam + L-BFGS-B hybrid
- `experiments/exp3`: Fixed-Box L-BFGS-B Compression + Binary Search (`28.71953` on 20-puzzle subset)
- `experiments/exp4`: Parallelized Fixed-Box L-BFGS-B Compression across all `n=1..1000` (`--workers`)
- `experiments/exp5`: Big Bang Bounded Simulated Annealing (`28.83824` on 20-puzzle subset)

Run any experiment from the repo root, e.g.:

```bash
PYTHONPATH=. .venv/bin/python experiments/exp4/run.py --workers 16
```
