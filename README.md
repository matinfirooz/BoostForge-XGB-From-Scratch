# BoostForge — XGBoost-Style Gradient Boosted Trees From Scratch

> A research-friendly implementation of second-order Gradient Boosted Decision
> Trees written from scratch with NumPy.

BoostForge is designed for people who do not just want to call `xgboost.fit()`.
The repository reconstructs the central ideas behind modern boosted-tree
systems: first- and second-order derivatives, regularized Newton leaf updates,
gradient/Hessian split gain, stochastic row and feature sampling, early stopping,
and feature-importance accumulation.

The **model itself uses only NumPy**. No XGBoost, LightGBM, CatBoost, or
scikit-learn estimator is used internally.

---

## Why this repository is interesting

Most educational boosting implementations simply fit trees to residuals.
BoostForge goes further and implements an **XGBoost-style second-order
optimization objective**.

It includes:

- binary classification with logistic loss
- non-linear regression with squared-error loss
- gradient + Hessian computation
- regularized optimal leaf weights
- gradient/Hessian split-gain formula
- exact greedy threshold search for small-cardinality features
- quantile-based candidate thresholds for larger feature sets
- L2 leaf regularization
- split penalty (`gamma`)
- minimum child Hessian constraint
- row subsampling
- feature subsampling per tree
- early stopping
- normalized split-gain feature importance
- human-readable tree dumps
- plots for training curves and feature importance
- CSV training CLI
- unit/smoke tests
- optional comparison against scikit-learn

---

## Repository structure

```text
BoostForge-XGB-From-Scratch/
├── boostforge/
│   ├── __init__.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── model.py
│   ├── objectives.py
│   ├── tree.py
│   └── visualization.py
├── docs/
│   └── algorithm.md
├── examples/
│   ├── classification_demo.py
│   ├── regression_demo.py
│   └── sklearn_benchmark.py
├── scripts/
│   └── train_csv.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Core mathematics

For each boosting round, BoostForge uses a second-order Taylor approximation of
the objective:

```text
current predictions
        │
        ▼
compute gradients g_i and Hessians h_i
        │
        ▼
search tree splits using gradient/Hessian gain
        │
        ▼
compute regularized Newton leaf values
        │
        ▼
add η × tree prediction to the ensemble
        │
        ▼
repeat
```

For a node with gradient sum `G` and Hessian sum `H`, the optimal leaf weight is

$$
w^*=-\frac{G}{H+\lambda}
$$

A candidate split is scored by

$$
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
$$

That is the key difference between this project and a basic residual-fitting
Gradient Boosting tutorial.

See [`docs/algorithm.md`](docs/algorithm.md) for the full derivation.

---

## Installation

### Minimal

```bash
git clone https://github.com/matinfirooz/BoostForge-XGB-From-Scratch.git
cd BoostForge-XGB-From-Scratch

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -e .
```

### Development / benchmark dependencies

```bash
pip install -e ".[dev]"
```

---

## Quick start — binary classification

```python
from boostforge import XGBoostScratch, accuracy_score
from boostforge.datasets import make_binary_classification, train_test_split

X, y = make_binary_classification(n_samples=1200, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = XGBoostScratch(
    n_estimators=80,
    learning_rate=0.12,
    max_depth=3,
    reg_lambda=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=10,
    verbose=10,
)

pred = model.predict(X_test)
print("accuracy =", accuracy_score(y_test, pred))
```

---

## Quick start — regression

```python
from boostforge import XGBoostScratch, rmse
from boostforge.datasets import make_regression, train_test_split

X, y = make_regression(n_samples=1200, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = XGBoostScratch(
    n_estimators=100,
    learning_rate=0.08,
    max_depth=3,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("RMSE =", rmse(y_test, pred))
```

---

## Run the demos

```bash
python examples/classification_demo.py
python examples/regression_demo.py
```

The demos generate plots inside an `artifacts/` directory.

---

## Inspect a learned tree

```python
print(model.dump_tree(0))
```

Example format:

```text
if x[0] <= 0.143211 (gain=52.418...):
  if x[1] <= -0.501...:
    Leaf(value=..., samples=...)
  else:
    Leaf(value=..., samples=...)
else:
  ...
```

This is useful for understanding what the first boosting stage actually learned.

---

## Feature importance

After fitting:

```python
print(model.feature_importances_)
```

The importance values are normalized accumulated split gains.

You can also generate a plot:

```python
from boostforge.visualization import plot_feature_importance
plot_feature_importance(model)
```

---

## Train from a CSV

Only numeric features are expected.

```bash
python scripts/train_csv.py data.csv \
    --target label \
    --task classification \
    --trees 80 \
    --depth 3
```

For regression:

```bash
python scripts/train_csv.py data.csv \
    --target price \
    --task regression
```

---

## Optional benchmark

```bash
pip install -e ".[dev]"
python examples/sklearn_benchmark.py
```

This compares the from-scratch implementation against scikit-learn on the same
synthetic dataset.

The goal is not to beat optimized C/C++ libraries. The goal is to make the
optimization mechanics transparent while still producing a useful model.

---

## Hyperparameters

| Parameter | Meaning |
|---|---|
| `n_estimators` | Maximum number of boosted trees |
| `learning_rate` | Shrinkage applied to each new tree |
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Minimum rows required to attempt a split |
| `min_child_weight` | Minimum Hessian mass allowed in a child |
| `reg_lambda` | L2 regularization on leaf values |
| `gamma` | Minimum split-gain penalty |
| `subsample` | Fraction of rows sampled for each tree |
| `colsample_bytree` | Fraction of features sampled for each tree |
| `max_bins` | Maximum candidate threshold count per feature |
| `objective` | `binary:logistic` or `reg:squarederror` |
| `random_state` | Reproducibility seed |

---

## Tests

```bash
pytest -q
```

The tests check that:

- binary classification learns a non-linear dataset
- regression achieves a meaningful positive R²
- predicted probabilities have the correct shape
- feature importances are finite
- a learned tree can be dumped as readable text

---

## What was implemented from scratch?

| Component | From scratch? |
|---|---:|
| Gradient boosting loop | Yes |
| Binary logistic gradient/Hessian | Yes |
| Squared-error gradient/Hessian | Yes |
| Decision tree construction | Yes |
| Split search | Yes |
| Split gain | Yes |
| Leaf optimization | Yes |
| Feature subsampling | Yes |
| Row subsampling | Yes |
| Early stopping | Yes |
| Feature importance | Yes |
| Accuracy / log loss / RMSE / R² | Yes |
| NumPy array operations | Library |
| Matplotlib visualization | Library |
| scikit-learn benchmark | Optional only |

---

## Limitations

BoostForge is intentionally educational rather than production optimized.

Current limitations:

- binary classification only; multiclass softmax is not yet implemented
- dense numeric features only
- no native missing-value direction learning
- no sparse-matrix acceleration
- Python-level tree traversal is slower than compiled libraries
- quantile candidates are recomputed locally instead of using a production
  histogram builder

These limitations make good future GitHub issues and extension projects.

---

## Good next extensions

1. multiclass softmax objective
2. missing-value split direction
3. histogram-based tree builder
4. monotonic constraints
5. SHAP-style contribution approximation
6. model serialization
7. parallel split search
8. categorical split support
9. GPU split evaluation with CuPy
10. hardware-aware tree inference benchmark

---

## License

MIT License. See [`LICENSE`](LICENSE).

---

## Author note

If you use this repository in a portfolio, describe it accurately:

> “Implemented an XGBoost-style second-order Gradient Boosted Decision Tree
> learner from scratch using NumPy, including gradient/Hessian split gain,
> regularized Newton leaf updates, stochastic sampling, early stopping, and
> feature importance.”

That statement is strong, technical, and truthful.
