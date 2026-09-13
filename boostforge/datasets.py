from __future__ import annotations

import numpy as np


def make_binary_classification(
    n_samples: int = 1200,
    n_features: int = 8,
    random_state: int = 42,
):
    """
    Non-linear synthetic binary classification dataset with interactions.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))

    score = (
        1.6 * X[:, 0]
        - 1.3 * X[:, 1]
        + 0.9 * X[:, 2] * X[:, 3]
        + 1.2 * np.sin(X[:, 4])
        - 0.7 * X[:, 5] ** 2
        + rng.normal(0, 0.55, size=n_samples)
    )
    threshold = np.median(score)
    y = (score > threshold).astype(int)
    return X, y


def make_regression(
    n_samples: int = 1200,
    n_features: int = 8,
    random_state: int = 42,
):
    """
    Non-linear synthetic regression dataset.
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2.5, 2.5, size=(n_samples, n_features))
    y = (
        3.0 * np.sin(X[:, 0])
        + 1.7 * X[:, 1] ** 2
        - 2.0 * X[:, 2] * X[:, 3]
        + 0.6 * X[:, 4]
        + rng.normal(0, 0.35, size=n_samples)
    )
    return X, y


def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    indices = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
