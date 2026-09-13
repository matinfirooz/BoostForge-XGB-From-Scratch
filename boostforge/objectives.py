from __future__ import annotations

import numpy as np


_EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def binary_logistic_grad_hess(
    y_true: np.ndarray, raw_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradients and Hessians of binary logistic loss w.r.t. raw logits.

    L = -y log(p) - (1-y) log(1-p)
    g = p - y
    h = p(1-p)
    """
    p = sigmoid(raw_pred)
    grad = p - y_true
    hess = np.clip(p * (1.0 - p), 1e-12, None)
    return grad, hess


def squared_error_grad_hess(
    y_true: np.ndarray, raw_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradients and Hessians for 1/2 * (prediction - target)^2.
    """
    grad = raw_pred - y_true
    hess = np.ones_like(grad, dtype=float)
    return grad, hess


def binary_logloss_from_raw(y_true: np.ndarray, raw_pred: np.ndarray) -> float:
    p = np.clip(sigmoid(raw_pred), _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def mse_from_raw(y_true: np.ndarray, raw_pred: np.ndarray) -> float:
    return float(np.mean((raw_pred - y_true) ** 2))
