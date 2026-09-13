from __future__ import annotations

import numpy as np


_EPS = 1e-12


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def log_loss(y_true, p1) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p1 = np.clip(np.asarray(p1, dtype=float).reshape(-1), _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p1) + (1.0 - y_true) * np.log(1.0 - p1)))


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
