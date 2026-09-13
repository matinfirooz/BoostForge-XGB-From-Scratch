"""
BoostForge: an educational XGBoost-style Gradient Boosted Decision Tree
implementation built from scratch with NumPy.
"""

from .model import XGBoostScratch
from .metrics import accuracy_score, log_loss, mse, rmse, r2_score

__all__ = [
    "XGBoostScratch",
    "accuracy_score",
    "log_loss",
    "mse",
    "rmse",
    "r2_score",
]

__version__ = "0.1.0"
