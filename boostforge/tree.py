from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
    is_leaf: bool
    value: float = 0.0
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    gain: float = 0.0
    n_samples: int = 0


class GradientTree:
    """
    Regression tree optimized directly on first- and second-order derivatives.

    This is the central idea behind XGBoost-style boosting:
      * each split is scored using gradient / Hessian statistics
      * each leaf stores a regularized Newton step
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        max_bins: int | None = 64,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.max_bins = max_bins
        self.random_state = random_state

        self.root: Node | None = None
        self.feature_gains_: dict[int, float] = {}

    @staticmethod
    def _score(grad_sum: float, hess_sum: float, reg_lambda: float) -> float:
        return (grad_sum * grad_sum) / (hess_sum + reg_lambda)

    def _leaf_weight(self, grad_sum: float, hess_sum: float) -> float:
        return -grad_sum / (hess_sum + self.reg_lambda)

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        """
        Return split thresholds.

        For low-cardinality features, evaluate all adjacent midpoints.
        For high-cardinality features, use quantile-based candidate bins to keep
        the implementation practical while still remaining NumPy-only.
        """
        unique = np.unique(values)
        if unique.size <= 1:
            return np.empty(0, dtype=float)

        if self.max_bins is None or unique.size <= self.max_bins:
            return (unique[:-1] + unique[1:]) / 2.0

        quantiles = np.linspace(0.0, 1.0, self.max_bins + 2)[1:-1]
        thresholds = np.unique(np.quantile(values, quantiles))
        return thresholds

    def fit(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature_indices: np.ndarray | None = None,
    ) -> "GradientTree":
        X = np.asarray(X, dtype=float)
        grad = np.asarray(grad, dtype=float).reshape(-1)
        hess = np.asarray(hess, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(grad) or len(X) != len(hess):
            raise ValueError("X, grad, and hess must contain the same number of rows.")

        if feature_indices is None:
            feature_indices = np.arange(X.shape[1], dtype=int)
        else:
            feature_indices = np.asarray(feature_indices, dtype=int)

        self.feature_gains_ = {}
        row_indices = np.arange(X.shape[0], dtype=int)
        self.root = self._build(
            X=X,
            grad=grad,
            hess=hess,
            rows=row_indices,
            feature_indices=feature_indices,
            depth=0,
        )
        return self

    def _build(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        rows: np.ndarray,
        feature_indices: np.ndarray,
        depth: int,
    ) -> Node:
        G = float(np.sum(grad[rows]))
        H = float(np.sum(hess[rows]))
        leaf_value = self._leaf_weight(G, H)

        node = Node(
            is_leaf=True,
            value=leaf_value,
            n_samples=int(rows.size),
        )

        if (
            depth >= self.max_depth
            or rows.size < self.min_samples_split
            or H < 2.0 * self.min_child_weight
        ):
            return node

        parent_score = self._score(G, H, self.reg_lambda)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        for feature in feature_indices:
            values = X[rows, feature]
            thresholds = self._candidate_thresholds(values)

            for threshold in thresholds:
                left_mask = values <= threshold
                n_left = int(np.sum(left_mask))
                n_right = int(rows.size - n_left)

                if n_left == 0 or n_right == 0:
                    continue

                left_rows = rows[left_mask]
                right_rows = rows[~left_mask]

                G_left = float(np.sum(grad[left_rows]))
                H_left = float(np.sum(hess[left_rows]))
                G_right = G - G_left
                H_right = H - H_left

                if (
                    H_left < self.min_child_weight
                    or H_right < self.min_child_weight
                ):
                    continue

                split_gain = (
                    0.5
                    * (
                        self._score(G_left, H_left, self.reg_lambda)
                        + self._score(G_right, H_right, self.reg_lambda)
                        - parent_score
                    )
                    - self.gamma
                )

                if split_gain > best_gain:
                    best_gain = split_gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_rows
                    best_right = right_rows

        if best_feature is None or best_gain <= 0.0:
            return node

        node.is_leaf = False
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.gain = float(best_gain)
        self.feature_gains_[best_feature] = (
            self.feature_gains_.get(best_feature, 0.0) + float(best_gain)
        )

        node.left = self._build(
            X, grad, hess, best_left, feature_indices, depth + 1
        )
        node.right = self._build(
            X, grad, hess, best_right, feature_indices, depth + 1
        )
        return node

    def _predict_row(self, row: np.ndarray, node: Node) -> float:
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root) for row in X], dtype=float)

    def _to_text(self, node: Node, depth: int, lines: list[str]) -> None:
        indent = "  " * depth
        if node.is_leaf:
            lines.append(
                f"{indent}Leaf(value={node.value:.6f}, samples={node.n_samples})"
            )
            return

        lines.append(
            f"{indent}if x[{node.feature_index}] <= {node.threshold:.6f} "
            f"(gain={node.gain:.6f}, samples={node.n_samples}):"
        )
        self._to_text(node.left, depth + 1, lines)
        lines.append(f"{indent}else:")
        self._to_text(node.right, depth + 1, lines)

    def to_text(self) -> str:
        if self.root is None:
            return "<unfitted tree>"
        lines: list[str] = []
        self._to_text(self.root, 0, lines)
        return "\n".join(lines)
