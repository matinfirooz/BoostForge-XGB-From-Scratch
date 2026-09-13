from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objectives import (
    binary_logistic_grad_hess,
    binary_logloss_from_raw,
    mse_from_raw,
    sigmoid,
    squared_error_grad_hess,
)
from .tree import GradientTree


@dataclass
class TrainingRecord:
    iteration: int
    train_loss: float
    eval_loss: float | None = None


class XGBoostScratch:
    """
    XGBoost-style Gradient Boosted Decision Trees implemented from scratch.

    Supported tasks:
      * binary classification: objective="binary:logistic"
      * regression: objective="reg:squarederror"

    The implementation includes:
      * first- and second-order optimization
      * exact / quantile-threshold split search
      * L2 regularization
      * minimum child Hessian constraint
      * split penalty (gamma)
      * row subsampling
      * feature subsampling
      * early stopping
      * feature-importance accumulation
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        max_bins: int | None = 64,
        objective: str = "binary:logistic",
        random_state: int | None = 42,
    ) -> None:
        if not (0 < learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1].")
        if not (0 < subsample <= 1):
            raise ValueError("subsample must be in (0, 1].")
        if not (0 < colsample_bytree <= 1):
            raise ValueError("colsample_bytree must be in (0, 1].")
        if objective not in {"binary:logistic", "reg:squarederror"}:
            raise ValueError("Unsupported objective.")

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.max_bins = max_bins
        self.objective = objective
        self.random_state = random_state

        self.trees_: list[GradientTree] = []
        self.base_score_: float = 0.0
        self.training_history_: list[TrainingRecord] = []
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: np.ndarray | None = None

    def _validate_xy(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")

        if self.objective == "binary:logistic":
            classes = np.unique(y)
            if not np.all(np.isin(classes, [0.0, 1.0])):
                raise ValueError("Binary classification requires labels encoded as 0/1.")

        return X, y

    def _initial_prediction(self, y: np.ndarray) -> float:
        if self.objective == "binary:logistic":
            p = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
            return float(np.log(p / (1.0 - p)))
        return float(np.mean(y))

    def _grad_hess(self, y, raw_pred):
        if self.objective == "binary:logistic":
            return binary_logistic_grad_hess(y, raw_pred)
        return squared_error_grad_hess(y, raw_pred)

    def _loss(self, y, raw_pred):
        if self.objective == "binary:logistic":
            return binary_logloss_from_raw(y, raw_pred)
        return mse_from_raw(y, raw_pred)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool | int = False,
    ) -> "XGBoostScratch":
        X, y = self._validate_xy(X, y)

        X_eval = y_eval = None
        if eval_set is not None:
            X_eval, y_eval = self._validate_xy(eval_set[0], eval_set[1])
            if X_eval.shape[1] != X.shape[1]:
                raise ValueError("Training and evaluation sets need the same features.")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.trees_ = []
        self.training_history_ = []
        self.base_score_ = self._initial_prediction(y)

        raw_train = np.full(n_samples, self.base_score_, dtype=float)
        raw_eval = (
            np.full(len(X_eval), self.base_score_, dtype=float)
            if X_eval is not None
            else None
        )

        best_score = np.inf
        best_iteration = -1
        rounds_without_improvement = 0

        all_features = np.arange(n_features)

        for iteration in range(self.n_estimators):
            grad, hess = self._grad_hess(y, raw_train)

            row_count = max(2, int(np.ceil(self.subsample * n_samples)))
            if row_count < n_samples:
                rows = rng.choice(n_samples, size=row_count, replace=False)
            else:
                rows = np.arange(n_samples)

            feature_count = max(1, int(np.ceil(self.colsample_bytree * n_features)))
            if feature_count < n_features:
                features = np.sort(
                    rng.choice(all_features, size=feature_count, replace=False)
                )
            else:
                features = all_features

            tree = GradientTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                max_bins=self.max_bins,
                random_state=None if self.random_state is None else self.random_state + iteration,
            )
            tree.fit(X[rows], grad[rows], hess[rows], feature_indices=features)

            # The tree was fitted on a row-subsample, but it uses original feature
            # indices, so it can immediately predict the entire dataset.
            self.trees_.append(tree)
            raw_train += self.learning_rate * tree.predict(X)

            train_loss = self._loss(y, raw_train)
            eval_loss = None

            if X_eval is not None:
                raw_eval += self.learning_rate * tree.predict(X_eval)
                eval_loss = self._loss(y_eval, raw_eval)
                monitored = eval_loss
            else:
                monitored = train_loss

            self.training_history_.append(
                TrainingRecord(
                    iteration=iteration + 1,
                    train_loss=float(train_loss),
                    eval_loss=None if eval_loss is None else float(eval_loss),
                )
            )

            should_print = bool(verbose)
            if isinstance(verbose, int) and not isinstance(verbose, bool) and verbose > 0:
                should_print = ((iteration + 1) % verbose == 0) or iteration == 0

            if should_print:
                msg = f"[{iteration + 1:03d}] train_loss={train_loss:.6f}"
                if eval_loss is not None:
                    msg += f" eval_loss={eval_loss:.6f}"
                print(msg)

            if monitored < best_score - 1e-12:
                best_score = monitored
                best_iteration = iteration
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if (
                early_stopping_rounds is not None
                and rounds_without_improvement >= early_stopping_rounds
            ):
                break

        self.best_iteration_ = best_iteration
        self.best_score_ = float(best_score)

        # Keep only trees up through the best iteration when early stopping is used.
        if early_stopping_rounds is not None and best_iteration >= 0:
            self.trees_ = self.trees_[: best_iteration + 1]

        self._compute_feature_importances()
        return self

    def _compute_feature_importances(self):
        gains = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.trees_:
            for feature, gain in tree.feature_gains_.items():
                gains[feature] += gain

        total = float(np.sum(gains))
        self.feature_importances_ = gains / total if total > 0 else gains

    def predict_raw(self, X) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong shape.")

        raw = np.full(X.shape[0], self.base_score_, dtype=float)
        for tree in self.trees_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict_proba(self, X) -> np.ndarray:
        if self.objective != "binary:logistic":
            raise RuntimeError("predict_proba is only available for binary classification.")
        p1 = sigmoid(self.predict_raw(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.objective == "binary:logistic":
            return (sigmoid(raw) >= threshold).astype(int)
        return raw

    def dump_tree(self, index: int = 0) -> str:
        if not self.trees_:
            raise RuntimeError("Model has not been fitted.")
        return self.trees_[index].to_text()
