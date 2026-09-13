from __future__ import annotations

import argparse
import csv

import numpy as np

from boostforge import XGBoostScratch, accuracy_score, rmse
from boostforge.datasets import train_test_split


def load_csv(path: str, target: str):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")
    if target not in rows[0]:
        raise ValueError(f"Target column '{target}' does not exist.")

    feature_names = [name for name in rows[0].keys() if name != target]
    X = np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    y = np.array([float(row[target]) for row in rows], dtype=float)
    return X, y, feature_names


def main():
    parser = argparse.ArgumentParser(description="Train BoostForge from a numeric CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
    )
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    X, y, feature_names = load_csv(args.csv_path, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    objective = (
        "binary:logistic" if args.task == "classification" else "reg:squarederror"
    )

    model = XGBoostScratch(
        n_estimators=args.trees,
        max_depth=args.depth,
        learning_rate=0.1,
        objective=objective,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=10)

    pred = model.predict(X_test)

    if args.task == "classification":
        print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    else:
        print(f"\nRMSE: {rmse(y_test, pred):.4f}")

    print("\nFeature importance:")
    order = np.argsort(model.feature_importances_)[::-1]
    for idx in order:
        print(f"  {feature_names[idx]:20s} {model.feature_importances_[idx]:.5f}")


if __name__ == "__main__":
    main()
