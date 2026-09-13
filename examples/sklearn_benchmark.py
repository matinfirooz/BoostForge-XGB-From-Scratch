"""
Optional benchmark: compare BoostForge against scikit-learn's
GradientBoostingClassifier on the same synthetic dataset.

scikit-learn is NOT used by BoostForge itself.
"""

from time import perf_counter

from boostforge import XGBoostScratch, accuracy_score
from boostforge.datasets import make_binary_classification, train_test_split


def main():
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score as sk_accuracy
    except ImportError:
        raise SystemExit(
            "Install optional benchmark dependencies with: pip install -e '.[dev]'"
        )

    X, y = make_binary_classification(n_samples=1800, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ours = XGBoostScratch(
        n_estimators=60,
        learning_rate=0.12,
        max_depth=3,
        max_bins=48,
        objective="binary:logistic",
        random_state=42,
    )

    t0 = perf_counter()
    ours.fit(X_train, y_train)
    ours_time = perf_counter() - t0
    ours_acc = accuracy_score(y_test, ours.predict(X_test))

    baseline = GradientBoostingClassifier(
        n_estimators=60,
        learning_rate=0.12,
        max_depth=3,
        random_state=42,
    )

    t0 = perf_counter()
    baseline.fit(X_train, y_train)
    sk_time = perf_counter() - t0
    sk_acc = sk_accuracy(y_test, baseline.predict(X_test))

    print("Model                          Accuracy     Train time (s)")
    print("-" * 58)
    print(f"BoostForge (from scratch)      {ours_acc:8.4f}     {ours_time:10.3f}")
    print(f"sklearn GradientBoosting       {sk_acc:8.4f}     {sk_time:10.3f}")


if __name__ == "__main__":
    main()
