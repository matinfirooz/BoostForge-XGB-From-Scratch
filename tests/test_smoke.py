import numpy as np

from boostforge import XGBoostScratch, accuracy_score, r2_score
from boostforge.datasets import (
    make_binary_classification,
    make_regression,
    train_test_split,
)


def test_binary_classifier_learns():
    X, y = make_binary_classification(n_samples=350, n_features=8, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    model = XGBoostScratch(
        n_estimators=20,
        learning_rate=0.15,
        max_depth=3,
        max_bins=24,
        objective="binary:logistic",
        random_state=1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    assert accuracy_score(y_test, pred) > 0.70
    assert model.predict_proba(X_test).shape == (len(X_test), 2)


def test_regressor_learns():
    X, y = make_regression(n_samples=350, n_features=8, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2
    )

    model = XGBoostScratch(
        n_estimators=25,
        learning_rate=0.12,
        max_depth=3,
        max_bins=24,
        objective="reg:squarederror",
        random_state=2,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    assert r2_score(y_test, pred) > 0.45
    assert np.isfinite(model.feature_importances_).all()


def test_tree_dump_contains_leaf():
    X, y = make_binary_classification(n_samples=120, random_state=3)
    model = XGBoostScratch(
        n_estimators=3,
        max_depth=2,
        max_bins=16,
        objective="binary:logistic",
        random_state=3,
    )
    model.fit(X, y)
    text = model.dump_tree(0)
    assert "Leaf" in text
