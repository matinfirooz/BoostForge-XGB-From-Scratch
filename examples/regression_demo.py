from pathlib import Path

from boostforge import XGBoostScratch, r2_score, rmse
from boostforge.datasets import make_regression, train_test_split
from boostforge.visualization import plot_feature_importance, plot_training_history


def main():
    X, y = make_regression(n_samples=1400, n_features=8, random_state=19)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=19
    )

    model = XGBoostScratch(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        reg_lambda=1.2,
        subsample=0.9,
        colsample_bytree=0.9,
        max_bins=48,
        objective="reg:squarederror",
        random_state=19,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=15,
        verbose=10,
    )

    pred = model.predict(X_test)
    print("\nRegression results")
    print("------------------")
    print(f"RMSE : {rmse(y_test, pred):.4f}")
    print(f"R^2  : {r2_score(y_test, pred):.4f}")
    print(f"Trees: {len(model.trees_)}")

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    plot_training_history(model, out / "regression_training.png")
    plot_feature_importance(model, save_path=out / "regression_importance.png")
    print("\nSaved plots to ./artifacts/")


if __name__ == "__main__":
    main()
