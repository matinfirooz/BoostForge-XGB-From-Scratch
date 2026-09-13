from pathlib import Path

from boostforge import XGBoostScratch, accuracy_score, log_loss
from boostforge.datasets import make_binary_classification, train_test_split
from boostforge.visualization import plot_feature_importance, plot_training_history


def main():
    X, y = make_binary_classification(n_samples=1400, n_features=8, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    model = XGBoostScratch(
        n_estimators=80,
        learning_rate=0.12,
        max_depth=3,
        min_child_weight=1.0,
        reg_lambda=1.0,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        max_bins=48,
        objective="binary:logistic",
        random_state=7,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=12,
        verbose=10,
    )

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification results")
    print("----------------------")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Log loss : {log_loss(y_test, prob):.4f}")
    print(f"Trees    : {len(model.trees_)}")

    print("\nFirst tree")
    print("----------")
    print(model.dump_tree(0))

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    plot_training_history(model, out / "classification_training.png")
    plot_feature_importance(model, save_path=out / "classification_importance.png")
    print("\nSaved plots to ./artifacts/")


if __name__ == "__main__":
    main()
