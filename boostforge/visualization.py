from __future__ import annotations

import numpy as np


def plot_training_history(model, save_path=None):
    import matplotlib.pyplot as plt

    iterations = [r.iteration for r in model.training_history_]
    train = [r.train_loss for r in model.training_history_]
    eval_loss = [r.eval_loss for r in model.training_history_]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train, label="Train loss")

    if any(v is not None for v in eval_loss):
        valid_x = [i for i, v in zip(iterations, eval_loss) if v is not None]
        valid_y = [v for v in eval_loss if v is not None]
        plt.plot(valid_x, valid_y, label="Validation loss")

    plt.xlabel("Boosting iteration")
    plt.ylabel("Loss")
    plt.title("BoostForge training curve")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()


def plot_feature_importance(model, feature_names=None, save_path=None):
    import matplotlib.pyplot as plt

    imp = np.asarray(model.feature_importances_)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(imp))]

    order = np.argsort(imp)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[order], imp[order])
    plt.xlabel("Normalized split gain")
    plt.title("Feature importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()
