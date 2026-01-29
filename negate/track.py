# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from negate import TrainResult


def in_console(train_result: TrainResult) -> None:
    """Print diagnostics and plots for a trained model."""
    X_train = train_result.X_train
    pca = train_result.pca
    d_matrix_test = train_result.d_matrix_test
    model = train_result.model
    scale_pos_weight = train_result.scale_pos_weight
    X_train_pca = train_result.X_train_pca
    y_test = train_result.y_test
    labels = train_result.labels
    feature_matrix = train_result.feature_matrix
    seed = train_result.seed

    print(f"""
    Original dimensions: {X_train.shape[1]}")
    PCA reduced dimensions: {X_train_pca.shape[1]}
    Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}
    Number of components selected: {pca.n_components_}
    """)

    print(f"Scale pos weight (negative/positive): {scale_pos_weight:.2f}")
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")
    y_pred_proba = model.predict(d_matrix_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"""
    Feature matrix shape: {feature_matrix.shape}\n
    Labels shape: {labels.shape}\n

    \nLabel distribution:
        Real (0): {np.sum(labels == 0)} samples ({np.sum(labels == 0) / len(labels) * 100:.1f}%)\n
        Synthetic (1): {np.sum(labels == 1)} samples ({np.sum(labels == 1) / len(labels) * 100:.1f}%)\n
        Class imbalance ratio: {np.sum(labels == 0) / np.sum(labels == 1):.2f}:1\n
        Random state seed: {seed}"
    """)

    separator = lambda: print("=" * 60)

    separator()
    print("CLASSIFICATION RESULTS")
    separator()
    print(f"""
Accuracy: {accuracy:.4f}
ROC-AUC: {roc_auc:.4f}
F1 Score (Macro): {f1_macro:.4f})
F1 Score (Weighted): {f1_weighted:.4f}""")
    separator()
    print("DETAILED CLASSIFICATION REPORT")
    separator()
    print(classification_report(y_test, y_pred, target_names=["Real", "Synthetic"]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color="aqua")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(range(min(20, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_[:20], color="aqua")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("First 20 Components")
    plt.tight_layout()
    plt.show()
