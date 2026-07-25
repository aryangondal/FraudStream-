import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_creditcard_data

MODEL_FILENAME = "isolation_forest.joblib"
METRICS_FILENAME = "metrics.json"


def get_model_path() -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    return root_dir / "models" / MODEL_FILENAME


def get_metrics_path() -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    return root_dir / "models" / METRICS_FILENAME


def train_model(
    random_state: int = 42,
    contamination: float = 0.0017,
    test_size: float = 0.2,
) -> None:
    df = load_creditcard_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratify on Class so the (tiny) fraud proportion is preserved in both
    # the train and test splits despite the extreme class imbalance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    hyperparameters = {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": contamination,
        "random_state": random_state,
        "n_jobs": -1,
    }

    print(f"Training IsolationForest on {len(X_train)} training rows (unsupervised, no Class label)...")
    # IsolationForest is trained without using the Class label.
    # The contamination parameter estimates the fraction of outliers in the dataset.
    # We choose a small contamination value close to the known fraud rate to make the
    # anomaly threshold more realistic for this highly imbalanced dataset.
    model = IsolationForest(**hyperparameters)
    model.fit(X_train)

    anomaly_scores = -model.decision_function(X_test)
    predicted_anomaly = model.predict(X_test)
    y_pred = np.where(predicted_anomaly == -1, 1, 0)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, anomaly_scores)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    print("\nEvaluation results (held-out test set):")
    print(f"Test rows: {len(X_test)} (fraud: {int(y_test.sum())})")
    print(f"Contamination used: {contamination:.6f}")
    print(f"Predicted fraud count: {int(y_pred.sum())}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nConfusion matrix:")
    print(f"  True Negative:  {tn}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")
    print(f"  True Positive:  {tp}")
    print(
        "\nNote: accuracy is not reported because the dataset is highly imbalanced."
        " A model that predicts all transactions as non-fraud can still achieve >99% accuracy while missing every fraud."
    )
    print(
        "PR-AUC and recall are more informative for this fraud detection task because they focus on the rare positive class."
    )

    model_path = get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to {model_path}")

    metrics = {
        "hyperparameters": hyperparameters,
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_fraud_rows": int(y_test.sum()),
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    metrics_path = get_metrics_path()
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    train_model()
