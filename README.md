# FraudStream — Fraud Detection Phase 1

Sentinel is a fraud detection portfolio project. Phase 1 focuses on dataset ingestion, exploratory analysis, and a baseline unsupervised anomaly detector using Isolation Forest.

## Project structure

- `src/` — core Python modules and pipeline scripts
- `data/` — expected dataset input
- `models/` — serialized model output
- `notebooks/` — exploratory artifacts and plots
- `tests/` — basic smoke tests

## Dataset file

Place the Kaggle Credit Card Fraud Detection dataset in:

- `data/creditcard.csv`

This script expects the exact filename `creditcard.csv`.

## How to run

From the project root:

```bash
python -m src.explore
python -m src.train
```

## Phase 1 goals

1. Load the dataset from `data/creditcard.csv`
2. Explore class imbalance, feature distributions, and basic statistics
3. Train an `IsolationForest` unsupervised anomaly detector without using the `Class` label for training
4. Evaluate with `precision`, `recall`, and `PR-AUC`
5. Save the trained model to `models/isolation_forest.joblib`

## Design notes

- The `Class` label is reserved only for evaluation. The model is trained unsupervised on raw features.
- `IsolationForest` uses a contamination estimate to decide how many points are treated as anomalies.
- Accuracy is not a reliable metric in this dataset because fraud is very rare. A model can predict "no fraud" everywhere and still achieve >99% accuracy while missing all frauds.
- `precision`, `recall`, and `PR-AUC` are better for imbalanced anomaly detection because they focus on correctly identifying the rare positive class.

## File overview

- `src/data_loader.py` — loads `creditcard.csv` and validates the data format
- `src/explore.py` — prints imbalance metrics, descriptive statistics, and feature distribution summaries
- `src/train.py` — trains an `IsolationForest`, evaluates with the true labels, and saves the model

## Future phases

Phase 2 can add a FastAPI inference service and Kafka integration once this Phase 1 pipeline is validated.
