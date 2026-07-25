import joblib
import numpy as np
import pandas as pd

from src import train as train_module
from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN


def _make_synthetic_df(n_rows: int = 400, fraud_rows: int = 20, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = {col: rng.normal(size=n_rows) for col in FEATURE_COLUMNS}
    labels = np.zeros(n_rows, dtype=int)
    labels[:fraud_rows] = 1
    rng.shuffle(labels)
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)


def test_class_label_excluded_from_feature_matrix(tmp_path, monkeypatch):
    df = _make_synthetic_df()
    monkeypatch.setattr(train_module, "load_creditcard_data", lambda: df)
    monkeypatch.setattr(train_module, "get_model_path", lambda: tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "get_metrics_path", lambda: tmp_path / "metrics.json")

    captured = {}
    original_fit = train_module.IsolationForest.fit

    def spy_fit(self, X, *args, **kwargs):
        captured["columns"] = list(X.columns)
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(train_module.IsolationForest, "fit", spy_fit)

    train_module.train_model()

    assert TARGET_COLUMN not in captured["columns"]
    assert captured["columns"] == FEATURE_COLUMNS


def test_model_saves_and_reloads_with_matching_predictions(tmp_path, monkeypatch):
    df = _make_synthetic_df()
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr(train_module, "load_creditcard_data", lambda: df)
    monkeypatch.setattr(train_module, "get_model_path", lambda: model_path)
    monkeypatch.setattr(train_module, "get_metrics_path", lambda: metrics_path)

    dumped = {}
    original_dump = train_module.joblib.dump

    def spy_dump(obj, path):
        dumped["model"] = obj
        return original_dump(obj, path)

    monkeypatch.setattr(train_module.joblib, "dump", spy_dump)

    train_module.train_model()

    assert model_path.exists()

    sample = df[FEATURE_COLUMNS].head(10)
    predictions_before_reload = dumped["model"].predict(sample)

    reloaded_model = joblib.load(model_path)
    predictions_after_reload = reloaded_model.predict(sample)

    assert np.array_equal(predictions_before_reload, predictions_after_reload)
