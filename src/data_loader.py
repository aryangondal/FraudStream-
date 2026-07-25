import os
from pathlib import Path

import pandas as pd

DATA_FILENAME = "creditcard.csv"

# Single source of truth for column layout. train.py, explore.py, and Phase 2's
# inference endpoint must all use FEATURE_COLUMNS (in this order) rather than
# redefining their own list, or feature order can silently drift between them.
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def get_data_path() -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    return root_dir / "data" / DATA_FILENAME


def load_creditcard_data() -> pd.DataFrame:
    data_path = get_data_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected dataset file at {data_path}. Please download the Kaggle Credit Card Fraud Detection dataset and place it there."
        )

    df = pd.read_csv(data_path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. Please verify the file is `creditcard.csv` from Kaggle."
        )

    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra:
        raise ValueError(
            f"Dataset contains unexpected columns: {extra}. Please verify the file is `creditcard.csv` from Kaggle."
        )

    return df
