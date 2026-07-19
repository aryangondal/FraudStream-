import os
from pathlib import Path

import pandas as pd

DATA_FILENAME = "creditcard.csv"
EXPECTED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Class",
]


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
