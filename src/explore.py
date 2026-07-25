from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.data_loader import FEATURE_COLUMNS, load_creditcard_data


def explore_data() -> pd.DataFrame:
    df = load_creditcard_data()
    print("Loaded dataset with shape:", df.shape)

    counts = df["Class"].value_counts().sort_index()
    total = len(df)
    fraud = counts.get(1, 0)
    nonfraud = counts.get(0, 0)
    fraud_ratio = fraud / total

    print("\nClass distribution:")
    print(counts)
    print(f"Total samples: {total}")
    print(f"Fraudulent samples: {fraud} ({fraud_ratio:.4%})")
    print(f"Non-fraud samples: {nonfraud} ({(nonfraud / total):.4%})")

    print("\nBasic statistics for features:")
    stats = df[FEATURE_COLUMNS].describe().T
    print(stats)

    print("\nCounting missing values:")
    missing = df.isna().sum()
    print(missing[missing > 0])

    print("\nTop feature skewness and kurtosis:")
    skewness = df[FEATURE_COLUMNS].skew().abs().sort_values(ascending=False).head(10)
    kurtosis = df[FEATURE_COLUMNS].kurt().abs().sort_values(ascending=False).head(10)
    print("Skewness (absolute):")
    print(skewness)
    print("Kurtosis (absolute):")
    print(kurtosis)

    print("\nGenerating distribution plots for selected features...")
    subset = ["V1", "V2", "V3", "V4", "Amount"]
    fig, axes = plt.subplots(len(subset), 1, figsize=(8, 3 * len(subset)))
    for feature, ax in zip(subset, axes):
        sns.histplot(df[feature], bins=60, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
    plt.tight_layout()
    notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    plot_path = notebooks_dir / "explore_distributions.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature distribution plot to {plot_path}")

    return df


if __name__ == "__main__":
    explore_data()
