"""Load preprocessed Sentiment140 dataset with train/val/test splits."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_cleaned_dataset(
    data_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned dataset and split 60/20/20 (train/val/test).

    Args:
        data_path: Path to cleaned CSV. Defaults to data/processed/cleaned_dataset_sentiment.csv
        sample_size: Optional stratified sample size before splitting
        random_state: Random seed

    Returns:
        train_df, val_df, test_df
    """
    root = Path(__file__).resolve().parents[2]
    if data_path is None:
        data_path = root / "data" / "processed" / "cleaned_dataset_sentiment.csv"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    if text_col not in df.columns:
        raise ValueError("Dataset must contain 'cleaned_text' or 'text' column")

    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column")

    if sample_size and sample_size < len(df):
        df = (
            df.groupby("target", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), sample_size // 2), random_state=random_state))
            .reset_index(drop=True)
        )

    x_temp, x_test, y_temp, y_test = train_test_split(
        df[text_col],
        df["target"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["target"],
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=random_state,
        stratify=y_temp,
    )

    train_df = pd.DataFrame({"cleaned_text": x_train, "target": y_train})
    val_df = pd.DataFrame({"cleaned_text": x_val, "target": y_val})
    test_df = pd.DataFrame({"cleaned_text": x_test, "target": y_test})

    return train_df, val_df, test_df


def get_splits_dict(
    sample_size: Optional[int] = None,
    data_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Return splits as a dictionary."""
    train_df, val_df, test_df = load_cleaned_dataset(
        data_path=data_path, sample_size=sample_size
    )
    return {"train": train_df, "val": val_df, "test": test_df}
