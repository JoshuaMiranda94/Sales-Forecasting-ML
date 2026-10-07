from __future__ import annotations

import pandas as pd


DATE_FEATURES = ("year", "month", "day", "weekday", "dayofyear", "weekofyear")


def engineer_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Convert a date column into model-friendly calendar features."""
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")

    if result[date_col].isna().any():
        bad = int(result[date_col].isna().sum())
        raise ValueError(f"{bad} rows contain invalid dates")

    result["year"] = result[date_col].dt.year
    result["month"] = result[date_col].dt.month
    result["day"] = result[date_col].dt.day
    result["weekday"] = result[date_col].dt.weekday
    result["dayofyear"] = result[date_col].dt.dayofyear
    result["weekofyear"] = result[date_col].dt.isocalendar().week.astype(int)

    return result


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by date and return older rows for training, newer rows for testing."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    ordered = df.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col], errors="coerce")

    if ordered[date_col].isna().any():
        raise ValueError("Invalid dates found before chronological split")

    ordered = ordered.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)

    if split_index == 0 or split_index == len(ordered):
        raise ValueError("Not enough rows for chronological split")

    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()
