import pandas as pd

from src.features import chronological_split, engineer_date_features


def test_engineer_date_features():
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "sales": [100, 120],
        }
    )

    result = engineer_date_features(df)

    for column in ["year", "month", "day", "weekday", "dayofyear", "weekofyear"]:
        assert column in result.columns


def test_chronological_split_is_ordered():
    df = pd.DataFrame(
        {
            "date": ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-04"],
            "sales": [30, 10, 20, 40],
        }
    )

    train, test = chronological_split(df, train_fraction=0.75)

    assert train["date"].max() < test["date"].min()
    assert len(train) == 3
    assert len(test) == 1
