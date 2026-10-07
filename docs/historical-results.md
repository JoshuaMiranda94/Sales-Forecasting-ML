# Historical Coursework Results

The original notebook was recovered from Git history after it had been deleted from the current `ironkaggleProject` working tree.

## Original Workflow

The notebook:

- loaded `sales.csv`
- converted `date` to datetime
- created year / month / day / weekday features
- one-hot encoded `state_holiday`
- sampled 50,000 rows
- used an 80/20 random train/test split
- trained `DecisionTreeRegressor(max_depth=10, random_state=42)`
- serialized the trained model with Joblib

## Reported Metrics

```text
MAE: 891.50
RMSE: 1363.86
R² Score: 0.88
```

Exact stored values:

```json
{
  "MAE": 891.4957557383034,
  "RMSE": 1363.857132862124,
  "R2": 0.8758117051293183
}
```

## Recovered Sample Predictions

```text
8143.35
6011.36
10879.16
9613.13
0.00
```

## Issue Found During Audit

The historical notebook dropped the original `date` column and later attempted to use `df_sales.groupby("date")`, which produced a `KeyError: 'date'`.

The refactored implementation avoids this by preserving chronological information until splitting and feature creation are complete.

## Why the New Evaluation Is Different

A random split can leak future patterns into the training set when data has a time component.

The refactored portfolio version sorts by date and holds out the most recent observations. This makes the evaluation more representative of a forecasting-style use case.
