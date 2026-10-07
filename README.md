# Sales Forecasting ML

Machine-learning portfolio project for predicting retail sales from calendar, store and operational features.

This repository is a cleaned reconstruction of an IronKaggle sales-regression notebook that was accidentally deleted from the current repository state but remained recoverable from Git history.

## Business Problem

Retail teams often need to estimate future or near-term sales in order to support:

- staffing
- inventory planning
- promotions
- store operations
- budgeting
- performance monitoring

This project builds a reproducible regression workflow and compares tree-based models using business-friendly evaluation metrics.

## Original Coursework Result

The recovered historical notebook trained a `DecisionTreeRegressor(max_depth=10)` on a random 50,000-row sample and reported:

| Metric | Historical Result |
|---|---:|
| MAE | 891.50 |
| RMSE | 1,363.86 |
| R² | 0.8758 |

Those metrics are preserved for historical context in `docs/historical-results.md`.

They should **not** be interpreted as the final production estimate because the original notebook used a random train/test split. This refactor uses a chronological split by default, which is more appropriate when evaluating forecasting-style problems.

## What This Refactor Improves

- removes machine-specific file paths
- keeps the date column until the chronological split is complete
- adds reusable feature engineering
- creates preprocessing pipelines for numeric and categorical features
- compares Decision Tree and XGBoost
- records MAE, RMSE and R²
- saves the best model
- exports evaluation artifacts
- includes automated tests
- includes a synthetic demo-data generator
- separates historical coursework results from current methodology

## Important Modeling Note

This project can be used in two different ways:

1. **Same-day sales prediction** — operational information such as customer count may be available.
2. **True future forecasting** — only information known before the forecast date should be used.

If a feature such as `nb_customers_on_day` is not known in advance, it should be removed for a true forecasting deployment.

## Repository Structure

```text
Sales-Forecasting-ML/
├── README.md
├── ATTRIBUTION.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   ├── historical-results.md
│   └── model-card.md
├── scripts/
│   └── generate_demo_data.py
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── evaluate.py
│   └── train.py
├── tests/
│   └── test_features.py
└── artifacts/
    └── README.md
```

## Quick Start

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Generate a reproducible synthetic dataset:

```bash
python scripts/generate_demo_data.py
```

Train and evaluate:

```bash
python -m src.train --data data/demo_sales.csv
```

The training script writes metrics and model artifacts into `artifacts/`.

## Using the Original Dataset

Place the original sales CSV at:

```text
data/sales.csv
```

Then run:

```bash
python -m src.train --data data/sales.csv
```

Required columns:

- `date`
- `sales`

Other numeric and categorical columns are detected automatically.

## Models

### Decision Tree

A decision tree is included as a transparent baseline and to preserve continuity with the original coursework.

### XGBoost

XGBoost is used as a stronger gradient-boosted tree model when the package is available.

The best model is selected by lowest RMSE on the chronological holdout set.

## Metrics

The project reports:

- **MAE** — average absolute prediction error
- **RMSE** — penalizes larger errors more strongly
- **R²** — fraction of target variance explained by the model

For a real business deployment, model quality should also be reviewed by store, season, promotion type and forecast horizon.

## Skills Demonstrated

- Python
- Pandas / NumPy
- data cleaning
- date feature engineering
- regression
- scikit-learn pipelines
- categorical encoding
- Decision Trees
- XGBoost
- time-aware validation
- MAE / RMSE / R²
- model serialization
- testing
- reproducible project structure

## Future Improvements

- walk-forward validation
- lag features
- rolling statistics
- holiday calendars
- promotion features
- store-level evaluation
- SHAP explainability
- hyperparameter tuning
- Power BI or Streamlit reporting
- model monitoring

## Author

**Joshua Miranda**

Data Analytics • Artificial Intelligence • Python • SQL • Machine Learning
