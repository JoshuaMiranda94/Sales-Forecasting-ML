from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .evaluate import regression_metrics, save_evaluation
from .features import chronological_split, engineer_date_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def candidate_models(random_state: int = 42):
    models = {
        "decision_tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
        )
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not available; continuing with Decision Tree only.")

    return models


def train(data_path: Path, target: str = "sales", date_col: str = "date") -> None:
    df = pd.read_csv(data_path)

    missing = [c for c in [target, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[target, date_col]).copy()
    train_df, test_df = chronological_split(df, date_col=date_col, train_fraction=0.8)

    train_dates = pd.to_datetime(train_df[date_col])
    test_dates = pd.to_datetime(test_df[date_col])

    train_features = engineer_date_features(train_df, date_col=date_col)
    test_features = engineer_date_features(test_df, date_col=date_col)

    X_train = train_features.drop(columns=[target, date_col])
    y_train = train_features[target]

    X_test = test_features.drop(columns=[target, date_col])
    y_test = test_features[target]

    results = {}
    fitted = {}

    for name, model in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = regression_metrics(y_test, predictions)
        results[name] = metrics
        fitted[name] = (pipeline, predictions)

        print(
            f"{name}: "
            f"MAE={metrics['mae']:.2f} "
            f"RMSE={metrics['rmse']:.2f} "
            f"R2={metrics['r2']:.4f}"
        )

    best_name = min(results, key=lambda name: results[name]["rmse"])
    best_pipeline, best_predictions = fitted[best_name]

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    joblib.dump(best_pipeline, output_dir / "best_model.joblib")

    (output_dir / "all_model_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    save_evaluation(
        y_true=y_test.to_numpy(),
        y_pred=best_predictions,
        dates=test_dates.to_numpy(),
        output_dir=output_dir,
        model_name=best_name,
        metrics=results[best_name],
    )

    print(f"Best model: {best_name}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train retail sales regression models.")
    parser.add_argument("--data", type=Path, required=True, help="Path to sales CSV.")
    parser.add_argument("--target", default="sales")
    parser.add_argument("--date-column", default="date")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, target=args.target, date_col=args.date_column)
