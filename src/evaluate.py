from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_evaluation(
    y_true,
    y_pred,
    dates,
    output_dir: Path,
    model_name: str,
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).astype(str),
            "actual_sales": np.asarray(y_true),
            "predicted_sales": np.asarray(y_pred),
        }
    )
    predictions["residual"] = predictions["actual_sales"] - predictions["predicted_sales"]
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    (output_dir / "metrics.json").write_text(
        json.dumps({"best_model": model_name, **metrics}, indent=2),
        encoding="utf-8",
    )

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(predictions["actual_sales"], predictions["predicted_sales"], alpha=0.35)
    minimum = min(predictions["actual_sales"].min(), predictions["predicted_sales"].min())
    maximum = max(predictions["actual_sales"].max(), predictions["predicted_sales"].max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.tight_layout()
    fig.savefig(output_dir / "prediction_scatter.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    plt.scatter(predictions["predicted_sales"], predictions["residual"], alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residual")
    plt.title(f"{model_name}: Residuals")
    plt.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=150)
    plt.close(fig)
