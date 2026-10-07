# Model Card

## Intended Use

Educational / portfolio demonstration of retail sales regression.

## Target

`sales`

## Validation Strategy

Chronological 80/20 holdout split.

## Candidate Models

- Decision Tree Regressor
- XGBoost Regressor

## Metrics

- MAE
- RMSE
- R²

## Limitations

- A single chronological holdout is weaker than walk-forward validation.
- Some operational features may not be known before prediction time.
- The synthetic demo dataset does not represent real business performance.
- No causal claims should be made from model feature importance.
- External economic, weather and event data are not included.

## Recommended Production Extensions

- walk-forward cross-validation
- explicit forecast horizon
- lagged target features
- rolling means
- holiday / promotion calendars
- store-level error analysis
- monitoring for drift
