from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    stores = [1, 2, 3]

    rows = []
    for date in dates:
        for store in stores:
            weekday = date.weekday()
            weekend = weekday >= 5
            promo = int(rng.random() < 0.20)
            customers = int(
                550
                + store * 45
                + (110 if weekend else 0)
                + promo * 130
                + rng.normal(0, 45)
            )

            state_holiday = "none"
            if date.month == 12 and date.day in {24, 25, 31}:
                state_holiday = "holiday"

            sales = (
                1800
                + store * 250
                + customers * 11.5
                + promo * 900
                + (550 if weekend else 0)
                + (700 if date.month in {11, 12} else 0)
                + rng.normal(0, 600)
            )

            rows.append(
                {
                    "date": date.date().isoformat(),
                    "store": store,
                    "state_holiday": state_holiday,
                    "promo": promo,
                    "nb_customers_on_day": max(customers, 0),
                    "sales": max(round(sales, 2), 0),
                }
            )

    df = pd.DataFrame(rows)
    output = Path(__file__).resolve().parents[1] / "data" / "demo_sales.csv"
    df.to_csv(output, index=False)

    print(f"Created {len(df):,} synthetic rows at {output}")


if __name__ == "__main__":
    main()
