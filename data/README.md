# Data

The original coursework dataset is not bundled into this portfolio repository.

## Expected minimum schema

```text
date
sales
```

The training pipeline automatically detects other numeric and categorical features.

## Demo data

Run:

```bash
python scripts/generate_demo_data.py
```

This creates:

```text
data/demo_sales.csv
```

The demo data is synthetic and exists only to make the project runnable without distributing the original coursework dataset.
