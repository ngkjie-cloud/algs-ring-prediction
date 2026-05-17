# Data

The raw development dataset is not included in this repository because it may be subject to source-data restrictions.

The repository contains the feature engineering, modeling, and evaluation pipeline. To reproduce the project, provide your own dataset with the expected schema at `data/apex_matches_raw.csv`, or pass a custom CSV path to `scripts/run_evaluation.py` with `--data`.

Included public files:

- `apex_matches_schema.csv`: header-only schema for a compatible input dataset.
- `sample_matches.csv`: tiny synthetic example for inspecting the expected shape. It is not intended for model training or benchmarking.
