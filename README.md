# ALGS Ring Prediction

**Full report:** [PDF](reports/report.pdf) | [Markdown](reports/report.md)

Predicting Apex Legends Global Series ring-5 center locations from early ring movement and historical map playability.

## Project Question

Given the first three ring centers in a competitive Apex Legends match, can we predict the ring-5 center well enough to beat simple geometric baselines?

The final model predicts ring-5 as an offset from ring 3. That keeps predictions comparable across differently sized rings and allows evaluation in world-coordinate error and ring-radius-normalized error.

## Workflow

1. Load raw ALGS match ring records from a local CSV matching the expected schema.
2. Filter to complete ring-5 examples on the three maps with enough data: Storm Point, World's Edge, and Broken Moon.
3. Explore ring-4 and ring-5 spatial distributions over map images.
4. Build playability fields from historical ring-5 locations.
5. Split train/test with map stratification.
6. Build modeling features from ring movement, ring scale, feasible ring-3 geometry, and map playability.
7. Evaluate interpretable baselines.
8. Train and evaluate XGBoost regression and candidate-selection models.
9. Save model leaderboard and by-map evaluation tables to `reports/metrics`.

One important cleanup from the original notebook: the modeling playability field is now built from the training split only. The full dataset is still used for EDA, but test-set ring-5 locations no longer leak into model features.

## Current Results

Run on the local development dataset with `random_state=42`, `test_size=0.2`, and a 50x50 playability grid:

| model | mean clipped error | median clipped error | mean normalized error |
|---|---:|---:|---:|
| Guide-inspired vector XGBoost | 610.876 | 586.677 | 1.761 |
| Joint XGBoost | 619.263 | 598.943 | 1.786 |
| Candidate ranker | 643.507 | 599.890 | 1.859 |
| Momentum baseline k=0.25 | 646.000 | 653.236 | 1.864 |
| Momentum baseline k=0.50 | 648.381 | 617.387 | 1.869 |
| Constrained field COM baseline | 666.388 | 691.642 | 1.925 |
| Ring 3 center baseline | 685.353 | 727.632 | 1.981 |
| Candidate classifier | 720.156 | 635.391 | 2.071 |
| Constrained field max baseline | 841.874 | 841.590 | 2.425 |

The strongest current result is the guide-inspired vector XGBoost regressor. It translates player-style ring prediction knowledge into vector-addition, counterpull, edge-gap, and playability-agreement features, then evaluates those features with the same leakage-free train/test split as the rest of the project.

## Repository Layout

```text
.
|-- assets/maps/                     # Map images used for EDA plots
|-- data/                            # Public schema/sample files; raw local CSV ignored
|-- notebooks/
|   `-- 01_algs_ring_prediction_case_study.ipynb
|-- reports/
|   |-- report.md                     # Written case-study report
|   `-- metrics/                      # Generated leaderboard and by-map metrics
|-- rings/                           # Reusable project package
|-- scripts/
|   `-- run_evaluation.py            # End-to-end experiment runner
`-- tests/                           # Lightweight smoke tests
```

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the full evaluation after placing a compatible dataset at `data/apex_matches_raw.csv`:

```bash
python scripts/run_evaluation.py
```

Run smoke tests:

```bash
python -m pytest
```

Open the cleaned case-study notebook:

```bash
jupyter notebook notebooks/01_algs_ring_prediction_case_study.ipynb
```

The polished notebook and `reports/report.md` are intended as the GitHub/resume-facing case study. The rough local EDA notebook is ignored so the public repo stays focused on the reproducible workflow.

## Data Notes

The raw development dataset is not included in this repository because it may be subject to source-data restrictions. To reproduce the project, provide your own dataset with the expected schema at `data/apex_matches_raw.csv` or pass a custom path to the evaluation script with `--data`.

Public data helpers live in `data/`:

- `apex_matches_schema.csv` contains the expected input columns.
- `sample_matches.csv` contains a tiny synthetic example for inspecting the file shape.

Private local data files are ignored by Git.
