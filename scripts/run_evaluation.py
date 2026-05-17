"""Run the end-to-end ALGS ring prediction experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rings.config import DEFAULT_DATA_PATH, METRICS_DIR, ensure_report_dirs
from rings.evaluation import summarize_errors
from rings.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_report_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    experiment = run_experiment(
        data_path=args.data,
        grid_size=args.grid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    leaderboard_path = args.out_dir / "leaderboard.csv"
    oracle_path = args.out_dir / "oracle_candidates.csv"
    experiment.leaderboard.to_csv(leaderboard_path, index=False)
    experiment.oracle.to_csv(oracle_path, index=True)

    for label, result in experiment.results.items():
        slug = label.lower().replace(" ", "_").replace("=", "").replace(".", "")
        summarize_errors(result.errors).to_csv(args.out_dir / f"{slug}_by_map.csv", index=False)

    print("Leaderboard")
    print(experiment.leaderboard.round(3).to_string(index=False))
    print(f"\nWrote metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
