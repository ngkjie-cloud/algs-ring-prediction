"""Prediction evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RING5_RADIUS_CAP


@dataclass
class PredictionResult:
    """Evaluation output for one model or baseline."""

    label: str
    predictions: np.ndarray
    errors: pd.DataFrame


def rel_to_abs(pred_rel: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert predictions relative to ring 3 back to world coordinates."""

    pred_x = pred_rel[:, 0] * df["ring3_radius"].to_numpy() + df["ring3_x"].to_numpy()
    pred_y = pred_rel[:, 1] * df["ring3_radius"].to_numpy() + df["ring3_y"].to_numpy()
    return pred_x, pred_y


def clip_to_feasible_ring3(
    pred_rel: np.ndarray,
    df: pd.DataFrame,
    ring5_radius_cap: float = RING5_RADIUS_CAP,
) -> np.ndarray:
    """Constrain a ring-5 center so the ring can fit within ring 3."""

    pred = pred_rel.copy()
    max_rel = (df["ring3_radius"].to_numpy() - ring5_radius_cap) / df["ring3_radius"].to_numpy()
    max_rel = np.maximum(max_rel, 0.01)
    dist = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
    outside = dist > max_rel
    pred[outside, 0] *= max_rel[outside] / (dist[outside] + 1e-12)
    pred[outside, 1] *= max_rel[outside] / (dist[outside] + 1e-12)
    return pred


def prediction_errors(
    pred_rel: np.ndarray,
    df: pd.DataFrame,
    ring5_radius_cap: float = RING5_RADIUS_CAP,
) -> pd.DataFrame:
    """Return raw, clipped, and ring-radius-normalized errors."""

    pred_rel_clip = clip_to_feasible_ring3(pred_rel, df, ring5_radius_cap=ring5_radius_cap)
    pred_x, pred_y = rel_to_abs(pred_rel, df)
    pred_x_clip, pred_y_clip = rel_to_abs(pred_rel_clip, df)

    true_x = df["ring5_x"].to_numpy()
    true_y = df["ring5_y"].to_numpy()

    out = df[["map", "ring5_radius"]].copy()
    out["raw_error"] = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
    out["clipped_error"] = np.sqrt(
        (pred_x_clip - true_x) ** 2 + (pred_y_clip - true_y) ** 2
    )
    out["norm_error"] = out["clipped_error"] / out["ring5_radius"]
    return out


def evaluate_predictions(
    pred_rel: np.ndarray,
    df: pd.DataFrame,
    label: str,
    ring5_radius_cap: float = RING5_RADIUS_CAP,
) -> PredictionResult:
    """Evaluate one prediction matrix."""

    errors = prediction_errors(pred_rel, df, ring5_radius_cap=ring5_radius_cap)
    return PredictionResult(label=label, predictions=pred_rel, errors=errors)


def summarize_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize errors by map with compact column names."""

    summary = errors.groupby("map")[["raw_error", "clipped_error", "norm_error"]].agg(
        ["mean", "median", "count"]
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    return summary.reset_index()


def leaderboard_from_results(results: dict[str, PredictionResult]) -> pd.DataFrame:
    """Build a one-row-per-model leaderboard."""

    rows = []
    for label, result in results.items():
        errors = result.errors
        rows.append(
            {
                "model": label,
                "mean_raw_error": errors["raw_error"].mean(),
                "mean_clipped_error": errors["clipped_error"].mean(),
                "median_clipped_error": errors["clipped_error"].median(),
                "mean_norm_error": errors["norm_error"].mean(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_clipped_error", ascending=True)
        .reset_index(drop=True)
    )
