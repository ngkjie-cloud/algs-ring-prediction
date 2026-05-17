"""Baselines and model helpers for the ring prediction workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_GRID_SIZE, MAX_COORD, RANDOM_STATE


CANDIDATE_FEATURE_COLS = [
    "cand_x",
    "cand_y",
    "cand_r",
    "cand_forward",
    "cand_lateral",
    "candidate_score",
    "dist_to_field_com",
    "dist_to_field_max",
    "field_mass",
    "field_spread",
    "field_forward_mean",
    "field_lateral_mean",
    "field_forward_mass",
    "r2_r3_dist_norm",
    "ring2_momentum_sin",
    "ring2_momentum_cos",
    "feasible_radius_norm",
]


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return simple, interpretable baselines in ring-3-relative coordinates."""

    center = np.zeros((len(df), 2))
    mom025 = np.column_stack(
        [
            0.25 * df["ring2_ring3_dx"].to_numpy() / df["ring3_radius"].to_numpy(),
            0.25 * df["ring2_ring3_dy"].to_numpy() / df["ring3_radius"].to_numpy(),
        ]
    )
    mom05 = np.column_stack(
        [
            0.50 * df["ring2_ring3_dx"].to_numpy() / df["ring3_radius"].to_numpy(),
            0.50 * df["ring2_ring3_dy"].to_numpy() / df["ring3_radius"].to_numpy(),
        ]
    )
    field_com = df[["field_com_x", "field_com_y"]].to_numpy()
    field_max = df[["field_max_x", "field_max_y"]].to_numpy()

    return {
        "Ring 3 center baseline": center,
        "Momentum baseline k=0.25": mom025,
        "Momentum baseline k=0.50": mom05,
        "Constrained field COM baseline": field_com,
        "Constrained field max baseline": field_max,
    }


def make_joint_xgb_regressor(random_state: int = RANDOM_STATE):
    """Create the main joint XGBoost regressor."""

    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=11,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=10.0,
        objective="reg:squarederror",
        multi_strategy="multi_output_tree",
        random_state=random_state,
    )


def make_vector_first_xgb_regressor(random_state: int = RANDOM_STATE):
    """Create the guide-inspired vector-first XGBoost regressor."""

    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=12,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=10.0,
        objective="reg:squarederror",
        multi_strategy="multi_output_tree",
        random_state=random_state,
    )


def make_candidate_ranker(random_state: int = RANDOM_STATE):
    """Create the candidate error regressor."""

    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        objective="reg:squarederror",
        random_state=random_state,
    )


def make_candidate_classifier(pos_weight: float, random_state: int = RANDOM_STATE):
    """Create the candidate best-choice classifier."""

    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=random_state,
    )


def get_field_score_at_relative_point(
    row: pd.Series,
    map_fields: dict[str, np.ndarray],
    rel_x: float,
    rel_y: float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> float:
    """Score a candidate relative point on the map playability field."""

    field = map_fields[row["map"]]
    cell_width = MAX_COORD / grid_size
    world_x = row["ring3_x"] + rel_x * row["ring3_radius"]
    world_y = row["ring3_y"] + rel_y * row["ring3_radius"]
    ix = int(np.clip(world_x // cell_width, 0, grid_size - 1))
    iy = int(np.clip(world_y // cell_width, 0, grid_size - 1))
    return float(field[ix, iy])


def candidate_points(row: pd.Series) -> dict[str, tuple[float, float]]:
    """Return the candidate ring-5 centers considered by the rankers."""

    return {
        "center": (0.0, 0.0),
        "mom025": (row["cand_mom025_x"], row["cand_mom025_y"]),
        "mom05": (row["cand_mom05_x"], row["cand_mom05_y"]),
        "field_com": (row["field_com_x"], row["field_com_y"]),
        "field_max": (row["field_max_x"], row["field_max_y"]),
    }


def build_oracle_candidate_errors(model_df: pd.DataFrame) -> pd.DataFrame:
    """Measure the best possible candidate choice for diagnostic context."""

    true_rel = model_df[["target_rel_x", "target_rel_y"]].to_numpy()
    r3_radius = model_df["ring3_radius"].to_numpy()
    candidate_arrays = {
        name: np.array([candidate_points(row)[name] for _, row in model_df.iterrows()])
        for name in ["center", "mom025", "mom05", "field_com", "field_max"]
    }

    candidate_errors = {}
    for name, pred_rel in candidate_arrays.items():
        rel_error = np.sqrt(
            (pred_rel[:, 0] - true_rel[:, 0]) ** 2
            + (pred_rel[:, 1] - true_rel[:, 1]) ** 2
        )
        candidate_errors[name] = rel_error * r3_radius

    out = pd.DataFrame(candidate_errors, index=model_df.index)
    candidate_cols = list(candidate_arrays.keys())
    out["oracle_best"] = out[candidate_cols].min(axis=1)
    out["best_candidate"] = out[candidate_cols].idxmin(axis=1)
    out["map"] = model_df["map"].to_numpy()
    out["ring5_radius"] = model_df["ring5_radius"].to_numpy()
    out["oracle_norm_error"] = out["oracle_best"] / out["ring5_radius"]
    return out


def build_candidate_ranking_frame(
    model_df: pd.DataFrame,
    map_fields: dict[str, np.ndarray],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> pd.DataFrame:
    """Build one row per match candidate for ranking or classification."""

    rows = []
    for idx, row in model_df.iterrows():
        true_rel_x = row["target_rel_x"]
        true_rel_y = row["target_rel_y"]

        mx = row["ring2_ring3_dx"]
        my = row["ring2_ring3_dy"]
        m_norm = np.sqrt(mx**2 + my**2)
        if m_norm > 0:
            ux = mx / m_norm
            uy = my / m_norm
        else:
            ux, uy = 1.0, 0.0

        for cand_name, (cx, cy) in candidate_points(row).items():
            candidate_error = np.sqrt((cx - true_rel_x) ** 2 + (cy - true_rel_y) ** 2)
            candidate_error *= row["ring3_radius"]
            cand_r = np.sqrt(cx**2 + cy**2)
            cand_forward = cx * ux + cy * uy
            cand_lateral = -cx * uy + cy * ux
            candidate_score = get_field_score_at_relative_point(
                row,
                map_fields,
                cx,
                cy,
                grid_size=grid_size,
            )
            dist_to_field_com = np.sqrt(
                (cx - row["field_com_x"]) ** 2 + (cy - row["field_com_y"]) ** 2
            )
            dist_to_field_max = np.sqrt(
                (cx - row["field_max_x"]) ** 2 + (cy - row["field_max_y"]) ** 2
            )

            rows.append(
                {
                    "match_idx": idx,
                    "map": row["map"],
                    "candidate": cand_name,
                    "cand_x": cx,
                    "cand_y": cy,
                    "cand_r": cand_r,
                    "cand_forward": cand_forward,
                    "cand_lateral": cand_lateral,
                    "candidate_score": candidate_score,
                    "dist_to_field_com": dist_to_field_com,
                    "dist_to_field_max": dist_to_field_max,
                    "field_mass": row["field_mass"],
                    "field_com_x": row["field_com_x"],
                    "field_com_y": row["field_com_y"],
                    "field_max_x": row["field_max_x"],
                    "field_max_y": row["field_max_y"],
                    "field_spread": row["field_spread"],
                    "field_forward_mean": row["field_forward_mean"],
                    "field_lateral_mean": row["field_lateral_mean"],
                    "field_forward_mass": row["field_forward_mass"],
                    "r2_r3_dist_norm": row["r2_r3_dist_norm"],
                    "ring2_momentum_sin": row["ring2_momentum_sin"],
                    "ring2_momentum_cos": row["ring2_momentum_cos"],
                    "feasible_radius_norm": row["feasible_radius_norm"],
                    "candidate_error": candidate_error,
                }
            )

    return pd.DataFrame(rows)


def build_candidate_design_matrices(
    train_cand_df: pd.DataFrame,
    test_cand_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode candidate rows with aligned train/test columns."""

    x_train = pd.get_dummies(
        train_cand_df[CANDIDATE_FEATURE_COLS + ["map", "candidate"]],
        columns=["map", "candidate"],
    )
    x_test = pd.get_dummies(
        test_cand_df[CANDIDATE_FEATURE_COLS + ["map", "candidate"]],
        columns=["map", "candidate"],
    )
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    return x_train, x_test


def choose_lowest_predicted_error(test_cand_df: pd.DataFrame) -> pd.DataFrame:
    """Select the candidate with the lowest predicted error for each match."""

    return (
        test_cand_df.sort_values(["match_idx", "pred_error"])
        .groupby("match_idx")
        .first()
        .reset_index()
    )


def choose_highest_probability(test_cand_df: pd.DataFrame) -> pd.DataFrame:
    """Select the candidate with the highest predicted best-candidate probability."""

    return (
        test_cand_df.sort_values(["match_idx", "pred_best_prob"], ascending=[True, False])
        .groupby("match_idx")
        .first()
        .reset_index()
    )


def add_best_candidate_target(cand_df: pd.DataFrame) -> pd.DataFrame:
    """Flag the oracle-best candidate within each match."""

    out = cand_df.copy()
    out["is_best_candidate"] = (
        out["candidate_error"]
        == out.groupby("match_idx")["candidate_error"].transform("min")
    ).astype(int)
    return out
