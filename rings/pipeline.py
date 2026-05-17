"""End-to-end experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_DATA_PATH, DEFAULT_GRID_SIZE, RANDOM_STATE
from .data import prepare_model_data
from .evaluation import PredictionResult, evaluate_predictions, leaderboard_from_results
from .features import build_feature_matrix, build_map_fields, build_vector_feature_matrix
from .models import (
    add_best_candidate_target,
    baseline_predictions,
    build_candidate_design_matrices,
    build_candidate_ranking_frame,
    build_oracle_candidate_errors,
    choose_highest_probability,
    choose_lowest_predicted_error,
    make_candidate_classifier,
    make_candidate_ranker,
    make_joint_xgb_regressor,
    make_vector_first_xgb_regressor,
)


@dataclass
class ExperimentResult:
    """Container for the complete experiment output."""

    raw_matches: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_model_df: pd.DataFrame
    test_model_df: pd.DataFrame
    map_fields: dict
    results: dict[str, PredictionResult]
    leaderboard: pd.DataFrame
    oracle: pd.DataFrame
    ranker_debug: pd.DataFrame
    classifier_debug: pd.DataFrame


def run_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    grid_size: int = DEFAULT_GRID_SIZE,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> ExperimentResult:
    """Run the portfolio workflow from raw CSV to model leaderboard."""

    matches = prepare_model_data(data_path)
    train_df, test_df = train_test_split(
        matches,
        test_size=test_size,
        random_state=random_state,
        stratify=matches["map"],
    )

    map_fields = build_map_fields(train_df, grid_size=grid_size)

    x_train, y_train, train_model_df, _ = build_feature_matrix(
        train_df,
        map_fields=map_fields,
        grid_size=grid_size,
    )
    x_test, _, test_model_df, _ = build_feature_matrix(
        test_df,
        map_fields=map_fields,
        grid_size=grid_size,
    )
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    x_vec_train, y_vec_train, _, _ = build_vector_feature_matrix(
        train_df,
        map_fields=map_fields,
        grid_size=grid_size,
    )
    x_vec_test, _, test_vec_df, _ = build_vector_feature_matrix(
        test_df,
        map_fields=map_fields,
        grid_size=grid_size,
    )
    x_vec_test = x_vec_test.reindex(columns=x_vec_train.columns, fill_value=0)

    results: dict[str, PredictionResult] = {}
    for label, pred_rel in baseline_predictions(test_model_df).items():
        results[label] = evaluate_predictions(pred_rel, test_model_df, label=label)

    joint_xgb = make_joint_xgb_regressor(random_state=random_state)
    joint_xgb.fit(x_train, y_train)
    joint_pred = joint_xgb.predict(x_test)
    results["Joint XGBoost"] = evaluate_predictions(
        joint_pred,
        test_model_df,
        label="Joint XGBoost",
    )

    vector_xgb = make_vector_first_xgb_regressor(random_state=random_state)
    vector_xgb.fit(x_vec_train, y_vec_train)
    vector_pred = vector_xgb.predict(x_vec_test)
    results["Guide-inspired vector XGBoost"] = evaluate_predictions(
        vector_pred,
        test_vec_df,
        label="Guide-inspired vector XGBoost",
    )

    oracle = build_oracle_candidate_errors(test_model_df)

    train_cand_df = build_candidate_ranking_frame(
        train_model_df,
        map_fields=map_fields,
        grid_size=grid_size,
    )
    test_cand_df = build_candidate_ranking_frame(
        test_model_df,
        map_fields=map_fields,
        grid_size=grid_size,
    )

    x_rank_train, x_rank_test = build_candidate_design_matrices(train_cand_df, test_cand_df)
    ranker = make_candidate_ranker(random_state=random_state)
    ranker.fit(x_rank_train, train_cand_df["candidate_error"])
    ranked_candidates = test_cand_df.copy()
    ranked_candidates["pred_error"] = ranker.predict(x_rank_test)

    chosen_ranker = choose_lowest_predicted_error(ranked_candidates)
    chosen_ranker_df = test_model_df.loc[chosen_ranker["match_idx"]].copy()
    chosen_ranker_pred = chosen_ranker[["cand_x", "cand_y"]].to_numpy()
    results["Candidate ranker"] = evaluate_predictions(
        chosen_ranker_pred,
        chosen_ranker_df,
        label="Candidate ranker",
    )

    ranker_debug = _candidate_debug_frame(chosen_ranker, oracle)

    train_cls_df = add_best_candidate_target(train_cand_df)
    test_cls_df = add_best_candidate_target(test_cand_df)
    x_cls_train, x_cls_test = build_candidate_design_matrices(train_cls_df, test_cls_df)
    y_cls_train = train_cls_df["is_best_candidate"]
    pos_weight = (len(y_cls_train) - y_cls_train.sum()) / y_cls_train.sum()

    classifier = make_candidate_classifier(pos_weight=pos_weight, random_state=random_state)
    classifier.fit(x_cls_train, y_cls_train)
    classified_candidates = test_cls_df.copy()
    classified_candidates["pred_best_prob"] = classifier.predict_proba(x_cls_test)[:, 1]

    chosen_classifier = choose_highest_probability(classified_candidates)
    chosen_classifier_df = test_model_df.loc[chosen_classifier["match_idx"]].copy()
    chosen_classifier_pred = chosen_classifier[["cand_x", "cand_y"]].to_numpy()
    results["Candidate classifier"] = evaluate_predictions(
        chosen_classifier_pred,
        chosen_classifier_df,
        label="Candidate classifier",
    )

    classifier_debug = _candidate_debug_frame(chosen_classifier, oracle)
    leaderboard = leaderboard_from_results(results)

    return ExperimentResult(
        raw_matches=matches,
        train_df=train_df,
        test_df=test_df,
        train_model_df=train_model_df,
        test_model_df=test_model_df,
        map_fields=map_fields,
        results=results,
        leaderboard=leaderboard,
        oracle=oracle,
        ranker_debug=ranker_debug,
        classifier_debug=classifier_debug,
    )


def _candidate_debug_frame(chosen_candidates: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    """Join chosen candidates to oracle diagnostics."""

    debug = chosen_candidates[["match_idx", "candidate", "candidate_error"]].copy()
    if "pred_error" in chosen_candidates.columns:
        debug["pred_error"] = chosen_candidates["pred_error"]
    if "pred_best_prob" in chosen_candidates.columns:
        debug["pred_best_prob"] = chosen_candidates["pred_best_prob"]

    debug = debug.merge(
        oracle[["oracle_best", "best_candidate", "map"]],
        left_on="match_idx",
        right_index=True,
        how="left",
    )
    debug["oracle_gap"] = debug["candidate_error"] - debug["oracle_best"]
    debug["picked_oracle_candidate"] = debug["candidate"] == debug["best_candidate"]
    return debug
