import numpy as np

from rings.config import PROJECT_ROOT
from rings.data import prepare_model_data
from rings.evaluation import evaluate_predictions
from rings.features import build_feature_matrix, build_map_fields, build_vector_feature_matrix


SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_matches.csv"


def test_feature_pipeline_builds_model_matrix():
    df = prepare_model_data(SAMPLE_DATA_PATH)
    fields = build_map_fields(df, grid_size=20)
    x, y, model_df, feature_cols = build_feature_matrix(df, fields, grid_size=20)

    assert len(x) == len(y) == len(model_df)
    assert "r2_r3_dx_norm" in feature_cols
    assert not x.isna().any().any()


def test_vector_feature_pipeline_builds_model_matrix():
    df = prepare_model_data(SAMPLE_DATA_PATH)
    fields = build_map_fields(df, grid_size=20)
    x, y, model_df, feature_cols = build_vector_feature_matrix(df, fields, grid_size=20)

    assert len(x) == len(y) == len(model_df)
    assert "vec_counter_to_field_max" in feature_cols
    assert not x.isna().any().any()


def test_evaluation_outputs_expected_columns():
    df = prepare_model_data(SAMPLE_DATA_PATH)
    fields = build_map_fields(df, grid_size=20)
    _, _, model_df, _ = build_feature_matrix(df, fields, grid_size=20)
    pred = np.zeros((len(model_df), 2))

    result = evaluate_predictions(pred, model_df, label="smoke")

    assert {"raw_error", "clipped_error", "norm_error"}.issubset(result.errors.columns)
    assert result.errors["clipped_error"].ge(0).all()
