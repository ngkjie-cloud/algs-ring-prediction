"""Feature engineering for ring-5 center prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import DEFAULT_GRID_SIZE, MAX_COORD, RING5_RADIUS_CAP


def ring_heatmap(
    df: pd.DataFrame,
    ring: int = 5,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Build a 2D histogram of observed ring centers."""

    heatmap, _, _ = np.histogram2d(
        df[f"ring{ring}_x"],
        df[f"ring{ring}_y"],
        bins=grid_size,
        range=[[0, MAX_COORD], [0, MAX_COORD]],
    )
    return heatmap


def create_playability_field(
    heatmap: np.ndarray,
    df: pd.DataFrame,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Smooth ring-5 observations into a probability-like spatial field."""

    cell_width = MAX_COORD / grid_size
    ring5_radius = df["ring5_radius"].median()
    sigma = max(ring5_radius / cell_width, 1e-6)

    smoothed = gaussian_filter(heatmap, sigma=sigma)
    total = smoothed.sum()
    if total > 0:
        smoothed = smoothed / total
    return smoothed


def build_map_fields(
    df: pd.DataFrame,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, np.ndarray]:
    """Build map-specific playability fields from the provided data only."""

    fields: dict[str, np.ndarray] = {}
    for map_name, map_df in df.groupby("map"):
        heatmap = ring_heatmap(map_df, ring=5, grid_size=grid_size)
        fields[map_name] = create_playability_field(heatmap, map_df, grid_size=grid_size)
    return fields


def world_to_grid(x: float, y: float, grid_size: int = DEFAULT_GRID_SIZE) -> tuple[int, int]:
    """Convert world coordinates into bounded grid-cell indices."""

    cell_width = MAX_COORD / grid_size
    ix = int(x // cell_width)
    iy = int(y // cell_width)
    ix = int(np.clip(ix, 0, grid_size - 1))
    iy = int(np.clip(iy, 0, grid_size - 1))
    return ix, iy


def field_value_at(
    field: np.ndarray,
    x: float,
    y: float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> float:
    """Read a playability field value at world coordinates."""

    ix, iy = world_to_grid(x, y, grid_size=grid_size)
    return float(field[ix, iy])


def extract_constrained_field_features(
    row: pd.Series,
    map_fields: dict[str, np.ndarray],
    grid_size: int = DEFAULT_GRID_SIZE,
    ring5_radius_cap: float = RING5_RADIUS_CAP,
) -> pd.Series:
    """Summarize the feasible part of a map field inside the observed ring 3."""

    field = map_fields[row["map"]]
    cell_width = MAX_COORD / grid_size
    centers = (np.arange(grid_size) + 0.5) * cell_width
    gx, gy = np.meshgrid(centers, centers, indexing="ij")

    rel_x = (gx - row["ring3_x"]) / row["ring3_radius"]
    rel_y = (gy - row["ring3_y"]) / row["ring3_radius"]
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    feasible_radius = max(row["ring3_radius"] - ring5_radius_cap, 1)
    feasible_radius_rel = feasible_radius / row["ring3_radius"]
    constrained = field * (rel_dist <= feasible_radius_rel)
    mass = constrained.sum()

    if mass <= 0:
        return pd.Series(
            {
                "field_mass": 0.0,
                "field_com_x": 0.0,
                "field_com_y": 0.0,
                "field_max_x": 0.0,
                "field_max_y": 0.0,
                "field_spread": 0.0,
                "field_forward_mean": 0.0,
                "field_lateral_mean": 0.0,
                "field_forward_mass": 0.0,
            }
        )

    weights = constrained / mass
    com_x = float((weights * rel_x).sum())
    com_y = float((weights * rel_y).sum())

    max_idx = np.unravel_index(np.argmax(constrained), constrained.shape)
    max_x = float(rel_x[max_idx])
    max_y = float(rel_y[max_idx])

    spread = float(np.sqrt((weights * ((rel_x - com_x) ** 2 + (rel_y - com_y) ** 2)).sum()))

    mx = row["ring2_ring3_dx"]
    my = row["ring2_ring3_dy"]
    m_norm = np.sqrt(mx**2 + my**2)
    if m_norm > 0:
        ux = mx / m_norm
        uy = my / m_norm
    else:
        ux, uy = 1.0, 0.0

    forward_coord = rel_x * ux + rel_y * uy
    lateral_coord = -rel_x * uy + rel_y * ux

    return pd.Series(
        {
            "field_mass": float(mass),
            "field_com_x": com_x,
            "field_com_y": com_y,
            "field_max_x": max_x,
            "field_max_y": max_y,
            "field_spread": spread,
            "field_forward_mean": float((weights * forward_coord).sum()),
            "field_lateral_mean": float((weights * lateral_coord).sum()),
            "field_forward_mass": float(weights[forward_coord > 0].sum()),
        }
    )


def build_feature_matrix(
    df: pd.DataFrame,
    map_fields: dict[str, np.ndarray],
    grid_size: int = DEFAULT_GRID_SIZE,
    ring5_radius_cap: float = RING5_RADIUS_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build model-ready features and relative ring-5 targets."""

    out = df.copy()
    out["target_rel_x"] = (out["ring5_x"] - out["ring3_x"]) / out["ring3_radius"]
    out["target_rel_y"] = (out["ring5_y"] - out["ring3_y"]) / out["ring3_radius"]

    out["r1_r2_dx_norm"] = out["ring1_ring2_dx"] / out["ring1_radius"]
    out["r1_r2_dy_norm"] = out["ring1_ring2_dy"] / out["ring1_radius"]
    out["r2_r3_dx_norm"] = out["ring2_ring3_dx"] / out["ring2_radius"]
    out["r2_r3_dy_norm"] = out["ring2_ring3_dy"] / out["ring2_radius"]
    out["r2_r3_dist_norm"] = out["ring2_ring3_dist"] / out["ring2_radius"]
    out["ring2_momentum_sin"] = np.sin(out["ring2_momentum"])
    out["ring2_momentum_cos"] = np.cos(out["ring2_momentum"])
    out["r3_radius_norm"] = out["ring3_radius"] / MAX_COORD
    out["feasible_radius_norm"] = (
        (out["ring3_radius"] - ring5_radius_cap) / out["ring3_radius"]
    )

    field_features = out.apply(
        lambda row: extract_constrained_field_features(
            row,
            map_fields=map_fields,
            grid_size=grid_size,
            ring5_radius_cap=ring5_radius_cap,
        ),
        axis=1,
    )
    out = pd.concat([out, field_features], axis=1)

    out["cand_mom025_x"] = 0.25 * out["ring2_ring3_dx"] / out["ring3_radius"]
    out["cand_mom025_y"] = 0.25 * out["ring2_ring3_dy"] / out["ring3_radius"]
    out["cand_mom05_x"] = 0.50 * out["ring2_ring3_dx"] / out["ring3_radius"]
    out["cand_mom05_y"] = 0.50 * out["ring2_ring3_dy"] / out["ring3_radius"]

    out["mom025_to_com_dist"] = np.sqrt(
        (out["cand_mom025_x"] - out["field_com_x"]) ** 2
        + (out["cand_mom025_y"] - out["field_com_y"]) ** 2
    )
    out["mom05_to_com_dist"] = np.sqrt(
        (out["cand_mom05_x"] - out["field_com_x"]) ** 2
        + (out["cand_mom05_y"] - out["field_com_y"]) ** 2
    )
    out["max_to_com_dist"] = np.sqrt(
        (out["field_max_x"] - out["field_com_x"]) ** 2
        + (out["field_max_y"] - out["field_com_y"]) ** 2
    )

    feature_cols = [
        "r1_r2_dx_norm",
        "r1_r2_dy_norm",
        "r2_r3_dx_norm",
        "r2_r3_dy_norm",
        "r2_r3_dist_norm",
        "ring2_momentum_sin",
        "ring2_momentum_cos",
        "r3_radius_norm",
        "feasible_radius_norm",
        "field_mass",
        "field_com_x",
        "field_com_y",
        "field_max_x",
        "field_max_y",
        "field_spread",
        "field_forward_mean",
        "field_lateral_mean",
        "field_forward_mass",
        "cand_mom025_x",
        "cand_mom025_y",
        "cand_mom05_x",
        "cand_mom05_y",
        "mom025_to_com_dist",
        "mom05_to_com_dist",
        "max_to_com_dist",
    ]

    x = pd.get_dummies(out[feature_cols + ["map"]], columns=["map"])
    y = out[["target_rel_x", "target_rel_y"]]
    return x, y, out, feature_cols


def clip_relative_point_to_feasible(
    rel_x: pd.Series,
    rel_y: pd.Series,
    feasible_radius_rel: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Clip ring-3-relative points to the legal ring-5 center region."""

    dist = np.sqrt(rel_x**2 + rel_y**2)
    scale = feasible_radius_rel / (dist + 1e-12)
    scale = scale.where(dist > feasible_radius_rel, 1.0)
    return rel_x * scale, rel_y * scale


def build_vector_feature_matrix(
    df: pd.DataFrame,
    map_fields: dict[str, np.ndarray],
    grid_size: int = DEFAULT_GRID_SIZE,
    ring5_radius_cap: float = RING5_RADIUS_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build guide-inspired vector features and relative ring-5 targets."""

    out = df.copy()
    out["target_rel_x"] = (out["ring5_x"] - out["ring3_x"]) / out["ring3_radius"]
    out["target_rel_y"] = (out["ring5_y"] - out["ring3_y"]) / out["ring3_radius"]

    out["feasible_radius_rel"] = (
        (out["ring3_radius"] - ring5_radius_cap) / out["ring3_radius"]
    ).clip(lower=0.01)
    out["r2_r1_radius_ratio"] = out["ring2_radius"] / out["ring1_radius"]
    out["r3_r2_radius_ratio"] = out["ring3_radius"] / out["ring2_radius"]
    out["r3_radius_norm"] = out["ring3_radius"] / MAX_COORD

    out["edge_gap_12_norm"] = (
        out["ring1_radius"] - out["ring2_radius"] - out["ring1_ring2_dist"]
    ) / out["ring1_radius"]
    out["edge_gap_23_norm"] = (
        out["ring2_radius"] - out["ring3_radius"] - out["ring2_ring3_dist"]
    ) / out["ring2_radius"]

    field_features = out.apply(
        lambda row: extract_constrained_field_features(
            row,
            map_fields=map_fields,
            grid_size=grid_size,
            ring5_radius_cap=ring5_radius_cap,
        ),
        axis=1,
    )
    out = pd.concat([out, field_features], axis=1)

    v12x = out["ring1_ring2_dx"]
    v12y = out["ring1_ring2_dy"]
    v23x = out["ring2_ring3_dx"]
    v23y = out["ring2_ring3_dy"]

    vector_candidates = {
        "vec_add": (
            out["ring2_x"] + v12x + v23x,
            out["ring2_y"] + v12y + v23y,
        ),
        "vec_counter": (
            out["ring2_x"] + v23x - v12x,
            out["ring2_y"] + v23y - v12y,
        ),
        "vec_alt": (
            out["ring2_x"] + v12x - v23x,
            out["ring2_y"] + v12y - v23y,
        ),
    }

    for prefix, (world_x, world_y) in vector_candidates.items():
        rel_x = (world_x - out["ring3_x"]) / out["ring3_radius"]
        rel_y = (world_y - out["ring3_y"]) / out["ring3_radius"]
        out[f"{prefix}_x"], out[f"{prefix}_y"] = clip_relative_point_to_feasible(
            rel_x,
            rel_y,
            out["feasible_radius_rel"],
        )
        out[f"{prefix}_r"] = np.sqrt(out[f"{prefix}_x"] ** 2 + out[f"{prefix}_y"] ** 2)
        out[f"{prefix}_score"] = out.apply(
            lambda row: field_value_at(
                map_fields[row["map"]],
                row["ring3_x"] + row[f"{prefix}_x"] * row["ring3_radius"],
                row["ring3_y"] + row[f"{prefix}_y"] * row["ring3_radius"],
                grid_size=grid_size,
            ),
            axis=1,
        )
        out[f"{prefix}_to_field_com"] = np.sqrt(
            (out[f"{prefix}_x"] - out["field_com_x"]) ** 2
            + (out[f"{prefix}_y"] - out["field_com_y"]) ** 2
        )
        out[f"{prefix}_to_field_max"] = np.sqrt(
            (out[f"{prefix}_x"] - out["field_max_x"]) ** 2
            + (out[f"{prefix}_y"] - out["field_max_y"]) ** 2
        )

    feature_cols = [
        "feasible_radius_rel",
        "r2_r1_radius_ratio",
        "r3_r2_radius_ratio",
        "r3_radius_norm",
        "edge_gap_12_norm",
        "edge_gap_23_norm",
        "field_mass",
        "field_com_x",
        "field_com_y",
        "field_max_x",
        "field_max_y",
        "field_spread",
        "vec_add_x",
        "vec_add_y",
        "vec_add_r",
        "vec_add_score",
        "vec_add_to_field_com",
        "vec_add_to_field_max",
        "vec_counter_x",
        "vec_counter_y",
        "vec_counter_r",
        "vec_counter_score",
        "vec_counter_to_field_com",
        "vec_counter_to_field_max",
        "vec_alt_x",
        "vec_alt_y",
        "vec_alt_r",
        "vec_alt_score",
        "vec_alt_to_field_com",
        "vec_alt_to_field_max",
    ]

    x = pd.get_dummies(out[feature_cols + ["map"]], columns=["map"])
    y = out[["target_rel_x", "target_rel_y"]]
    return x, y, out, feature_cols
