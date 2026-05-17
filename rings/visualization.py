"""Plot helpers for EDA and reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle

from .config import MAP_CONFIGS, MAP_EXTENT, MAPS_DIR, MAX_COORD, SUPPORTED_MAPS
from .evaluation import clip_to_feasible_ring3, rel_to_abs
from .features import build_map_fields, ring_heatmap


def _draw_map_background(ax, map_name: str, maps_dir: Path = MAPS_DIR) -> None:
    config = MAP_CONFIGS[map_name]
    image_path = config.image_path(maps_dir)
    if image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img, extent=MAP_EXTENT)
    else:
        ax.set_facecolor("#151515")


def plot_ring_distributions(
    df,
    rings: tuple[int, ...] = (4, 5),
    maps: tuple[str, ...] = SUPPORTED_MAPS,
    maps_dir: Path = MAPS_DIR,
):
    """Plot observed ring center density over the map images."""

    fig, axes = plt.subplots(len(rings), len(maps), figsize=(7 * len(maps), 6 * len(rings)))
    axes = np.atleast_2d(axes)
    for row_idx, ring in enumerate(rings):
        for col_idx, map_name in enumerate(maps):
            ax = axes[row_idx, col_idx]
            _draw_map_background(ax, map_name, maps_dir=maps_dir)
            x_col, y_col = f"ring{ring}_x", f"ring{ring}_y"
            map_df = df[df["map"] == map_name].dropna(subset=[x_col, y_col])
            if not map_df.empty:
                sns.kdeplot(
                    data=map_df,
                    x=x_col,
                    y=y_col,
                    ax=ax,
                    fill=True,
                    alpha=0.42,
                    cmap="magma",
                    levels=12,
                    thresh=0.1,
                )
                ax.scatter(map_df[x_col], map_df[y_col], c="cyan", s=3, alpha=0.2)
            ax.set_title(f"{MAP_CONFIGS[map_name].display_name}: Ring {ring} ({len(map_df)} games)")
            ax.set_xlim(0, MAX_COORD)
            ax.set_ylim(MAX_COORD, 0)
            ax.axis("off")
    fig.suptitle("Spatial Distribution of Late-Game Ring Centers", fontsize=18)
    fig.tight_layout()
    return fig


def plot_playability_fields(
    df,
    grid_size: int = 50,
    maps: tuple[str, ...] = SUPPORTED_MAPS,
    maps_dir: Path = MAPS_DIR,
):
    """Plot smoothed ring-5 playability fields built from the supplied data."""

    fields = build_map_fields(df, grid_size=grid_size)
    fig, axes = plt.subplots(1, len(maps), figsize=(7 * len(maps), 6))
    axes = np.atleast_1d(axes)
    for ax, map_name in zip(axes, maps):
        _draw_map_background(ax, map_name, maps_dir=maps_dir)
        field = fields[map_name]
        im = ax.imshow(
            field.T,
            origin="upper",
            extent=MAP_EXTENT,
            cmap="magma",
            alpha=0.62,
        )
        ax.set_title(f"{MAP_CONFIGS[map_name].display_name}: Ring 5 playability")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.65)
    fig.tight_layout()
    return fig


def plot_ring5_heatmap(
    df,
    map_name: str,
    grid_size: int = 50,
    maps_dir: Path = MAPS_DIR,
):
    """Plot a raw ring-5 heatmap for one map."""

    map_df = df[df["map"] == map_name]
    heatmap = ring_heatmap(map_df, ring=5, grid_size=grid_size)
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_map_background(ax, map_name, maps_dir=maps_dir)
    im = ax.imshow(
        heatmap.T,
        origin="upper",
        extent=MAP_EXTENT,
        cmap="magma",
        alpha=0.62,
    )
    ax.set_title(f"{MAP_CONFIGS[map_name].display_name}: raw ring-5 endings")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.75, label="matches")
    fig.tight_layout()
    return fig


def select_representative_test_samples(
    experiment,
    model_name: str | None = None,
    maps: tuple[str, ...] = SUPPORTED_MAPS,
) -> dict[str, int]:
    """Pick one test match per map with near-median model error."""

    if model_name is None:
        model_name = experiment.leaderboard.loc[0, "model"]

    errors = experiment.results[model_name].errors
    sample_indices: dict[str, int] = {}
    for map_name in maps:
        map_errors = errors[errors["map"] == map_name]
        if map_errors.empty:
            continue
        median_error = map_errors["clipped_error"].median()
        sample_idx = (map_errors["clipped_error"] - median_error).abs().idxmin()
        sample_indices[map_name] = int(sample_idx)
    return sample_indices


def select_random_test_samples(
    experiment,
    maps: tuple[str, ...] = SUPPORTED_MAPS,
    random_state: int | None = 42,
) -> dict[str, int]:
    """Pick one random test match per map."""

    rng = np.random.default_rng(random_state)
    sample_indices: dict[str, int] = {}
    for map_name in maps:
        map_indices = experiment.test_model_df.index[
            experiment.test_model_df["map"] == map_name
        ].to_numpy()
        if len(map_indices) == 0:
            continue
        sample_indices[map_name] = int(rng.choice(map_indices))
    return sample_indices


def _prediction_for_match(result, match_idx: int, row_df: pd.DataFrame) -> tuple[float, float]:
    """Return a clipped absolute prediction for one result and match."""

    matches = np.flatnonzero(result.errors.index.to_numpy() == match_idx)
    if len(matches) == 0:
        raise KeyError(f"Match index {match_idx} is not present for {result.label}.")

    pred_rel = result.predictions[matches[0] : matches[0] + 1]
    pred_rel = clip_to_feasible_ring3(pred_rel, row_df)
    pred_x, pred_y = rel_to_abs(pred_rel, row_df)
    return float(pred_x[0]), float(pred_y[0])


def plot_model_predictions_for_match(
    experiment,
    match_idx: int,
    models: list[str] | None = None,
    maps_dir: Path = MAPS_DIR,
    ring3_edgecolor: str = "#FF2D2D",
    show_error_table: bool = True,
    prediction_alpha: float = 0.82,
):
    """Plot true Ring 5 and model predictions for one test match."""

    if models is None:
        models = experiment.leaderboard["model"].tolist()

    row_df = experiment.test_model_df.loc[[match_idx]]
    row = row_df.iloc[0]
    map_name = row["map"]

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_map_background(ax, map_name, maps_dir=maps_dir)

    ring3 = Circle(
        (row["ring3_x"], row["ring3_y"]),
        row["ring3_radius"],
        fill=False,
        linewidth=2.8,
        edgecolor=ring3_edgecolor,
        alpha=0.96,
        label="Ring 3",
        zorder=5,
    )
    ring5 = Circle(
        (row["ring5_x"], row["ring5_y"]),
        row["ring5_radius"],
        fill=False,
        linewidth=2.2,
        edgecolor="#7CFF6B",
        alpha=0.95,
        label="True Ring 5",
        zorder=5,
    )
    ax.add_patch(ring3)
    ax.add_patch(ring5)
    ax.scatter(
        row["ring5_x"],
        row["ring5_y"],
        s=170,
        marker="*",
        c="#7CFF6B",
        edgecolor="black",
        linewidth=0.8,
        label="True Ring 5 center",
        zorder=7,
    )

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(models), 1)))
    error_rows = []
    for color, model in zip(colors, models):
        pred_x, pred_y = _prediction_for_match(experiment.results[model], match_idx, row_df)
        model_errors = experiment.results[model].errors.loc[match_idx]
        clipped_error = float(model_errors["clipped_error"])
        ax.scatter(
            pred_x,
            pred_y,
            s=86,
            marker="o",
            c=[color],
            edgecolor="black",
            linewidth=0.75,
            alpha=prediction_alpha,
            label=model,
            zorder=6,
        )
        error_rows.append((model, clipped_error))

    ax.set_title(
        f"{MAP_CONFIGS[map_name].display_name}: model predictions vs true Ring 5 "
        f"(test match {match_idx})"
    )
    ax.set_xlim(0, MAX_COORD)
    ax.set_ylim(MAX_COORD, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)
    if show_error_table:
        error_text = "\n".join(
            f"{model}: {error:,.0f} units" for model, error in error_rows
        )
        ax.text(
            1.01,
            0.02,
            "Clipped error\n" + error_text,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "alpha": 0.92,
            },
        )
    fig.tight_layout()
    return fig, ax


def sample_prediction_errors(experiment, match_idx: int, models: list[str] | None = None):
    """Return clipped errors for each model on one test match."""

    if models is None:
        models = experiment.leaderboard["model"].tolist()

    rows = []
    row = experiment.test_model_df.loc[match_idx]
    for model in models:
        error_row = experiment.results[model].errors.loc[match_idx]
        rows.append(
            {
                "map": row["map"],
                "match_idx": match_idx,
                "model": model,
                "clipped_error_units": error_row["clipped_error"],
                "ring5_radius": row["ring5_radius"],
                "error_in_ring5_radii": error_row["norm_error"],
            }
        )
    return pd.DataFrame(rows)


def plot_error_distribution_with_radius_refs(
    experiment,
    models: list[str] | None = None,
    reference_multiples: tuple[int, ...] = (1, 2, 3),
):
    """Plot clipped-error distributions with Ring 5 radius reference lines."""

    if models is None:
        models = experiment.leaderboard["model"].tolist()

    frames = []
    for model in models:
        errors = experiment.results[model].errors.copy()
        errors["model"] = model
        frames.append(errors)
    if not frames:
        raise ValueError("At least one model is required for the error distribution plot.")
    error_df = pd.concat(frames, ignore_index=True)

    median_ring5_radius = experiment.test_model_df["ring5_radius"].median()

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.boxplot(
        data=error_df,
        x="clipped_error",
        y="model",
        order=models,
        ax=ax,
        color="#DDE7F2",
        fliersize=2.5,
        linewidth=1.1,
    )
    sns.stripplot(
        data=error_df,
        x="clipped_error",
        y="model",
        order=models,
        ax=ax,
        color="#1F2937",
        alpha=0.24,
        size=2.4,
        jitter=0.22,
    )

    for multiple in reference_multiples:
        x_value = median_ring5_radius * multiple
        ax.axvline(
            x_value,
            color="#C2410C",
            linestyle="--",
            linewidth=1.2,
            alpha=0.88,
        )
        ax.text(
            x_value,
            1.01,
            f"{multiple}x median radius of ring 5",
            transform=ax.get_xaxis_transform(),
            rotation=45,
            va="bottom",
            ha="center",
            color="#9A3412",
            fontsize=8,
        )

    ax.set_title("Prediction error distribution by model", y=-0.2)
    ax.set_xlabel("Clipped error in map units")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.24)
    fig.tight_layout()
    return fig, ax
