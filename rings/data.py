"""Loading and preparing ALGS ring match data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_DATA_PATH, SUPPORTED_MAPS


def ring_columns(max_ring: int = 5) -> list[str]:
    """Return the canonical columns used by the modeling workflow."""

    columns = ["match_id", "map", "total_rings"]
    for ring in range(1, max_ring + 1):
        columns.extend(
            [
                f"ring{ring}_stage",
                f"ring{ring}_x",
                f"ring{ring}_y",
                f"ring{ring}_radius",
                f"ring{ring}_shrink",
                f"ring{ring}_t",
                f"ring{ring}_ts",
            ]
        )
    return columns


def load_matches(
    path: str | Path = DEFAULT_DATA_PATH,
    maps: Iterable[str] = SUPPORTED_MAPS,
    require_ring: int = 5,
) -> pd.DataFrame:
    """Load raw match records and keep complete examples for the target ring."""

    path = Path(path)
    df = pd.read_csv(path)
    keep_cols = [col for col in ring_columns(max_ring=require_ring) if col in df.columns]
    df = df[keep_cols].copy()
    df = df[df["map"].isin(tuple(maps))]
    df = df.dropna(subset=[f"ring{require_ring}_x", f"ring{require_ring}_y"])
    df = df.reset_index(drop=True)
    return df


def add_ring_vectors(df: pd.DataFrame, max_transition: int = 4) -> pd.DataFrame:
    """Add ring-to-ring movement, distance, angle, and momentum features."""

    out = df.copy()
    for ring in range(1, max_transition + 1):
        curr = f"ring{ring}"
        nxt = f"ring{ring + 1}"

        dx = out[f"{nxt}_x"] - out[f"{curr}_x"]
        dy = out[f"{nxt}_y"] - out[f"{curr}_y"]

        out[f"{curr}_{nxt}_dx"] = dx
        out[f"{curr}_{nxt}_dy"] = dy
        out[f"{curr}_{nxt}_dist"] = np.sqrt(dx**2 + dy**2)
        out[f"{curr}_{nxt}_angle"] = np.arctan2(dy, dx)

        if ring > 1:
            prev_angle = out[f"ring{ring - 1}_{curr}_angle"]
            curr_angle = out[f"{curr}_{nxt}_angle"]
            diff = curr_angle - prev_angle
            out[f"ring{ring}_momentum"] = np.arctan2(np.sin(diff), np.cos(diff))

    return out


def prepare_model_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load complete supported-map matches and add reusable vector features."""

    matches = load_matches(path)
    return add_ring_vectors(matches)
