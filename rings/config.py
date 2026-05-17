"""Project-level constants and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
MAPS_DIR = ASSETS_DIR / "maps"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_DATA_PATH = DATA_DIR / "apex_matches_raw.csv"

# constants
SCALE = 4.0
IMG_DIM = 4096
MAX_COORD = int(IMG_DIM * SCALE)
MAP_EXTENT = [0, MAX_COORD, MAX_COORD, 0]
DEFAULT_GRID_SIZE = 50
RING5_RADIUS_CAP = 381
RANDOM_STATE = 42


@dataclass(frozen=True)
class MapConfig:
    """Display metadata for a supported competitive map."""

    display_name: str
    image_filename: str

    def image_path(self, maps_dir: Path = MAPS_DIR) -> Path:
        return maps_dir / self.image_filename


MAP_CONFIGS = {
    "StormPoint": MapConfig(
        display_name="Storm Point",
        image_filename="mp_rr_tropic_island_mu2.png",
    ),
    "WorldsEdge": MapConfig(
        display_name="World's Edge",
        image_filename="mp_rr_desertlands_hu.png",
    ),
    "BrokenMoon": MapConfig(
        display_name="Broken Moon",
        image_filename="mp_rr_divided_moon_mu1.png",
    ),
}

SUPPORTED_MAPS = tuple(MAP_CONFIGS.keys())


def ensure_report_dirs() -> None:
    """Create local report output folders used by scripts."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
