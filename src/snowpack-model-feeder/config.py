"""
Project configuration for distributed SNOWPACK forcing generation.

Edit paths and station metadata to match your setup.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProjectConfig:
    # --- Paths ---
    project_dir: Path = Path(".")
    dem_path: Path = Path("data/dem/251126_Professor_PTC_DSM.tiff")
    survey_dir: Path = Path("data/surveys")
    weather_csv: Path = Path("data/weather/weather_data.csv")
    boundary_kml: Path = Path("data/boundaries/Little_Proff.kml")
    output_dir: Path = Path("outputs")

    # --- Survey file pattern ---
    # Expected format: YYMMDD_*_snowHeight.tif
    # The date prefix is extracted automatically
    survey_glob: str = "*_snowHeight.tif"
    bare_ground_date: str = "251126"  # YYMMDD of the bare ground DEM

    # --- Station metadata ---
    summit_id: str = "CAABT"
    summit_lat: float = 39.6424
    summit_lon: float = -105.8718
    summit_alt_m: float = 3798.3

    base_id: str = "CAABM"
    base_lat: float = 39.6424
    base_lon: float = -105.8718
    base_alt_m: float = 3554.0

    # --- Processing parameters ---
    target_resolution_m: float = 1.0
    flight_hour_utc: int = 18
    sx_search_distance_m: float = 300.0
    sx_azimuths_deg: list = field(default_factory=lambda: [
        0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
        180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5
    ])
    transport_smoothing_window_m: int = 15
    min_valid_hs_m: float = 0.0
    max_valid_hs_m: float = 12.0

    # --- Domain mask ---
    min_slope_deg: float = 15.0

    # --- Clustering ---
    target_cells_per_cluster: int = 500
    n_clusters_override: Optional[int] = None  # set to force a specific count

    def __post_init__(self):
        """Resolve paths relative to project_dir."""
        self.dem_path = self.project_dir / self.dem_path
        self.survey_dir = self.project_dir / self.survey_dir
        self.weather_csv = self.project_dir / self.weather_csv
        self.boundary_kml = self.project_dir / self.boundary_kml
        self.output_dir = self.project_dir / self.output_dir

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def smet_dir(self) -> Path:
        return self.output_dir / "smet"

    @property
    def grids_dir(self) -> Path:
        return self.output_dir / "hourly_grids"

    @property
    def resampled_dir(self) -> Path:
        return self.output_dir / "resampled_1m"

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.output_dir, self.analysis_dir, self.smet_dir,
                  self.grids_dir, self.resampled_dir]:
            d.mkdir(parents=True, exist_ok=True)
