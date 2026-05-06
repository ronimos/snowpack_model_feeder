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
    dem_path: Path = Path("data/dem/251126_Professor-DSM.tiff")
    survey_dir: Path = Path("data/surveys")
    weather_csv: Path = Path("data/weather/weather_data.csv")
    boundary_kml: Path = Path("data/boundaries/Little_Proff.kml")
    start_zone_kml: Path = Path("data/boundaries/Litte_prof_start_zone.kml")
    output_dir: Path = Path("outputs")
    windninja_library_dir: Path = Path("windninja/library")

    # --- SNOWPACK paths ---
    snowpack_dir: Path = Path("snowpack/little_prof")
    release_geojson: Path = Path("data/boundaries/avalanche_release_area.geojson")

    # --- Scenario defaults ---
    default_snapshot: str = "2026-01-17"
    n_triggers: int = 5
    size_factors: list = field(default_factory=lambda: [0.70, 0.85, 1.00, 1.15, 1.30])
    depth_percentiles: list = field(default_factory=lambda: [10, 50, 90])
    depth_scales: dict = field(default_factory=lambda: {10: 0.75, 50: 1.00, 90: 1.30})
    mu: float = 0.155           # Voellmy dry friction (uncalibrated default)
    xi: float = 1500.0          # Voellmy turbulent friction m/s² (uncalibrated default)
    stauchwall_deg: float = 28.0

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
    target_cells_per_cluster: int = 50  # auto-determines initial cluster count
    max_cells_per_cluster: int = 20     # recursively split clusters larger than this
    min_cluster_size: int = 4       # minimum cluster size to avoid excessive splitting
    n_pca_components: float = 0.99  # Explained variance threshold for PCA dimensionality reduction before clustering
    n_clusters_override: Optional[int] = None  # set to force a specific initial count
    max_cluster_std_m: float = 0.08  # 8 cm

    def __post_init__(self):
        """Resolve paths relative to project_dir."""
        self.dem_path = self.project_dir / self.dem_path
        self.survey_dir = self.project_dir / self.survey_dir
        self.weather_csv = self.project_dir / self.weather_csv
        self.boundary_kml  = self.project_dir / self.boundary_kml
        self.start_zone_kml = self.project_dir / self.start_zone_kml
        self.output_dir = self.project_dir / self.output_dir
        self.windninja_library_dir = self.project_dir / self.windninja_library_dir
        self.snowpack_dir = self.project_dir / self.snowpack_dir
        self.release_geojson = self.project_dir / self.release_geojson

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"
    

    @property
    def smet_dir(self) -> Path:
        return self.output_dir / "smet"

    @property
    def grids_dir(self) -> Path:
        return self.output_dir / "hourly_grids"

    @property
    def resampled_dir(self) -> Path:
        return self.output_dir / "resampled_1m"

    @property
    def pro_dir(self) -> Path:
        return self.snowpack_dir / "output"

    @property
    def zarr_path(self) -> Path:
        return self.pro_dir / "slope_snowpack.zarr"

    @property
    def scenarios_dir(self) -> Path:
        return self.output_dir / "scenarios"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def dem_1m_path(self) -> Path:
        return self.resampled_dir / "dem_1m.tif"

    @property
    def cluster_map_path(self) -> Path:
        return self.analysis_dir / "cluster_map.npy"

    @property
    def cluster_map_tif_path(self) -> Path:
        return self.analysis_dir / "cluster_map.tif"

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.output_dir, self.analysis_dir, self.plots_dir,
                  self.smet_dir, self.grids_dir, self.resampled_dir,
                  self.scenarios_dir, self.models_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            