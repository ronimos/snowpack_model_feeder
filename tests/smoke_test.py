#!/usr/bin/env python3
"""
smoke_test.py — Post-pipeline sanity checks.

Run after any pipeline change to catch the classes of bugs that
keep recurring: unit mismatches, file naming, data format issues,
temporal gaps, and CRS problems.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --snapshot-date 2026-01-17
    python scripts/smoke_test.py --verbose

Exit code 0 = all checks passed, 1 = failures detected.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "snowpack-model-feeder"))


class SmokeTest:
    def __init__(self, project_dir='.', snapshot_date='2026-01-17',
                 verbose=False):
        self.project_dir = Path(project_dir)
        self.snapshot = snapshot_date
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []

    def check(self, name, condition, detail=''):
        if condition:
            self.passed += 1
            if self.verbose:
                print(f"  ✓ {name}")
        else:
            self.failed += 1
            msg = f"  ✗ {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)
            self.errors.append(name)

    def warn(self, name, detail=''):
        self.warnings += 1
        msg = f"  ⚠ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    def file_exists(self, path, name=None):
        p = self.project_dir / path
        label = name or str(path)
        self.check(f"{label} exists", p.exists(), f"missing: {p}")
        return p.exists()

    # ==================================================================
    # Test groups
    # ==================================================================

    def test_resampled_outputs(self):
        print("\n--- Resampled outputs ---")
        res_dir = self.project_dir / "outputs" / "resampled_1m"

        if not self.file_exists("outputs/resampled_1m/dem_1m.tif", "DEM 1m"):
            return

        import rasterio
        with rasterio.open(str(res_dir / "dem_1m.tif")) as src:
            dem = src.read(1).astype(float)
            transform = src.transform
            crs = src.crs

            # Pixel size should be ~1m
            self.check("DEM pixel size ~1m",
                       0.5 < abs(transform.a) < 2.0,
                       f"pixel={transform.a:.4f}")

            # CRS should exist
            if crs is not None:
                self.check("DEM has CRS", True)
            else:
                self.warn("DEM has no CRS",
                          "will need EPSG:6342 fallback everywhere")

            # DEM values should be reasonable elevation (3000-4500m for A-Basin)
            valid = dem[~np.isnan(dem)]
            if len(valid):
                self.check("DEM elevation range sane",
                           2500 < np.median(valid) < 5000,
                           f"median={np.median(valid):.0f}m")

        # HS grids exist
        hs_files = sorted(res_dir.glob("hs_*.npy"))
        self.check("HS grids exist", len(hs_files) > 0,
                   f"found {len(hs_files)}")
        if hs_files:
            # Spot-check one HS grid
            hs = np.load(str(hs_files[-1]))
            self.check("HS grid shape matches DEM",
                       hs.shape == dem.shape,
                       f"HS={hs.shape} DEM={dem.shape}")
            valid_hs = hs[~np.isnan(hs)]
            if len(valid_hs):
                self.check("HS values in metres (0-15m range)",
                           0 <= np.percentile(valid_hs, 95) < 15,
                           f"P95={np.percentile(valid_hs, 95):.2f}")

    def test_cluster_map(self):
        print("\n--- Cluster map ---")
        cm_path = self.project_dir / "outputs/analysis/cluster_map.npy"
        if not self.file_exists(cm_path, "cluster_map.npy"):
            return

        cm = np.load(str(cm_path))
        n_clusters = len(np.unique(cm[cm > 0]))
        self.check("Cluster count > 100",
                   n_clusters > 100,
                   f"n={n_clusters}")
        self.check("Cluster count < 20000",
                   n_clusters < 20000,
                   f"n={n_clusters}")

        # Cluster sizes
        cids = np.unique(cm[cm > 0])
        sizes = [int(np.sum(cm == c)) for c in cids[:100]]  # sample
        median_size = np.median(sizes)
        self.check("Median cluster size 3-50 pixels",
                   3 <= median_size <= 50,
                   f"median={median_size:.0f}")

    def test_features(self):
        print("\n--- Feature CSVs ---")
        analysis_dir = self.project_dir / "outputs/analysis"

        # Try full start zone first, then group-level
        feat_csv = analysis_dir / f"all_start_zone_features_{self.snapshot}.csv"
        if not feat_csv.exists():
            feat_csv = analysis_dir / f"release_zone_features_{self.snapshot}.csv"
        if not self.file_exists(feat_csv, "Features CSV"):
            return

        df = pd.read_csv(str(feat_csv), index_col=0)
        n = len(df)
        self.check(f"Features CSV has clusters (n={n})", n > 50)

        # slab_thickness units: should be in metres (0.1 - 5.0m typical)
        if 'slab_thickness' in df.columns:
            st = df['slab_thickness'].dropna()
            if len(st):
                med = float(st.median())
                self.check("slab_thickness in metres (0.05-5.0m)",
                           0.05 < med < 5.0,
                           f"median={med:.3f}")
                # Check for cm values (median > 10 suggests cm not m)
                if med > 10:
                    self.warn("slab_thickness looks like cm, not m",
                              f"median={med:.1f}")
        else:
            self.warn("slab_thickness column missing")

        # Sk38 should be positive, mostly 0-6 range
        if 'min_sk38' in df.columns:
            sk = df['min_sk38'].dropna()
            if len(sk):
                self.check("Sk38 values in 0-10 range",
                           0 <= sk.median() < 10,
                           f"median={sk.median():.3f}")
                # Check for all-NaN (the ±0.05cm bug)
                self.check("Sk38 not all-NaN",
                           len(sk) > n * 0.5,
                           f"{len(sk)}/{n} non-NaN")

        # Meloche features
        mel_csv = analysis_dir / f"meloche_features_all_{self.snapshot}.csv"
        if not mel_csv.exists():
            mel_csv = analysis_dir / f"meloche_features_{self.snapshot}.csv"
        if mel_csv.exists():
            mel = pd.read_csv(str(mel_csv), index_col=0)
            if 'tau_g' in mel.columns:
                tg = mel['tau_g'].dropna()
                if len(tg):
                    self.check("tau_g in Pa (50-2000 typical)",
                               10 < tg.median() < 5000,
                               f"median={tg.median():.1f} Pa")
            if 'Lambda' in mel.columns:
                lam = mel['Lambda'].dropna()
                if len(lam):
                    self.check("Lambda in metres (0.5-10m typical)",
                               0.1 < lam.median() < 20,
                               f"median={lam.median():.2f}m")

    def test_smet_files(self):
        print("\n--- SMET files ---")
        smet_dir = self.project_dir / "outputs/smet"
        smets = sorted(smet_dir.glob("cluster_*.smet"))
        self.check("SMET files exist", len(smets) > 0,
                   f"found {len(smets)}")
        if not smets:
            return

        # Check one SMET for temporal gaps
        smet_file = smets[len(smets) // 2]  # middle file
        timestamps = []
        in_data = False
        with open(str(smet_file)) as f:
            for line in f:
                if line.strip() == '[DATA]':
                    in_data = True
                    continue
                if in_data and line.strip():
                    ts = line.split()[0]
                    try:
                        timestamps.append(pd.Timestamp(ts))
                    except Exception:
                        pass

        if len(timestamps) > 2:
            diffs = pd.Series(timestamps).diff().dropna()
            max_gap_hours = diffs.max().total_seconds() / 3600

            self.check("SMET max gap < 6 hours",
                       max_gap_hours <= 6,
                       f"max gap={max_gap_hours:.1f}h in {smet_file.name}")
            if max_gap_hours > 6:
                # Find where the gap is
                gap_idx = diffs.argmax()
                self.warn(f"Gap at {timestamps[gap_idx]}",
                          f"{max_gap_hours:.0f}h — refill may not have run")

    def test_snowpack_output(self):
        print("\n--- SNOWPACK output ---")
        from config import ProjectConfig
        cfg = ProjectConfig(project_dir=self.project_dir)

        # .pro files
        pro_files = sorted(cfg.pro_dir.glob("*.pro"))
        self.check("SNOWPACK .pro files exist", len(pro_files) > 0,
                   f"found {len(pro_files)}")

        # Check one log for completion
        log_files = sorted(cfg.pro_dir.glob("*.log"))
        if log_files:
            n_ok = sum(1 for l in log_files
                       if 'done!' in l.read_text(errors='ignore'))
            n_fail = len(log_files) - n_ok
            self.check("SNOWPACK runs completed",
                       n_fail == 0,
                       f"OK={n_ok} Failed={n_fail}")
            if n_fail > 0:
                failed = [l.stem for l in log_files
                          if 'done!' not in l.read_text(errors='ignore')]
                self.warn(f"Failed clusters: {failed[:5]}...")

        # Zarr exists and has expected dimensions
        zarr_path = cfg.zarr_path
        if zarr_path.exists():
            import xarray as xr
            ds = xr.open_zarr(str(zarr_path))
            n_locs = len(ds.coords['location'])
            n_times = len(ds.coords['time'])

            self.check("Zarr location count > 100",
                       n_locs > 100,
                       f"n_locations={n_locs}")
            self.check("Zarr time steps > 100",
                       n_times > 100,
                       f"n_times={n_times}")
            self.check("hand_hardness in Zarr",
                       'hand_hardness' in ds.data_vars)
            self.check("HS in Zarr",
                       'HS' in ds.data_vars)
            ds.close()
        else:
            self.warn("Zarr store not found", str(zarr_path))

    def test_scenarios(self):
        print("\n--- Scenarios ---")
        scen_dir = (self.project_dir / "outputs/scenarios" / self.snapshot)
        if not scen_dir.exists():
            self.warn(f"Scenario dir not found: {scen_dir}")
            return

        # metadata.json
        meta_path = scen_dir / "metadata.json"
        if self.file_exists(meta_path, "metadata.json"):
            meta = json.loads(meta_path.read_text())
            self.check("metadata has n_scenarios",
                       'n_scenarios' in meta,
                       str(meta.get('n_scenarios', 'missing')))

        # scenario_weights.json
        weights_path = scen_dir / "scenario_weights.json"
        if self.file_exists(weights_path, "scenario_weights.json"):
            weights = json.loads(weights_path.read_text())
            total = sum(weights.values())
            self.check("Weights sum to ~1.0",
                       abs(total - 1.0) < 0.01,
                       f"sum={total:.6f}")

        # Check one scenario directory
        scen_dirs = sorted((scen_dir / "scenarios").glob("scenario_*"))
        self.check("Scenario directories exist",
                   len(scen_dirs) > 0,
                   f"found {len(scen_dirs)}")
        if not scen_dirs:
            return

        s1 = scen_dirs[0]

        # params.json
        params_path = s1 / "params.json"
        if self.file_exists(params_path, f"{s1.name}/params.json"):
            p = json.loads(params_path.read_text())

            self.check("params has scenario_probability",
                       'scenario_probability' in p)
            self.check("params has trigger_cluster",
                       'trigger_cluster' in p)

            # Release area sane
            area = p.get('release_area_m2', 0)
            self.check("Release area > 100 m²",
                       area > 100,
                       f"area={area}")
            self.check("Release area < 50000 m²",
                       area < 50000,
                       f"area={area}")

            # Mean depth sane (should be in metres)
            depth = p.get('mean_depth_m', 0)
            self.check("Mean depth 0.1-5.0m",
                       0.1 < depth < 5.0,
                       f"depth={depth:.3f}m")

            # Volume sanity: area × depth ≈ volume
            vol = p.get('total_volume_m3', 0)
            expected_vol = area * depth
            if expected_vol > 0:
                ratio = vol / expected_vol
                self.check("Volume ≈ area × depth (within 2×)",
                           0.3 < ratio < 3.0,
                           f"vol={vol:.0f} expected={expected_vol:.0f} "
                           f"ratio={ratio:.2f}")

        # release.geojson CRS
        rel_path = s1 / "release.geojson"
        if rel_path.exists():
            gj = json.loads(rel_path.read_text())
            crs_name = (gj.get('crs', {})
                          .get('properties', {})
                          .get('name', ''))
            self.check("release.geojson has EPSG:6342",
                       '6342' in crs_name,
                       f"crs={crs_name or 'missing'}")

        # depth.prj exists alongside depth.asc
        if (s1 / "depth.asc").exists():
            self.check("depth.prj sidecar exists",
                       (s1 / "depth.prj").exists())

    def test_sno_naming(self):
        print("\n--- SNO file naming ---")
        from config import ProjectConfig
        cfg = ProjectConfig(project_dir=self.project_dir)

        sno_files = sorted(cfg.pro_dir.glob("*.sno"))
        if not sno_files:
            self.warn("No .sno files found")
            return

        # Check naming convention: cluster_XXXX_cluster_XXXX.sno
        good = 0
        bad = 0
        for f in sno_files[:20]:  # sample
            name = f.stem
            parts = name.split('_')
            # Expected: cluster_XXXX_cluster_XXXX
            if (len(parts) == 4 and parts[0] == 'cluster'
                    and parts[2] == 'cluster' and parts[1] == parts[3]):
                good += 1
            else:
                bad += 1
                if bad <= 3:
                    self.warn(f"Unexpected .sno name: {f.name}")

        self.check("SNO naming convention consistent",
                   bad == 0,
                   f"{good} good, {bad} bad in sample")

    def test_groups(self):
        print("\n--- Cluster groups ---")
        groups_path = (self.project_dir / "outputs/analysis" /
                       "release_zone_groups.json")
        if not self.file_exists(groups_path, "release_zone_groups.json"):
            return

        with open(str(groups_path)) as f:
            groups = json.load(f)

        for grp in ['release', 'adjacent', 'reference']:
            n = len(groups.get(grp, []))
            self.check(f"Group '{grp}' non-empty",
                       n > 0,
                       f"n={n}")

        # No overlap between groups
        release = set(groups.get('release', []))
        adjacent = set(groups.get('adjacent', []))
        reference = set(groups.get('reference', []))
        overlap_ra = release & adjacent
        overlap_rr = release & reference
        self.check("No release/adjacent overlap",
                   len(overlap_ra) == 0,
                   f"{len(overlap_ra)} shared")
        self.check("No release/reference overlap",
                   len(overlap_rr) == 0,
                   f"{len(overlap_rr)} shared")

    # ==================================================================
    # Runner
    # ==================================================================

    def run_all(self):
        print("=" * 60)
        print(f"  Smoke Test — {self.project_dir}")
        print(f"  Snapshot: {self.snapshot}")
        print("=" * 60)

        self.test_resampled_outputs()
        self.test_cluster_map()
        self.test_features()
        self.test_smet_files()
        self.test_snowpack_output()
        self.test_scenarios()
        self.test_sno_naming()
        self.test_groups()

        print("\n" + "=" * 60)
        print(f"  Results: {self.passed} passed, "
              f"{self.failed} failed, "
              f"{self.warnings} warnings")
        if self.errors:
            print(f"\n  Failed checks:")
            for e in self.errors:
                print(f"    ✗ {e}")
        print("=" * 60)

        return self.failed == 0


def main():
    ap = argparse.ArgumentParser(description="Pipeline smoke test")
    ap.add_argument('--project-dir', default='.')
    ap.add_argument('--snapshot-date', default='2026-01-17')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    st = SmokeTest(
        project_dir=args.project_dir,
        snapshot_date=args.snapshot_date,
        verbose=args.verbose,
    )
    ok = st.run_all()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

