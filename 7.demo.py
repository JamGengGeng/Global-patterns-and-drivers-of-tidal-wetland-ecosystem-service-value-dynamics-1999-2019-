# -*- coding: utf-8 -*-
"""
1. Crop one GeoTIFF to the requested bounding box.
2. Convert valid raster cells to point centers and fill source values from a CSV
   using 5 km / 10 km / 50 km neighborhood means, then median fallback.
3. Apply S1 scale factors to rows whose ``fill_method`` is ``median``.
4. Build the final output table for the demo year series.

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.windows import from_bounds
from sklearn.neighbors import BallTree

# 0) Code Ocean input / output paths
INPUT_ROOT = Path("/data/demonstration")
OUTPUT_ROOT = Path("/results/demonstration")
FINAL_OUTPUT_ROOT = OUTPUT_ROOT / "final_outputs"

INPUT_TIF = Path("/data/demonstration/netgain_North_American_04.tif")
POINTS_CSV = INPUT_ROOT / "SOC_1m_North_American.csv"
SCALE_XLSX = INPUT_ROOT / "SOC-AE-S1_factors.xlsx"

OUTPUT_TIF = Path("/results/demonstration/netgain_North_American_04_bbox_29N30N_92W30W.tif")
FILLED_CSV = OUTPUT_ROOT / "netgain_North_American_04_bbox_29N30N_92W30W_SOC_1m_filled.csv"
SCALED_CSV = OUTPUT_ROOT / "netgain_North_American_04_bbox_29N30N_92W30W_SOC_1m_filled_scaled_S1.csv"
FINAL_CSV = (
    FINAL_OUTPUT_ROOT
    / "SOC_1m_North_American_bbox_29N30N_92W30W_final_total_STRICT_CLEARLOSS.csv"
)

# 1) Demo parameters
LAT_MIN = 29.0
LAT_MAX = 30.0
LON_MIN = -92.0
LON_MAX = -30.0

RADII_KM = [5, 10, 50]
QUERY_BATCH_SIZE = 20_000
EARTH_RADIUS_KM = 6371.0088
EPS = 1e-9

REGION = "North_American"
GENRE = "SOC_1m"
BBOX_TAG = "bbox_29N30N_92W30W"
YEARS = ["04"]

GRID_COL = "grid_code"
GAIN_COL = "vals"
VALUE_COL = "value_fill"
LAT_COL = "Latitude"
LON_COL = "Longitude"

W04 = [0.575, 0.126, 0.086, 0.067, 0.056, 0.049]
WOTH = [0.472, 0.174, 0.101, 0.075, 0.061]

OUT_FIELDS = [
    "pointid",
    "grid_code",
    "Latitude",
    "Longitude",
    "value_fill",
    "fill_method",
    "vals",
]

YEAR_FILES = {"04": SCALED_CSV}


# 2) Utilities
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def pick_field(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        match = lower_map.get(name.lower())
        if match is not None:
            return match
    return None


def to_1deg_center(arr: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(arr, dtype="float64")
    return np.floor(arr - EPS) + 0.5


# 3) Step 1: crop raster
def crop_single_tif(
    input_tif: Path,
    output_tif: Path,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> None:
    logging.info("Cropping raster: %s", input_tif)
    ensure_parent(output_tif)

    with rasterio.open(input_tif) as src:
        src_left, src_bottom, src_right, src_top = src.bounds

        inter_left = max(lon_min, src_left)
        inter_right = min(lon_max, src_right)
        inter_bottom = max(lat_min, src_bottom)
        inter_top = min(lat_max, src_top)

        if inter_left >= inter_right or inter_bottom >= inter_top:
            raise ValueError(
                "Requested bounding box does not overlap the input raster. "
                f"Raster bounds={src.bounds}, request=({lon_min}, {lat_min}, {lon_max}, {lat_max})"
            )

        window = from_bounds(
            inter_left,
            inter_bottom,
            inter_right,
            inter_top,
            transform=src.transform,
        ).round_offsets().round_lengths()

        data = src.read(window=window)
        out_transform = src.window_transform(window)

        meta = src.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
            }
        )

        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(data)

    logging.info("Cropped raster written to: %s", output_tif)


# 4) Step 2: fill raster points from source CSV
def read_value_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    lat_col = pick_field(df.columns.tolist(), ["Lat.", "Lat_", "Latitude", "latitude", "lat"])
    lon_col = pick_field(df.columns.tolist(), ["Lon.", "Lon_", "Longitude", "longitude", "lon"])
    val_col = pick_field(
        df.columns.tolist(),
        ["Value_USD/ha/year", "Value_USD_ha_year", "value", "value_usd_ha_year"],
    )

    missing = {
        "latitude": lat_col,
        "longitude": lon_col,
        "value": val_col,
    }
    missing_names = [name for name, col in missing.items() if col is None]
    if missing_names:
        raise KeyError(f"Missing required source CSV columns: {missing_names}")

    out = pd.DataFrame(
        {
            "Longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "Latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "value": pd.to_numeric(df[val_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["Longitude", "Latitude", "value"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No valid source points found in: {csv_path}")
    return out


def read_raster_points(raster_path: Path) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata

        if nodata is None:
            valid_mask = np.ones(arr.shape, dtype=bool)
        elif isinstance(nodata, float) and np.isnan(nodata):
            valid_mask = ~np.isnan(arr)
        else:
            valid_mask = arr != nodata

        rows, cols = np.where(valid_mask)
        xs, ys = xy(src.transform, rows, cols, offset="center")
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        vals = arr[rows, cols]

        return pd.DataFrame(
            {
                "pointid": np.arange(1, rows.size + 1, dtype=np.int64),
                "grid_code": vals.astype(np.float64),
                "Latitude": ys.astype(np.float64),
                "Longitude": xs.astype(np.float64),
                "value_fill": np.nan,
                "fill_method": pd.Series([None] * rows.size, dtype="object"),
            }
        )


def build_balltree(latitudes: np.ndarray, longitudes: np.ndarray) -> BallTree:
    coords_rad = np.deg2rad(np.column_stack([latitudes, longitudes]))
    return BallTree(coords_rad, metric="haversine")


def fill_by_radius_mean(
    target_df: pd.DataFrame,
    radius_km: float,
    tree: BallTree,
    source_values: np.ndarray,
) -> int:
    need_mask = target_df["value_fill"].isna().to_numpy()
    need_idx = np.where(need_mask)[0]
    radius_rad = radius_km / EARTH_RADIUS_KM

    for start in range(0, need_idx.size, QUERY_BATCH_SIZE):
        idx_batch = need_idx[start : start + QUERY_BATCH_SIZE]
        batch_coords = np.deg2rad(
            np.column_stack(
                [
                    target_df.loc[idx_batch, "Latitude"].to_numpy(),
                    target_df.loc[idx_batch, "Longitude"].to_numpy(),
                ]
            )
        )

        neighbors = tree.query_radius(batch_coords, r=radius_rad, return_distance=False)
        fill_values = np.full(idx_batch.shape[0], np.nan, dtype=np.float64)
        fill_flags = np.zeros(idx_batch.shape[0], dtype=bool)

        for i, nb in enumerate(neighbors):
            if nb.size > 0:
                fill_values[i] = float(np.mean(source_values[nb]))
                fill_flags[i] = True

        if np.any(fill_flags):
            chosen_idx = idx_batch[fill_flags]
            target_df.loc[chosen_idx, "value_fill"] = fill_values[fill_flags]
            target_df.loc[chosen_idx, "fill_method"] = f"{int(radius_km)}km"

    return int(target_df["value_fill"].isna().sum())


def fill_with_median(target_df: pd.DataFrame, median_value: float) -> int:
    need_mask = target_df["value_fill"].isna()
    filled_count = int(need_mask.sum())
    target_df.loc[need_mask, "value_fill"] = float(median_value)
    target_df.loc[need_mask, "fill_method"] = "median"
    return filled_count


def fill_raster_point_values(raster_path: Path, csv_path: Path, output_csv: Path) -> None:
    logging.info("Filling raster points from CSV: %s", csv_path)
    ensure_parent(output_csv)

    source_df = read_value_points(csv_path)
    median_value = float(np.median(source_df["value"].to_numpy()))
    target_df = read_raster_points(raster_path)

    tree = build_balltree(
        source_df["Latitude"].to_numpy(),
        source_df["Longitude"].to_numpy(),
    )
    source_values = source_df["value"].to_numpy(dtype=np.float64)

    for rkm in RADII_KM:
        remaining = fill_by_radius_mean(
            target_df=target_df,
            radius_km=rkm,
            tree=tree,
            source_values=source_values,
        )
        logging.info("Remaining nulls after %skm = %s", rkm, remaining)

    median_filled = fill_with_median(target_df, median_value)
    logging.info("Rows filled by median = %s", median_filled)

    target_df["vals"] = target_df["value_fill"].astype(float) * target_df["grid_code"].astype(float)
    target_df = target_df[OUT_FIELDS].copy()
    target_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    logging.info("Filled CSV written to: %s", output_csv)
    logging.info("Fill-method counts:\n%s", target_df["fill_method"].value_counts(dropna=False))


# 5) Step 3: apply S1 scaling
def apply_s1_scaling(input_csv: Path, scale_xlsx: Path, output_csv: Path) -> None:
    logging.info("Applying S1 scaling: %s", scale_xlsx)
    ensure_parent(output_csv)

    scale_df = pd.read_excel(scale_xlsx)
    scale_df = scale_df.copy()
    scale_df["lat_c"] = pd.to_numeric(scale_df["latitude"], errors="coerce")
    scale_df["lon_c"] = pd.to_numeric(scale_df["longitude"], errors="coerce")
    scale_df["S1"] = pd.to_numeric(scale_df["S1"], errors="coerce")
    scale_df = scale_df.dropna(subset=["lat_c", "lon_c", "S1"]).drop_duplicates(
        subset=["lat_c", "lon_c"]
    )

    s1_map = {(r.lat_c, r.lon_c): r.S1 for r in scale_df.itertuples(index=False)}

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    lat = pd.to_numeric(df["Latitude"], errors="coerce").to_numpy(dtype="float64")
    lon = pd.to_numeric(df["Longitude"], errors="coerce").to_numpy(dtype="float64")
    grid_code = pd.to_numeric(df["grid_code"], errors="coerce").to_numpy(dtype="float64")
    value_fill = pd.to_numeric(df["value_fill"], errors="coerce").to_numpy(dtype="float64")

    lat_c = to_1deg_center(lat)
    lon_c = to_1deg_center(lon)

    s1 = np.array([s1_map.get((a, b), np.nan) for a, b in zip(lat_c, lon_c)], dtype="float64")
    is_median = df["fill_method"].astype(str).str.lower().eq("median").to_numpy()
    match_s1 = ~np.isnan(s1)
    matched = match_s1 & is_median

    df["lat_c"] = lat_c
    df["lon_c"] = lon_c
    df["S1"] = s1
    df["match_S1"] = match_s1

    df["vals"] = value_fill * grid_code
    df.loc[matched, "vals"] = value_fill[matched] * grid_code[matched] * s1[matched]

    df["value_fill_scaled"] = value_fill
    df.loc[matched, "value_fill_scaled"] = value_fill[matched] * s1[matched]

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logging.info("Scaled CSV written to: %s", output_csv)
    logging.info("Matched median rows with S1 = %s", int(matched.sum()))


# 6) Step 4: final table
def add_latlon_keys_strict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df["lat_str"] = df[LAT_COL].astype(str)
    df["lon_str"] = df[LON_COL].astype(str)
    return df


def build_year_df(csv_path: Path, year: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[[LAT_COL, LON_COL, GRID_COL, GAIN_COL, VALUE_COL]].copy()
    df = add_latlon_keys_strict(df)

    df[f"grid_{year}"] = pd.to_numeric(df[GRID_COL], errors="coerce").fillna(0).astype(int)
    df[f"gain_{year}"] = pd.to_numeric(df[GAIN_COL], errors="coerce").fillna(0.0)
    df[f"val_{year}"] = pd.to_numeric(df[VALUE_COL], errors="coerce").fillna(0.0)

    df = df[
        [
            "lat_str",
            "lon_str",
            LAT_COL,
            LON_COL,
            f"grid_{year}",
            f"gain_{year}",
            f"val_{year}",
        ]
    ]

    return df.groupby(["lat_str", "lon_str"], as_index=False).agg(
        {
            LAT_COL: "first",
            LON_COL: "first",
            f"grid_{year}": "max",
            f"gain_{year}": "mean",
            f"val_{year}": "mean",
        }
    )


def build_wide_table() -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None

    for year in YEARS:
        dfy = build_year_df(YEAR_FILES[year], year)
        if merged is None:
            merged = dfy.copy()
        else:
            dfy2 = dfy.drop(columns=[LAT_COL, LON_COL], errors="ignore")
            merged = merged.merge(dfy2, on=["lat_str", "lon_str"], how="outer")

    if merged is None:
        raise ValueError("No yearly files were configured.")

    for year in YEARS:
        merged[f"grid_{year}"] = pd.to_numeric(merged[f"grid_{year}"], errors="coerce").fillna(0).astype(int)
        merged[f"gain_{year}"] = pd.to_numeric(merged[f"gain_{year}"], errors="coerce").fillna(0.0)
        merged[f"val_{year}"] = pd.to_numeric(merged[f"val_{year}"], errors="coerce").fillna(0.0)

    return merged


def compute_delta_and_cumulative(row: pd.Series) -> dict[str, float]:
    delta = {year: 0.0 for year in YEARS}
    cumulative = {year: 0.0 for year in YEARS}
    running = 0.0

    chain_active = False
    gain_base = 0.0
    w_list: Optional[list[float]] = None
    w_idx = 0
    applied_weight_sum = 0.0

    def start_chain(start_year: str, base_gain: float) -> None:
        nonlocal chain_active, gain_base, w_list, w_idx, applied_weight_sum
        chain_active = True
        gain_base = base_gain
        w_list = W04 if start_year == "04" else WOTH
        w_idx = 0
        applied_weight_sum = 0.0

    def end_chain() -> None:
        nonlocal chain_active, gain_base, w_list, w_idx, applied_weight_sum
        chain_active = False
        gain_base = 0.0
        w_list = None
        w_idx = 0
        applied_weight_sum = 0.0

    for year in YEARS:
        g = int(row[f"grid_{year}"])
        gain_y = float(row[f"gain_{year}"])
        val_y = float(row[f"val_{year}"])

        if g == -1:
            if chain_active:
                d = -gain_base * applied_weight_sum
                end_chain()
            else:
                d = -1.0 * val_y
            running += d
        elif g == 1:
            if chain_active:
                end_chain()

            start_chain(year, gain_y)

            if w_list is not None and w_idx < len(w_list):
                w = w_list[w_idx]
                applied_weight_sum += w
                w_idx += 1
                d = gain_base * w
                running += d
            else:
                d = 0.0
        elif g == 0 and chain_active:
            if w_list is not None and w_idx < len(w_list):
                w = w_list[w_idx]
                applied_weight_sum += w
                w_idx += 1
                d = gain_base * w
                running += d
            else:
                d = 0.0
        else:
            d = 0.0

        delta[year] = d
        cumulative[year] = running

    out: dict[str, float] = {"final_total": running}
    for year in YEARS:
        out[f"delta_{year}"] = delta[year]
        out[f"cumulative_{year}"] = cumulative[year]

    return out


def build_final_output(output_csv: Path) -> None:
    logging.info("Building final output table")
    ensure_parent(output_csv)

    wide = build_wide_table()
    calc_df = wide.apply(lambda row: pd.Series(compute_delta_and_cumulative(row)), axis=1)
    out = pd.concat([wide, calc_df], axis=1)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    logging.info("Final CSV written to: %s", output_csv)
    logging.info("Final rows = %s", len(out))



def main() -> None:
    setup_logging()

    logging.info("Checking required input files")
    for path in [INPUT_TIF, POINTS_CSV, SCALE_XLSX]:
        require_file(path)

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    crop_single_tif(
        input_tif=INPUT_TIF,
        output_tif=OUTPUT_TIF,
        lon_min=LON_MIN,
        lat_min=LAT_MIN,
        lon_max=LON_MAX,
        lat_max=LAT_MAX,
    )
    fill_raster_point_values(
        raster_path=OUTPUT_TIF,
        csv_path=POINTS_CSV,
        output_csv=FILLED_CSV,
    )
    apply_s1_scaling(
        input_csv=FILLED_CSV,
        scale_xlsx=SCALE_XLSX,
        output_csv=SCALED_CSV,
    )
    build_final_output(FINAL_CSV)

    logging.info("Pipeline finished successfully")
    logging.info("Cropped TIFF: %s", OUTPUT_TIF)
    logging.info("Filled CSV: %s", FILLED_CSV)
    logging.info("Scaled CSV: %s", SCALED_CSV)
    logging.info("Final CSV: %s", FINAL_CSV)

    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, "Python 已运行结束", "提示", 0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
