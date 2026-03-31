"""
Merge S1/S2 scaling factors into batch CSV outputs.
Apply scaling only to rows whose ``fill_method`` is ``median``.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

def parse_args():
    """Parse input and output paths from CLI or environment variables."""
    parser = argparse.ArgumentParser(
        description="Merge S1/S2 scaling tables into batch CSV outputs."
    )
    parser.add_argument(
        "--input-root",
        default=os.getenv("INPUT_ROOT", "./data/batch_outputs_5_10_50km_median_only_with_method"),
        help="Root directory containing genre subfolders with CSV files.",
    )
    parser.add_argument(
        "--s1-xlsx",
        default=os.getenv("S1_XLSX", "./data/scales/SOC-AE-S1_scale.xlsx"),
        help="Path to the S1 scaling-factor Excel file.",
    )
    parser.add_argument(
        "--s2-xlsx",
        default=os.getenv("S2_XLSX", "./data/scales/EECR-WP-EC-S2_scale.xlsx"),
        help="Path to the S2 scaling-factor Excel file.",
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("OUTPUT_ROOT", "./outputs/batch_outputs_5_10_50km_median_only_with_method_scaled_S1S2"),
        help="Output root directory. The script preserves the input relative layout.",
    )
    return parser.parse_args()


ARGS = parse_args()

INPUT_ROOT = ARGS.input_root
S1_XLSX = ARGS.s1_xlsx
S2_XLSX = ARGS.s2_xlsx
OUTPUT_ROOT = ARGS.output_root
os.makedirs(OUTPUT_ROOT, exist_ok=True)


GENRE_TO_SCALE = {
    # ---- Use S1 scaling table ----
    "alk": ("S1", S1_XLSX),
    "SOC": ("S1", S1_XLSX),
    "SOC_1m": ("S1", S1_XLSX),

    # ---- Use S2 scaling table (EECR/WP/EC/FRM/FM) ----
    "EECR": ("S2", S2_XLSX),
    "WP": ("S2", S2_XLSX),
    "EC": ("S2", S2_XLSX),
    "FRM": ("S2", S2_XLSX),
    "FM": ("S2", S2_XLSX),
}

GENRES = list(GENRE_TO_SCALE.keys())


NEED_BASE_COLS = {"latitude", "longitude"}


def load_scale_map(xlsx_path: str, preferred_cols):
    """Load one Excel scaling table and build a lookup dict."""
    df = pd.read_excel(xlsx_path)

    missing = NEED_BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"[{xlsx_path}] Missing required columns: {missing}. "
            "The file must contain latitude and longitude."
        )

    val_col = None
    for c in preferred_cols:
        if c in df.columns:
            val_col = c
            break

    if val_col is None:
        raise ValueError(
            f"[{xlsx_path}] Cannot find a scaling-factor column. "
            f"Tried: {preferred_cols}."
        )

    df = df.copy()
    df["lat_c"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon_c"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["scale_val"] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["lat_c", "lon_c", "scale_val"]).drop_duplicates(
        subset=["lat_c", "lon_c"]
    )

    scale_map = {(r.lat_c, r.lon_c): r.scale_val for r in df.itertuples(index=False)}
    return scale_map, val_col, len(df)

SCALE_CACHE = {}  # xlsx_path -> (scale_map, val_col, nrows)

def get_scale_map(scale_name: str, xlsx_path: str):
    """Return a cached scaling map, loading it on first use."""
    if xlsx_path in SCALE_CACHE:
        return SCALE_CACHE[xlsx_path]

    if scale_name == "S1":
        preferred = ["S1", "scale", "factor", "Scale", "Factor"]
    elif scale_name == "S2":
        preferred = ["S2", "scale", "factor", "Scale", "Factor"]
    else:
        preferred = ["scale", "factor", "Scale", "Factor"]

    scale_map, val_col, nrows = load_scale_map(xlsx_path, preferred)
    SCALE_CACHE[xlsx_path] = (scale_map, val_col, nrows)

    print(f"[LOAD] {scale_name} | {xlsx_path} | value_col={val_col} | rows={nrows}")
    return SCALE_CACHE[xlsx_path]


EPS = 1e-9


def to_1deg_center(arr):
    """Convert coordinates to 1-degree grid-cell centers ending in .5."""
    arr = np.asarray(arr, dtype="float64")
    return np.floor(arr - EPS) + 0.5


total_files = 0
total_rows = 0

median_rows = 0
matched_median_rows = 0
unmatched_median_rows = 0

for genre in GENRES:
    in_dir = os.path.join(INPUT_ROOT, genre)
    if not os.path.isdir(in_dir):
        print(f"[SKIP] Missing directory: {in_dir}")
        continue

    scale_name, scale_xlsx = GENRE_TO_SCALE[genre]
    scale_map, scale_val_col, _ = get_scale_map(scale_name, scale_xlsx)

    csv_list = glob.glob(os.path.join(in_dir, "**", "*.csv"), recursive=True)
    print(f"\n=== GENRE: {genre} | scale={scale_name} | files: {len(csv_list)} ===")

    for csv_path in csv_list:
        total_files += 1

        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="utf-8")

        req = ["Latitude", "Longitude", "grid_code", "value_fill", "fill_method"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            print(f"[WARN] Missing columns {miss} -> skipped: {csv_path}")
            continue

        # Convert relevant columns to numeric arrays.
        lat = pd.to_numeric(df["Latitude"], errors="coerce").to_numpy(dtype="float64")
        lon = pd.to_numeric(df["Longitude"], errors="coerce").to_numpy(dtype="float64")
        grid_code = pd.to_numeric(df["grid_code"], errors="coerce").to_numpy(dtype="float64")
        value_fill = pd.to_numeric(df["value_fill"], errors="coerce").to_numpy(dtype="float64")

        total_rows += len(df)

        lat_center = to_1deg_center(lat)
        lon_center = to_1deg_center(lon)

        scale_val = np.array(
            [scale_map.get((a, b), np.nan) for a, b in zip(lat_center, lon_center)],
            dtype="float64",
        )
        match_scale = ~np.isnan(scale_val)

        is_median = df["fill_method"].astype(str).str.lower().eq("median").to_numpy()

        matched = match_scale & is_median
        unmatched = (~match_scale) & is_median


        median_rows_file = int(is_median.sum())
        median_rows += median_rows_file
        matched_median_rows += int(matched.sum())
        unmatched_median_rows += int(unmatched.sum())


        if "vals" in df.columns:
            base_vals = pd.to_numeric(df["vals"], errors="coerce").to_numpy(dtype="float64")
            fallback = value_fill * grid_code
            nan_mask = np.isnan(base_vals)
            if nan_mask.any():
                base_vals[nan_mask] = fallback[nan_mask]
        else:
            base_vals = value_fill * grid_code

        val_S = base_vals.copy()
        val_S[matched] = base_vals[matched] * scale_val[matched]

        # Write result columns back to the DataFrame.
        df["lat_center"] = lat_center
        df["lon_center"] = lon_center
        df["scale_name"] = scale_name
        df["scale_value"] = scale_val
        df["match_scale"] = match_scale
        df["val_S"] = val_S

        # Preserve the original relative directory structure.
        rel = os.path.relpath(csv_path, INPUT_ROOT)
        out_path = os.path.join(OUTPUT_ROOT, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(
            f"matched_median={int(matched.sum())} | unmatched_median={int(unmatched.sum())}"
        )