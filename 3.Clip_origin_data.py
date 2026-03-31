"""
Split Excel point records into region-specific CSV files.

1. Scan all Excel files matching the configured filename pattern.
2. Match each point against predefined region polygons.
   - Multiple matches are allowed.
   - Boundary points are included by using `covers`.
"""

import os
import glob
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

INPUT_DIR = os.getenv("INPUT_DIR", "./data/values")
FILE_PATTERN = os.getenv("FILE_PATTERN", "*_new_20260206.xlsx")
SHEET_NAME = int(os.getenv("SHEET_NAME", "0"))

REQUIRED_COLS = ["latitude", "longitude", "value_usd_ha_year"]

OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", INPUT_DIR)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Suffix used to derive genre from filenames such as: EECR_new_20260206.xlsx -> EECR_South_America
FILE_SUFFIX = os.getenv("FILE_SUFFIX", "_new_20260206.xlsx")


REGIONS = [
    {
        "name": "South_America",
        "geometry": {"type": "Polygon", "coordinates": [[
            [-31.570, 14.100], [-72.140, 14.100], [-82.10, 0.000],
            [-180.000, 0.000], [-180.000, -66.3400], [-31.570, -66.3400],
            [-31.570, 14.100]
        ]]}
    },
    {
        "name": "North_American",
        "geometry": {"type": "Polygon", "coordinates": [[
            [-180.000, 66.340], [-180.000, 0.000], [-82.10, 0.000],
            [-72.140, 14.100], [-31.570, 14.100], [-31.570, 66.340],
            [-180.000, 66.340]
        ]]}
    },
    {
        "name": "Europe",
        "geometry": {"type": "Polygon", "coordinates": [[
            [-31.570, 66.340], [-31.570, 37.524], [11.178, 37.524],
            [32.196, 30.910], [39.869, 48.048], [29.236, 66.340], [-31.570, 66.340]
        ]]}
    },
    {
        "name": "Africa",
        "geometry": {"type": "Polygon", "coordinates": [[
            [-31.570, -66.3400], [52.640, -66.3400], [52.640, 14.092],
            [43.594, 12.591], [32.196, 30.910], [11.178, 37.524],
            [-31.570, 37.524], [-31.570, -66.3400]
        ]]}
    },
    {
        "name": "South_East_Asia",
        "geometry": {"type": "Polygon", "coordinates": [[
            [88.837, 17.176], [97.568, 28.548], [108.100, 21.500],
            [141.019, 21.500], [141.019, -10.500], [88.387, -10.500],
            [88.837, 17.176]
        ]]}
    },
    {
        "name": "Australia",
        "geometry": {"type": "Polygon", "coordinates": [[
            [52.640, -10.500], [52.640, -66.3400], [180.000, -66.3400],
            [180.000, 0.000], [141.019, 0.000], [141.019, -10.500],
            [52.640, -10.500]
        ]]}
    },
    {
        "name": "Asia_noEAS",
        "geometry": {"type": "Polygon", "coordinates": [[
            [29.236, 66.340], [39.869, 48.048], [32.196, 30.910],
            [43.594, 12.591], [52.640, 14.092], [52.640, -10.500],
            [88.387, -10.500], [88.837, 17.176], [97.568, 28.548],
            [108.100, 21.500], [141.019, 21.500], [141.019, 0.000],
            [180.000, 0.000], [180.000, 66.340], [29.236, 66.340]
        ]]}
    }
]

prepared_regions = []
for region in REGIONS:
    coords = region["geometry"]["coordinates"][0]
    prepared_regions.append((region["name"], prep(Polygon(coords))))


def match_regions(lon, lat):
    """
    Return all matching regions for a point.
    Returns None if the point does not fall in any region.
    """
    pt = Point(float(lon), float(lat))
    hits = []

    for name, prepared_poly in prepared_regions:
        if prepared_poly.covers(pt):  # includes boundary points
            hits.append(name)

    return hits if hits else None


def derive_genre(filename, suffix):
    """Extract genre from a filename by removing the configured suffix."""
    base = os.path.basename(filename)
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]



excel_files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN)))
print(f"Found {len(excel_files)} files in {INPUT_DIR} matching {FILE_PATTERN}:")
for fp in excel_files:
    print("  -", os.path.basename(fp))

region_names = [r["name"] for r in REGIONS]

for xls in excel_files:
    genre = derive_genre(xls, FILE_SUFFIX)
    print(f"\n=== Processing genre: {genre} ===")

    genre_dir = os.path.join(OUTPUT_ROOT, f"{genre}_csv")
    os.makedirs(genre_dir, exist_ok=True)

    try:
        df = pd.read_excel(xls, sheet_name=SHEET_NAME)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns {missing}; existing columns={list(df.columns)}"
            )

        # Clean numeric fields and drop rows without coordinates.
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["value_usd_ha_year"] = pd.to_numeric(
            df["value_usd_ha_year"], errors="coerce"
        )
        df = df.dropna(subset=["latitude", "longitude"]).copy()

        # Allow overlapping matches and expand them into one row per region.
        df["region"] = [
            match_regions(lon, lat)
            for lon, lat in zip(df["longitude"], df["latitude"])
        ]
        df_matched = df.dropna(subset=["region"]).explode("region")
        df_unmatched = df[df["region"].isna()].copy()

        # Export matched and unmatched records.
        keep_cols = ["latitude", "longitude", "value_usd_ha_year", "region"]
        df_matched = df_matched[keep_cols].copy()
        df_unmatched = df_unmatched[
            ["latitude", "longitude", "value_usd_ha_year"]
        ].copy()

        for reg in region_names:
            sub = df_matched[df_matched["region"] == reg].copy()
            out_csv = os.path.join(genre_dir, f"{genre}_{reg}.csv")
            sub.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[SAVED] {os.path.basename(out_csv)} rows={len(sub)}")

        unmatched_csv = os.path.join(genre_dir, f"{genre}_unmatched.csv")
        df_unmatched.to_csv(unmatched_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {os.path.basename(unmatched_csv)} rows={len(df_unmatched)}")

    except Exception as exc:
        print(f"[SKIP] {os.path.basename(xls)}")
        print(f"       Reason: {type(exc).__name__}: {exc}")
        continue

print("\n=== ALL DONE ===")