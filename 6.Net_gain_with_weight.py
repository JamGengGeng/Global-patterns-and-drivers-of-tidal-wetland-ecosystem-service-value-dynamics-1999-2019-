"""
Compute yearly delta and cumulative values from region-level CSV files.

Apply the final gain/loss chain rules year by year:
   - grid == 1: start a gain chain and apply the first weight
   - grid == 0: continue an active chain with the next weight
   - grid == -1:
       * if a chain is active, subtract the accumulated weighted gain and stop
       * otherwise, subtract value_fill directly

"""

import os
import pandas as pd

ROOT = os.getenv("ROOT_DIR", "./data/count_weight")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "./outputs/final_results") #from "4.Calculate_value.py"

YEARS = ["04", "07", "10", "13", "16", "19"]

# Set to None to run all genres, or provide a set such as {"alk"}.
SELECTED_GENRES = None

GRID_COL = "grid_code"
GAIN_COL = "vals"
VALUE_COL = "value_fill"
LAT_COL = "Latitude"
LON_COL = "Longitude"

# If the gain started from 2004, then use W04. If the gain started from any other year, then use WOTH.
W04 = [0.575, 0.126, 0.086, 0.067, 0.056, 0.049]
WOTH = [0.472, 0.174, 0.101, 0.075, 0.061]


def discover_all_genres(root: str):
    """Discover all candidate genre directories under ROOT."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"ROOT does not exist: {root}")

    genres = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith("_")
    ]
    genres.sort()
    return genres


def is_region_dir(genre_dir: str, region_name: str) -> bool:
    if region_name.startswith("_"):
        return False

    region_dir = os.path.join(genre_dir, region_name)
    if not os.path.isdir(region_dir):
        return False

    return any(os.path.isdir(os.path.join(region_dir, y)) for y in YEARS)


def find_csv(region_dir: str, year: str, region: str, genre: str) -> str:
    year_dir = os.path.join(region_dir, year)
    if not os.path.isdir(year_dir):
        return ""

    preferred = os.path.join(year_dir, f"netgain_{region}_{year}_{genre}.csv")
    if os.path.exists(preferred):
        return preferred

    for fn in os.listdir(year_dir):
        if fn.lower().endswith(".csv"):
            return os.path.join(year_dir, fn)

    return ""


def add_latlon_keys_strict(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df["lat_str"] = df[LAT_COL].astype(str)
    df["lon_str"] = df[LON_COL].astype(str)
    return df


def build_year_df(csv_path: str, year: str) -> pd.DataFrame:
    """
    Read one yearly CSV and normalize required fields.
    """
    df = pd.read_csv(csv_path)

    required = [LAT_COL, LON_COL, GRID_COL, GAIN_COL, VALUE_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}: {csv_path}")

    df = df[required].copy()
    df = add_latlon_keys_strict(df)

    df[f"grid_{year}"] = pd.to_numeric(df[GRID_COL], errors="coerce").fillna(0).astype(int)
    df[f"gain_{year}"] = pd.to_numeric(df[GAIN_COL], errors="coerce").fillna(0.0)
    df[f"val_{year}"] = pd.to_numeric(df[VALUE_COL], errors="coerce").fillna(0.0)

    df = df[
        ["lat_str", "lon_str", LAT_COL, LON_COL,
         f"grid_{year}", f"gain_{year}", f"val_{year}"]
    ]

    # Resolve duplicated points within the same year.
    df = df.groupby(["lat_str", "lon_str"], as_index=False).agg({
        LAT_COL: "first",
        LON_COL: "first",
        f"grid_{year}": "max",
        f"gain_{year}": "mean",
        f"val_{year}": "mean",
    })

    return df


def build_wide_table(genre_dir: str, region: str, genre: str) -> pd.DataFrame:
    region_dir = os.path.join(genre_dir, region)

    year_dfs = {}
    for y in YEARS:
        csv_path = find_csv(region_dir, y, region, genre)
        if csv_path:
            year_dfs[y] = build_year_df(csv_path, y)

    if not year_dfs:
        return pd.DataFrame(columns=["lat_str", "lon_str", LAT_COL, LON_COL])

    merged = None
    for y in YEARS:
        if y not in year_dfs:
            continue

        dfy = year_dfs[y]

        if merged is None:
            merged = dfy.copy()
        else:
            # Keep Latitude/Longitude only from the first merged table.
            dfy2 = dfy.drop(columns=[LAT_COL, LON_COL], errors="ignore")
            merged = merged.merge(dfy2, on=["lat_str", "lon_str"], how="outer")

    if LAT_COL not in merged.columns:
        merged[LAT_COL] = pd.NA
    if LON_COL not in merged.columns:
        merged[LON_COL] = pd.NA

    for y in YEARS:
        if f"grid_{y}" not in merged.columns:
            merged[f"grid_{y}"] = 0
            merged[f"gain_{y}"] = 0.0
            merged[f"val_{y}"] = 0.0
        else:
            merged[f"grid_{y}"] = pd.to_numeric(
                merged[f"grid_{y}"], errors="coerce"
            ).fillna(0).astype(int)
            merged[f"gain_{y}"] = pd.to_numeric(
                merged[f"gain_{y}"], errors="coerce"
            ).fillna(0.0)
            merged[f"val_{y}"] = pd.to_numeric(
                merged[f"val_{y}"], errors="coerce"
            ).fillna(0.0)

    return merged


# Delta and cumulative calculation
def compute_delta_and_cumulative(row):
    """
    Apply the final gain/loss chain rules and return total.
    """
    delta = {y: 0.0 for y in YEARS}
    cumulative = {y: 0.0 for y in YEARS}
    running = 0.0

    chain_active = False
    gain_base = 0.0
    w_list = None
    w_idx = 0
    applied_weight_sum = 0.0

    def start_chain(start_year: str, base_gain: float):
        nonlocal chain_active, gain_base, w_list, w_idx, applied_weight_sum
        chain_active = True
        gain_base = base_gain
        w_list = W04 if start_year == "04" else WOTH
        w_idx = 0
        applied_weight_sum = 0.0

    def end_chain():
        nonlocal chain_active, gain_base, w_list, w_idx, applied_weight_sum
        chain_active = False
        gain_base = 0.0
        w_list = None
        w_idx = 0
        applied_weight_sum = 0.0

    for y in YEARS:
        g = int(row[f"grid_{y}"])
        gain_y = float(row[f"gain_{y}"])
        val_y = float(row[f"val_{y}"])

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

            start_chain(y, gain_y)

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

        delta[y] = d
        cumulative[y] = running

    out = {"final_total": running}
    for y in YEARS:
        out[f"delta_{y}"] = delta[y]
        out[f"cumulative_{y}"] = cumulative[y]

    return out


def process_one_genre_region(genre: str, region: str):

    genre_dir = os.path.join(ROOT, genre)

    out_dir = os.path.join(OUTPUT_ROOT, f"{genre}_final_outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"{genre}_{region}_final_total_latlon_STRICT_CLEARLOSS.csv"
    )


    if os.path.exists(out_csv):
        print(f"[SKIP] exists -> {out_csv}")
        return

    wide = build_wide_table(genre_dir, region, genre)
    if wide.empty:
        print(f"[SKIP] empty: genre={genre} region={region}")
        return

    calc_df = wide.apply(lambda r: pd.Series(compute_delta_and_cumulative(r)), axis=1)
    out = pd.concat([wide, calc_df], axis=1)

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

    base_cols = ["lat_str", "lon_str", LAT_COL, LON_COL]
    for y in YEARS:
        yearly_df = out[base_cols + [f"delta_{y}", f"cumulative_{y}"]].copy()
        yearly_df.rename(
            columns={
                f"delta_{y}": "delta",
                f"cumulative_{y}": "cumulative",
            },
            inplace=True,
        )

        yearly_out = os.path.join(
            out_dir,
            f"{genre}_{region}_yearly_cumulative_{y}.csv"
        )
        yearly_df.to_csv(yearly_out, index=False, encoding="utf-8-sig")

def main():
    all_genres = discover_all_genres(ROOT)

    print("\n=== ALL GENRES FOUND ===")
    for g in all_genres:
        print(" -", g)

    if SELECTED_GENRES is None:
        genres_to_run = all_genres
    else:
        genres_to_run = [g for g in all_genres if g in SELECTED_GENRES]

    if not genres_to_run:
        print("[STOP] No matching genres found")
        return

    for genre in genres_to_run:
        genre_dir = os.path.join(ROOT, genre)
        if not os.path.isdir(genre_dir):
            continue

        regions = [d for d in os.listdir(genre_dir) if is_region_dir(genre_dir, d)]
        regions.sort()

        for region in regions:
            print(f"\n=== Processing genre={genre} region={region} ===")
            process_one_genre_region(genre, region)


if __name__ == "__main__":
    main()