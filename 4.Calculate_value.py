'''
Note: arcpy cannot be installed here due to Esri license restrictions.

The uploaded scripts are for transparency. All analyses using arcpy were run locally in ArcGIS Pro.

For each available year of the same region:
   - convert the raster to points,
   - fill missing values sequentially using mean values within
     5 km, 10 km, and 50 km neighborhoods,
   - fill any remaining nulls with the genre-region median,
   - record the fill source in `fill_method`,
   - export selected fields to CSV.
'''

import arcpy
from arcpy import env
from arcpy.sa import *
import os
import re
import glob
import numpy as np

gdb_path = os.getenv("ARCGIS_GDB_PATH", r"./workspace/MyProject1.gdb")
csv_root = os.getenv("CSV_ROOT", r"./data/values")
raster_root = os.getenv("RASTER_ROOT", r"./data/netgain_tiles") # from "2.Cliptif_by_region.py"
output_root = os.getenv("OUTPUT_ROOT", r"./outputs/batch_outputs_5_10_50km_median_only")

env.workspace = gdb_path
env.overwriteOutput = True
env.addOutputsToMap = False

os.makedirs(output_root, exist_ok=True)

SAFE_VAL_F = "value"
VAL_FILL_F = "value_fill"
METHOD_F = "fill_method"

OUT_FIELDS = [
    "pointid", "grid_code",
    "Latitude", "Longitude",
    VAL_FILL_F, METHOD_F, "vals"
]

RADII_KM = [5, 10, 50]

if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise RuntimeError("Spatial Analyst extension is not available")

def safe_delete(x):
    try:
        if x and arcpy.Exists(x):
            arcpy.management.Delete(x)
    except Exception:
        pass

def get_fields(fc):
    return [f.name for f in arcpy.ListFields(fc)]

def get_count(fc):
    return int(arcpy.management.GetCount(fc)[0])

def pick_field(fields, names):
    lower = {f.lower(): f for f in fields}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def list_csv_jobs(root):
    """Find all CSV jobs as (genre, region, csv_path)."""
    jobs = []
    for d in os.listdir(root):
        if not d.endswith("_csv"):
            continue
        genre = d[:-4]
        for fp in glob.glob(os.path.join(root, d, "*.csv")):
            region = os.path.basename(fp).replace(f"{genre}_", "").replace(".csv", "")
            jobs.append((genre, region, fp))
    return jobs


def list_rasters(root):
    """Build a mapping: (region, year) -> raster_path."""
    out = {}
    pat = re.compile(r"netgain_(.+)_([0-9]{2})\.tif")
    for fp in glob.glob(os.path.join(root, "netgain_*_*.tif")):
        m = pat.search(os.path.basename(fp))
        if m:
            out[(m.group(1), m.group(2))] = fp
    return out

# =========================================================
# CSV -> GDB table -> point features
# =========================================================
def import_csv(csv_path, genre, region):
    """
    Import one CSV into the geodatabase and normalize the value field.
    """
    tbl = f"tbl_{genre}_{region}"[:60]
    safe_delete(tbl)
    arcpy.conversion.TableToTable(csv_path, env.workspace, tbl)

    f = get_fields(tbl)
    lat = pick_field(f, ["Lat.", "Lat_", "Latitude", "latitude"])
    lon = pick_field(f, ["Lon.", "Lon_", "Longitude", "longitude"])
    val = pick_field(f, ["Value_USD/ha/year", "Value_USD_ha_year", "value"])

    if not lat or not lon or not val:
        raise KeyError(f"Missing lat/lon/value field. Available fields: {f}")

    if SAFE_VAL_F not in f:
        arcpy.management.AddField(tbl, SAFE_VAL_F, "DOUBLE")

    arcpy.management.CalculateField(
        tbl,
        SAFE_VAL_F,
        f"float(!{val}!) if !{val}! is not None else None",
        "PYTHON3"
    )
    return tbl, lat, lon


def make_value_points(tbl, lon, lat, out_fc):
    """Convert the imported table to WGS84 point features."""
    safe_delete(out_fc)
    arcpy.management.XYTableToPoint(
        tbl,
        out_fc,
        lon,
        lat,
        coordinate_system=arcpy.SpatialReference(4326)
    )


def compute_median(vp):
    """Compute the fallback median from source value points."""
    vals = [v for (v,) in arcpy.da.SearchCursor(vp, [SAFE_VAL_F]) if v is not None]
    if not vals:
        raise RuntimeError("value_points is empty; median cannot be computed")
    return float(np.median(vals))

# =========================================================
# Radius-based mean filling
# =========================================================
def fill_by_radius_mean(rp, vp, radius_km):
    """
    Fill null values in raster points using the mean of source points
    within the given search radius.
    """
    lyr = f"lyr_null_{radius_km}"
    buf = f"buf_{radius_km}"
    sj = f"sj_{radius_km}"
    safe_delete(lyr)
    safe_delete(buf)
    safe_delete(sj)

    # Select only points that still need filling.
    arcpy.management.MakeFeatureLayer(rp, lyr, f"{VAL_FILL_F} IS NULL")
    if get_count(lyr) == 0:
        return 0

    # Buffer null target points and compute mean source value inside each buffer.
    arcpy.analysis.Buffer(lyr, buf, f"{radius_km} Kilometers")

    fms = arcpy.FieldMappings()
    fms.addTable(buf)

    fm = arcpy.FieldMap()
    fm.addInputField(vp, SAFE_VAL_F)
    fm.mergeRule = "Mean"
    of = fm.outputField
    of.name = f"mean_{radius_km}"
    fm.outputField = of
    fms.addFieldMap(fm)

    arcpy.analysis.SpatialJoin(
        buf,
        vp,
        sj,
        "JOIN_ONE_TO_ONE",
        "KEEP_ALL",
        field_mapping=fms
    )

    orig = "ORIG_FID" if "ORIG_FID" in get_fields(sj) else "ORIG_FID_1"
    meanf = of.name

    arcpy.management.JoinField(rp, "OBJECTID", sj, orig, [meanf])

    arcpy.management.CalculateField(
        rp,
        VAL_FILL_F,
        f"!{VAL_FILL_F}! if !{VAL_FILL_F}! is not None else !{meanf}!",
        "PYTHON3"
    )

    arcpy.management.CalculateField(
        rp,
        METHOD_F,
        f"'{radius_km}km' if (!{METHOD_F}! is None and !{meanf}! is not None) else !{METHOD_F}!",
        "PYTHON3"
    )

    arcpy.management.MakeFeatureLayer(rp, lyr, f"{VAL_FILL_F} IS NULL")
    return get_count(lyr)

# Median fallback
def fill_with_median(rp, median):
    """Fill any remaining null values with the precomputed median."""
    cnt = 0
    with arcpy.da.UpdateCursor(rp, [VAL_FILL_F, METHOD_F]) as cur:
        for v, m in cur:
            if v is None:
                cur.updateRow((median, "median"))
                cnt += 1
    return cnt

# CSV export
def export_csv(rp, out_dir, name):
    """
    Export selected fields and compute longitude, latitude, and `vals`.
    """
    if "Longitude" not in get_fields(rp):
        arcpy.management.AddField(rp, "Longitude", "DOUBLE")
    if "Latitude" not in get_fields(rp):
        arcpy.management.AddField(rp, "Latitude", "DOUBLE")

    arcpy.management.CalculateGeometryAttributes(
        rp,
        [["Longitude", "POINT_X"], ["Latitude", "POINT_Y"]],
        coordinate_system=arcpy.SpatialReference(4326)
    )

    if "vals" not in get_fields(rp):
        arcpy.management.AddField(rp, "vals", "DOUBLE")

    arcpy.management.CalculateField(
        rp,
        "vals",
        f"!{VAL_FILL_F}! * !grid_code!",
        "PYTHON3"
    )

    fms = arcpy.FieldMappings()
    for f in OUT_FIELDS:
        if f in get_fields(rp):
            fm = arcpy.FieldMap()
            fm.addInputField(rp, f)
            fms.addFieldMap(fm)

    arcpy.conversion.TableToTable(rp, out_dir, name, field_mapping=fms)

csv_jobs = list_csv_jobs(csv_root)
raster_map = list_rasters(raster_root)

for genre, region, csv_path in csv_jobs:
    years = sorted([yy for (r, yy) in raster_map if r == region])
    if not years:
        continue

    tbl, lat, lon = import_csv(csv_path, genre, region)

    vp = f"vp_{genre}_{region}"[:60]
    make_value_points(tbl, lon, lat, vp)

    # One median is computed per genre-region source CSV
    median_val = compute_median(vp)

    for yy in years:
        out_dir = os.path.join(output_root, genre, region, yy)
        os.makedirs(out_dir, exist_ok=True)

        out_csv = f"netgain_{region}_{yy}_{genre}.csv"
        if os.path.exists(os.path.join(out_dir, out_csv)):
            continue

        rp = f"rp_{genre}_{region}_{yy}"[:60]
        safe_delete(rp)

        # Convert the yearly raster to points 
        arcpy.conversion.RasterToPoint(raster_map[(region, yy)], rp, "Value")

        for f in [VAL_FILL_F, METHOD_F]:
            if f not in get_fields(rp):
                arcpy.management.AddField(
                    rp,
                    f,
                    "TEXT" if f == METHOD_F else "DOUBLE"
                )

        # expand the neighborhood radius
        for rkm in RADII_KM:
            rem = fill_by_radius_mean(rp, vp, rkm)
            print(f"{genre}-{region}-{yy}: {rkm}km remaining = {rem}")

        filled = fill_with_median(rp, median_val)
        print(f"{genre}-{region}-{yy}: median filled = {filled}")

        export_csv(rp, out_dir, out_csv)
        safe_delete(rp)

    safe_delete(vp)
    safe_delete(tbl)

arcpy.CheckInExtension("Spatial")
print("=== ALL DONE ===")