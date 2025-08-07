
import arcpy
from arcpy import env
from arcpy.sa import *
import os

env.workspace = r"G:\Acrgis\Mid-East Asia\Mid-East Asia.gdb"
env.overwriteOutput = True
env.addOutputsToMap = False  
csv_folder = r"C:\Users\hp\Desktop\desk\somethinneedpy\LinJia_mangrove\20250418By_Region_Excel"
output_csv_folder = r"G:\South_East_Asia_data"
input_folder = r"G:\250416regions\South_East_Asia"
os.makedirs(output_csv_folder, exist_ok=True)

typ_list = [
    'alk',
    'CP_1',
    'EC', 
    'EECR',
    'FM', 
    'FRM',
    'SOC',
    'SOC_1',
    'WPQ'
    ]

file_list = [
    'South_East_Asia_10.tif',
    'South_East_Asia_13.tif',
    'South_East_Asia_16.tif',
    'South_East_Asia_19.tif',
    'South_East_Asia_4.tif',
    'South_East_Asia_7.tif',
    'South_East_Asia_sum.tif'
]

if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    arcpy.AddError("Spatial Analyst can`t be used")
    raise SystemExit

try:
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    active_map = aprx.activeMap

    for typ in typ_list:
        all_csv_exist = True
        for tif in file_list:
            base = os.path.splitext(tif)[0]
            output_csv = os.path.join(output_csv_folder, f"{base}_{typ}.csv")
            if not os.path.exists(output_csv):
                all_csv_exist = False
                break
        
        if all_csv_exist:
            print(f"⚠️ skip {typ}")
            continue

        csv_file = os.path.join(csv_folder, f"{typ}_South_East_Asia.csv")
        print(f" typ={typ}  CSV :{csv_file}")
        if not os.path.exists(csv_file):
            arcpy.AddWarning(f"can`t find CSV :{csv_file},skip typ={typ}")
            continue

        table_view = f"{typ}_table_view"
        arcpy.management.MakeTableView(csv_file, table_view)
        sum_val = 0.0
        count = 0
        with arcpy.da.SearchCursor(table_view, ["value"]) as cursor:
            for row in cursor:
                if row[0] is not None:
                    sum_val += row[0]
                    count += 1
        mean = sum_val / count if count > 0 else 0.0
        print(f"{typ} CSV  value mean: {float(mean)}")

        point_fc = f"South_East_Asia_valuepoint_{typ}"
        if arcpy.Exists(point_fc):
            arcpy.management.Delete(point_fc)
        arcpy.management.XYTableToPoint(
            table_view,
            point_fc,
            x_field="Longitude",
            y_field="Latitude",
            z_field="value",
            coordinate_system=arcpy.SpatialReference(4326)
        )

        for tif in file_list:
            base = os.path.splitext(tif)[0]
            print(f"handling tif={tif}, typ={typ}")

            arcpy.conversion.RasterToPoint(
                os.path.join(input_folder, tif),
                f"{base}_{typ}_point",
                "Value"
            )

            arcpy.analysis.Buffer(
                f"{base}_{typ}_point",
                f"{base}_{typ}_5kmBuffer",
                "5 Kilometers"
            )

            arcpy.analysis.SpatialJoin(
                f"{base}_{typ}_5kmBuffer",
                point_fc,
                f"{base}_{typ}_5kmvalue",
                "JOIN_ONE_TO_ONE", "KEEP_ALL", None,
                "INTERSECT", "1 Meters", None
            )

            temp_lyr = arcpy.management.MakeFeatureLayer(
                f"{base}_{typ}_5kmvalue", f"null_lyr_{base}_{typ}", "value IS NULL"
            )
            arcpy.management.FeatureToPoint(
                temp_lyr,
                f"{base}_{typ}_10kmpoint",
                "INSIDE"
            )

            arcpy.analysis.Buffer(
                f"{base}_{typ}_10kmpoint",
                f"{base}_{typ}_10kmBuffer",
                "10 Kilometers"
            )

            arcpy.analysis.SpatialJoin(
                f"{base}_{typ}_10kmBuffer",
                point_fc,
                f"{base}_{typ}_10kmvalue",
                "JOIN_ONE_TO_ONE", "KEEP_ALL", None,
                "INTERSECT", "1 Meters", None
            )

            arcpy.management.CalculateField(
                f"{base}_{typ}_10kmvalue", "value_1",
                f"{float(mean)} if !value_1! is None else !value_1!", "PYTHON3"
            )
            arcpy.management.AddField(
                f"{base}_{typ}_10kmvalue", "value_tmp", "DOUBLE"
            )
            arcpy.management.CalculateField(
                f"{base}_{typ}_10kmvalue", "value_tmp",
                f"float(!value_1!) if !value_1! not in [None, ''] else {float(mean)}",
                "PYTHON3"
            )
            arcpy.management.DeleteField(f"{base}_{typ}_10kmvalue", "value_1")
            arcpy.management.AlterField(
                f"{base}_{typ}_10kmvalue", "value_tmp", new_field_name="value_1"
            )

            arcpy.management.JoinField(
                f"{base}_{typ}_5kmvalue", "TARGET_FID",
                f"{base}_{typ}_10kmvalue", "TARGET_FID",
                ["value_1"]
            )

            arcpy.management.CalculateField(
                f"{base}_{typ}_5kmvalue", "value_1",
                "!value_1! if !value_1! is not None else !value!", "PYTHON3"
            )

            arcpy.management.AddField(
                f"{base}_{typ}_5kmvalue", "vals", "DOUBLE"
            )
            arcpy.management.CalculateField(
                f"{base}_{typ}_5kmvalue", "vals",
                "!value_1! * !grid_code!", "PYTHON3"
            )

            arcpy.management.FeatureToPoint(
                f"{base}_{typ}_5kmvalue", f"{base}_{typ}_values", "INSIDE"
            )
            arcpy.management.CalculateGeometryAttributes(
                f"{base}_{typ}_values",
                [["Longitude", "POINT_X"], ["Latitude", "POINT_Y"]],
                coordinate_system=arcpy.SpatialReference(4326)
            )

            csv_name = f"{base}_{typ}.csv"
            arcpy.conversion.TableToTable(
                in_rows=f"{base}_{typ}_values",
                out_path=output_csv_folder,
                out_name=csv_name
            )

            total_vals = 0.0
            with arcpy.da.SearchCursor(f"{base}_{typ}_values", ["vals"]) as cur:
                for r in cur:
                    if r[0] is not None:
                        total_vals += r[0]
            print(f"{base} ({typ}) sum vals: {total_vals}")

            arcpy.conversion.PointToRaster(
                f"{base}_{typ}_values", "vals",
                f"{base}_{typ}_values_raster", "SUM", "NONE", 0.5
            )

            in_ras = Raster(f"{base}_{typ}_values_raster")
            for cond, suffix in [ (in_ras != 0, '_zero'), (in_ras <= 0, '_positive'), (in_ras >= 0, '_negative') ]:
                out_rast = SetNull(cond, in_ras)
                name = f"{base}{suffix}_{typ}"
                out_path = os.path.join(env.workspace, name)
                arcpy.management.CopyRaster(out_rast, out_path)
                active_map.addDataFromPath(out_path)
                print(f" {name} saved!")

    print("all typ done!")

except arcpy.ExecuteError:
    print(arcpy.GetMessages(2))

finally:
    arcpy.CheckInExtension("Spatial")

print("finished!")
