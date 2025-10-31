import os
import glob
import rasterio
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# CONFIGURATION
# ----------------------------
# 修改为你的路径（可以是某个区域的文件夹）
TIF_DIR = "/Volumes/New Volume/Tile Data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A01_Africa/static/v1.2/geotiff/HRLC10/tiles/2019"
OUT_CSV = "/Users/sunzheng/Python/LC_sample/samples/check_tif_validity.csv"

CLASS_CODES = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,141,142,150]
NODATA_CODES = {0}

# ----------------------------
# FUNCTION: check single tif
# ----------------------------
def check_tif(tif_path):
    result = {
        "tif_name": os.path.basename(tif_path),
        "path": tif_path,
        "valid": False,
        "error": None,
        "total_pixels": 0,
        "valid_pixels": 0,
        "unique_classes": [],
    }
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(1, masked=True)
            if arr.mask.all():
                result["error"] = "All pixels masked"
                return result

            data = np.ma.compressed(arr)
            result["total_pixels"] = data.size

            # 去掉NODATA
            data = data[~np.isin(data, list(NODATA_CODES))]
            result["valid_pixels"] = data.size

            if result["valid_pixels"] == 0:
                result["error"] = "No valid pixels (only nodata)"
                return result

            vals = np.unique(data)
            result["unique_classes"] = vals.tolist()

            # 是否包含有效的class codes
            if any(v in CLASS_CODES for v in vals):
                result["valid"] = True
            else:
                result["error"] = "No target classes found"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    return result

# ----------------------------
# MAIN
# ----------------------------
def main():
    tif_paths = sorted(glob.glob(os.path.join(TIF_DIR, "*.tif")))
    print(f"🗺 Found {len(tif_paths)} tiles in {TIF_DIR}")

    results = []
    max_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

    print(f"🚀 Checking tiles using {max_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(check_tif, tif): tif for tif in tif_paths}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = fut.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "tif_name": os.path.basename(futures[fut]),
                    "path": futures[fut],
                    "valid": False,
                    "error": f"{type(e).__name__}: {e}",
                    "total_pixels": 0,
                    "valid_pixels": 0,
                    "unique_classes": []
                })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Summary saved -> {OUT_CSV}")

    # 打印简单统计
    valid_count = df["valid"].sum()
    invalid_count = len(df) - valid_count
    print(f"\nValid: {valid_count}, Invalid: {invalid_count}")
    print("🚫 Invalid files:")
    print(df.loc[~df["valid"], ["tif_name", "error"]].head(20))

if __name__ == "__main__":
    main()