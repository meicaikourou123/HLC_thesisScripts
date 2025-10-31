import os, glob, math, random
from collections import Counter
import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------
# CONFIGURATION
# ------------------------
tif_dirs = [
    r"/Volumes/New Volume/Tile Data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A01_Africa/static/v1.2/geotiff/HRLC10/tiles/2019",
    r"/Volumes/New Volume/Tile Data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A02_Amazonia/static/v1.2/geotiff/HRLC10/tiles/2019",
    r"/Volumes/New Volume/Tile Data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A03_Siberia/static/v1.2/geotiff/HRLC10/tiles/2019"
]
out_csv = r"/Users/sunzheng/Python/LC_sample/Global_HRLC_Stratified_Samples.csv"
SAMPLES_PER_CLASS = 1000
CLASS_CODES = [10,20,30,40,50,60,70,80,90,
               100,110,120,130,140,141,142,150]
NODATA_CODES = {0}
random.seed(42)

# ------------------------
# STEP 1: COUNT PIXELS PER TILE
# ------------------------
def count_tile_classes(tif_path):
    per_tile = Counter()
    try:
        with rasterio.open(tif_path) as src:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=True)
                if arr.mask.all():
                    continue
                data = np.ma.compressed(arr)
                data = data[~np.isin(data, list(NODATA_CODES))]
                data = data[np.isin(data, CLASS_CODES)]
                if data.size == 0:
                    continue
                vals, cnts = np.unique(data, return_counts=True)
                per_tile.update({int(v): int(c) for v, c in zip(vals, cnts)})
    except Exception as e:
        return tif_path, Counter(), str(e)
    return tif_path, per_tile, None

# ------------------------
# STEP 2: STRATIFIED SAMPLING (Reservoir)
# ------------------------
def sample_tile(tif_path, quota_per_class, class_codes):
    rows_out = []
    try:
        with rasterio.open(tif_path) as src:
            transformer = None
            if src.crs and str(src.crs).upper() != "EPSG:4326":
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

            rng = random.Random(42)
            reservoir = {c: [] for c in class_codes}
            seen = {c: 0 for c in class_codes}

            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=True)
                if arr.mask.all():
                    continue
                rr, cc = np.where(~arr.mask)
                vals = arr[rr, cc].astype(np.int64)
                mask_keep = (~np.isin(vals, list(NODATA_CODES))) & (np.isin(vals, class_codes))
                rr, cc, vals = rr[mask_keep], cc[mask_keep], vals[mask_keep]
                if len(rr) == 0:
                    continue
                rr_glob = rr + window.row_off
                cc_glob = cc + window.col_off
                for r, ccol, val in zip(rr_glob, cc_glob, vals):
                    quota = quota_per_class.get(int(val), 0)
                    if quota <= 0:
                        continue
                    seen[val] += 1
                    k = len(reservoir[val])
                    if k < quota:
                        reservoir[val].append((int(r), int(ccol)))
                    else:
                        j = rng.randint(0, seen[val] - 1)
                        if j < quota:
                            reservoir[val][j] = (int(r), int(ccol))

            region = os.path.basename(os.path.dirname(os.path.dirname(tif_path)))
            for c in class_codes:
                for (r, ccol) in reservoir[c]:
                    x, y = xy(src.transform, r, ccol, offset="center")
                    if transformer:
                        lon, lat = transformer.transform(x, y)
                    else:
                        lon, lat = x, y
                    rows_out.append({
                        "region": region,
                        "tif_name": os.path.basename(tif_path),
                        "class_code": c,
                        "lon": lon,
                        "lat": lat
                    })
    except Exception as e:
        print(f"⚠️ {os.path.basename(tif_path)}: {e}")
    return rows_out

# ------------------------
# MAIN
# ------------------------
def main():
    tif_paths = []
    for d in tif_dirs:
        tif_paths.extend(sorted(glob.glob(os.path.join(d, "*.tif"))))
    print(f"🗺 Found {len(tif_paths)} tiles from {len(tif_dirs)} regions")

    # Pass 1: count pixels per tile
    print("📊 Counting pixels (parallel)...")
    tile_class_counts = {}
    global_class_counts = Counter()
    with ProcessPoolExecutor(max_workers=6) as exe:
        futures = [exe.submit(count_tile_classes, tif) for tif in tif_paths]
        for f in tqdm(as_completed(futures), total=len(futures)):
            tif, counts, err = f.result()
            if err:
                continue
            tile_class_counts[tif] = counts
            global_class_counts.update(counts)

    # Compute sampling quota
    tile_quota = {t: {c: 0 for c in CLASS_CODES} for t in tif_paths}
    for c in CLASS_CODES:
        # find tiles that actually contain this class
        tiles_with_c = [t for t in tif_paths if tile_class_counts[t][c] > 0]
        if not tiles_with_c:
            print(f"⚠️ Class {c} not found in any tile, skip.")
            continue

        total_c = sum(tile_class_counts[t][c] for t in tiles_with_c)
        if total_c == 0:
            continue

        # assign proportional quotas only to those tiles
        for t in tiles_with_c:
            ratio = tile_class_counts[t][c] / total_c
            tile_quota[t][c] = int(round(ratio * SAMPLES_PER_CLASS))

        # fix rounding discrepancy
        total_assigned = sum(tile_quota[t][c] for t in tiles_with_c)
        diff = SAMPLES_PER_CLASS - total_assigned
        if diff > 0:
            for t in random.sample(tiles_with_c, min(diff, len(tiles_with_c))):
                tile_quota[t][c] += 1

        available = total_c
        if available < SAMPLES_PER_CLASS:
            print(f"⚠️ Class {c} has only {available} available pixels, will sample up to that.")

    # Pass 2: sampling
    print("🎯 Sampling points (parallel)...")
    rows_out_all = []
    with ProcessPoolExecutor(max_workers=6) as exe:
        futures = [exe.submit(sample_tile, tif, tile_quota[tif], CLASS_CODES) for tif in tif_paths]
        for f in tqdm(as_completed(futures), total=len(futures)):
            rows_out_all.extend(f.result())

    # Save results
    df = pd.DataFrame(rows_out_all)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved {len(df)} samples -> {out_csv}")

    # Report global pixel counts across all tiles
    global_counts_df = pd.DataFrame(list(global_class_counts.items()), columns=["class_code", "pixel_count"])
    global_counts_df = global_counts_df.sort_values("class_code").reset_index(drop=True)
    print("\n📊 Total pixel counts across all tiles:")
    print(global_counts_df.to_string(index=False))
    global_counts_df.to_csv("/Users/sunzheng/Python/LC_sample/global_class_pixel_counts.csv", index=False)

if __name__ == "__main__":
    main()