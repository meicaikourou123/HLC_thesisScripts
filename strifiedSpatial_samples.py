import os, glob, math, random
from collections import Counter
import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing

# ------------------------
# CONFIGURATION
# ------------------------

tile_root=''
tif_dirs = [
    # r"E:/HLC_data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A01_Africa/static/v1.2/geotiff/HRLC10/tiles/2019",
    r"E:/HLC_data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A02_Amazonia/static/v1.2/geotiff/HRLC10/tiles/2019",
    r"E:/HLC_data/dap.ceda.ac.uk/neodc/esacci/high_resolution_land_cover/data/land_cover_maps/A03_Siberia/static/v1.2/geotiff/HRLC10/tiles/2019"
]
# regionlist=['Africa','Amazonia','Siberia']
regionlist=['Amazonia','Siberia']
out_root = r"E:/HLC_data/samples/"
SAMPLES_PER_CLASS = 1000
CLASS_CODES = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,141,142,150]
NODATA_CODES = {0}
random.seed(42)

# ------------------------
# STEP 1: COUNT PIXELS PER TILE
# ------------------------
def count_tile_classes(tif_path):
    per_tile = Counter()
    try:
        with rasterio.open(tif_path) as src:
            for win_obj in src.block_windows(1):
                if isinstance(win_obj, tuple) and len(win_obj) == 2:
                    _, window = win_obj
                else:
                    window = win_obj
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
        return tif_path, Counter(), f"{type(e).__name__} - {e}"
    return tif_path, per_tile, None

# ------------------------
# SPACE-UNIFORM SAMPLING
# ------------------------
def spatially_uniform_pick(points, grid_deg=0.1):
    df = pd.DataFrame(points, columns=["lon", "lat"])
    df["gx"] = (df["lon"] / grid_deg).astype(int)
    df["gy"] = (df["lat"] / grid_deg).astype(int)
    grouped = df.groupby(["gx", "gy"])
    selected = grouped.apply(lambda g: g.sample(1, random_state=42))
    return selected[["lon", "lat"]].values.tolist()

# ------------------------
# STEP 2: SAMPLE TILE
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

            for i, win_obj in enumerate(src.block_windows(1)):
                if i % 10 != 0:  # 只处理部分 block，降低内存占用
                    continue
                if isinstance(win_obj, tuple) and len(win_obj) == 2:
                    _, window = win_obj
                else:
                    window = win_obj
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
                points = []
                for (r, ccol) in reservoir[c]:
                    x, y = xy(src.transform, r, ccol, offset="center")
                    if transformer:
                        lon, lat = transformer.transform(x, y)
                    else:
                        lon, lat = x, y
                    points.append((lon, lat))
                # enforce spatial uniformity
                if len(points) > 2000:
                    points = spatially_uniform_pick(points, grid_deg=0.1)
                for lon, lat in points:
                    rows_out.append({
                        "region": region,
                        "tif_name": os.path.basename(tif_path),
                        "class_code": c,
                        "lon": lon,
                        "lat": lat
                    })
    except Exception as e:
        print(f"⚠️ Error processing {os.path.basename(tif_path)}: {type(e).__name__} - {e}")
    return rows_out

# ------------------------
# MAIN
# ------------------------
def process_region(region_dir, regionname):
    region_name = regionname
    tif_paths = sorted(glob.glob(os.path.join(region_dir, "*.tif")))
    tif_paths = [os.path.normpath(p) for p in tif_paths]  # ✅ 统一路径格式
    print(f"\n🌍 Processing region: {region_name}, found {len(tif_paths)} tiles")

    # --- Load intermediate tile_class_counts if already saved ---
    # --- Load intermediate tile_class_counts if already saved ---
    intermediate_cache = os.path.join(out_root, f"{region_name}_tile_class_counts.json")
    if os.path.exists(intermediate_cache):
        print(f"📂 Loading intermediate cached tile class counts from {intermediate_cache}")
        with open(intermediate_cache, 'r') as f:
            cached = json.load(f)

        # ✅ Normalize all paths for matching
        tile_class_counts = {os.path.normpath(k): Counter(v) for k, v in cached["tile_class_counts"].items()}
        global_class_counts = Counter(cached["global_class_counts"])
    else:
        print("⚠️ Intermediate cache not found, proceeding with existing in-memory data.")

    # ✅ Normalize tif paths too
    tif_paths = [os.path.normpath(p) for p in tif_paths]

    # ✅ 提供备用按文件名查找机制，防止路径不完全匹配
    cached_by_name = {os.path.basename(k): v for k, v in tile_class_counts.items()}
    def get_counts_for_path(p):
        return tile_class_counts.get(p) or cached_by_name.get(os.path.basename(p)) or Counter()

    # --- Assign quotas with adaptive total sampling (30,000 globally) ---
    TOTAL_SAMPLES = 30000
    MIN_SAMPLES_PER_CLASS = 1000
    MAX_SAMPLES_PER_CLASS = 3000
    print("🔍 Example cache key:", list(tile_class_counts.keys())[0])
    print("🔍 Example tif path:", tif_paths[0])

    total_pixels_all_classes = sum(global_class_counts[c] for c in CLASS_CODES)
    tile_quota = {t: {c: 0 for c in CLASS_CODES} for t in tif_paths}

    for c in CLASS_CODES:
        tiles_with_c, total_c = [], 0
        for t in tif_paths:
            cnts = get_counts_for_path(t)
            if cnts[c] > 0:
                tiles_with_c.append(t)
                total_c += cnts[c]
        if not tiles_with_c or total_c == 0:
            print(f"⚠️ Class {c} not found in {region_name}")
            continue

        # 按像元占比计算样本数量，并限制在 [1000, 3000]
        proportion = total_c / total_pixels_all_classes
        target_c = int(round(TOTAL_SAMPLES * proportion))
        target_c = min(MAX_SAMPLES_PER_CLASS, max(MIN_SAMPLES_PER_CLASS, target_c))
        target_c = min(target_c, total_c)  # 如果像元不足则取最大可能数

        # 按比例分配到各 tile
        assigned = 0
        for i, t in enumerate(tiles_with_c):
            cnts = get_counts_for_path(t)
            if i < len(tiles_with_c) - 1:
                q = int(round(cnts[c] / total_c * target_c))
                tile_quota[t][c] = q
                assigned += q
            else:
                tile_quota[t][c] = max(0, target_c - assigned)

        print(f"📊 Class {c:3d} | total_pixels={total_c:8d} | target_samples={target_c:5d} | tiles={len(tiles_with_c)}")

    # ✅ 检查配额是否全为 0
    total_quota = sum(sum(tile_quota[t].values()) for t in tif_paths)
    if total_quota == 0:
        print(f"❌ All quotas are 0 in {region_name}. Path mismatch likely.")
        return pd.DataFrame(columns=["region", "tif_name", "class_code", "lon", "lat"])

    # --- Sampling ---
    rows_out_all = []
    max_workers = max(2, min(6, multiprocessing.cpu_count() // 2))
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(sample_tile, tif, tile_quota[tif], CLASS_CODES) for tif in tif_paths]
        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result:
                rows_out_all.extend(result)
            # 定期保存局部结果
            if len(rows_out_all) % 20000 == 0 and len(rows_out_all) > 0:
                tmp_csv = os.path.join(out_root, f"{region_name}_partial.csv")
                pd.DataFrame(rows_out_all, columns=["region","tif_name","class_code","lon","lat"]).to_csv(tmp_csv, index=False)
                print(f"🕒 Partial progress saved -> {tmp_csv}")

    # ✅ 固定列名构建 DataFrame（即使空）
    df = pd.DataFrame(rows_out_all, columns=["region","tif_name","class_code","lon","lat"])
    region_csv = os.path.join(out_root, f"{region_name}_Stratified_Spatial_Samples.csv")
    df.to_csv(region_csv, index=False)
    print(f"✅ Saved {len(df)} samples for {region_name} -> {region_csv}")

    # ✅ 若 df 为空则短路返回，避免 KeyError
    if df.empty:
        print(f"⚠️ No samples produced for {region_name}. Please verify cache/path consistency.")
        return df

    # --- Summary report ---
    print("\n📋 Sampling Summary for Region:", region_name)
    summary_data = []
    for c in CLASS_CODES:
        total_pixels = global_class_counts.get(c, 0)
        actual_samples = len(df[df["class_code"] == c])
        expected_target = min(MAX_SAMPLES_PER_CLASS, max(MIN_SAMPLES_PER_CLASS,
                                int(round(TOTAL_SAMPLES * (total_pixels / total_pixels_all_classes)))))
        status = "✅" if actual_samples >= expected_target else "❌"
        summary_data.append({
            "Class": c,
            "Total Pixels": total_pixels,
            "Target Samples": expected_target,
            "Actual Samples": actual_samples,
            "Status": status
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("\n⚠️  Legend: ✅ sufficient | ❌ below target\n")

    return df

def main():
    all_samples = []
    for d,n in zip(tif_dirs, regionlist):
        region_df = process_region(d,n)
        all_samples.append(region_df)
    global_df = pd.concat(all_samples, ignore_index=True)
    global_csv = os.path.join(out_root, "Global_HRLC_Stratified_Spatial_Samples.csv")
    global_df.to_csv(global_csv, index=False)
    print(f"\n🌎 Global combined samples saved -> {global_csv}")

if __name__ == "__main__":
    main()