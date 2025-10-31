import os, glob, math, random, json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from tqdm import tqdm

# =========================
# 配置
# =========================
TIF_DIR = r"E:\HLC_data\dap.ceda.ac.uk\neodc\esacci\high_resolution_land_cover\data\land_cover_maps\A01_Africa\static\v1.2\geotiff\HRLC10\tiles\2019"                 # .tif 文件夹
OUT_CSV = r"E:\HLC_data\samples\Africa_stratifieespatial.csv"
MAX_WORKERS = 12                             # 并发进程数；None=自动取CPU核心数
RANDOM_SEED = 42

# 需要抽样的类别（HRLC legend）
CLASS_CODES = [10,20,30,40,50,60,70,80,90,
               100,110,120,130,140,141,142,150]
NODATA_CODES = {0}                             # NoData 像元值
TOTAL_SAMPLES = 20000                          # 全局目标样本数
BALANCE_BY_CLASS = False                       # True=各类均衡；False=按总体占比
MIN_PER_CLASS = 1000                             # 每类最少样本数

# =========================
# 工具函数
# =========================
def is_class_map(path: str) -> bool:
    """过滤掉 UNCERT / PROB / CONF，只保留分类图"""
    name = os.path.basename(path).upper()
    if "UNCERT" in name or "PROB" in name or "CONF" in name:
        return False
    return True


def count_tile_classes(tif_path: str, class_codes: list[int]) -> tuple[str, Counter, str|None]:
    """Pass1: 统计每个tile的类别像元数（逐块，低内存）"""
    per_tile = Counter()
    try:
        with rasterio.open(tif_path) as src:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=True)
                mask2d = np.ma.getmaskarray(arr)   # 保证是二维mask
                if mask2d.all():
                    continue
                vals = np.asarray(arr)[~mask2d].astype(np.int64)
                if vals.size == 0:
                    continue
                if NODATA_CODES:
                    vals = vals[~np.isin(vals, list(NODATA_CODES))]
                    if vals.size == 0:
                        continue
                vals = vals[np.isin(vals, class_codes)]
                if vals.size == 0:
                    continue
                u, c = np.unique(vals, return_counts=True)
                per_tile.update({int(k): int(v) for k, v in zip(u, c)})
    except Exception as e:
        return tif_path, Counter(), f"{e}"
    return tif_path, per_tile, None


def spatially_uniform_pick(points: list[dict], grid_deg=0.1) -> list[dict]:
    """基于网格随机法筛选点，保证空间均匀性，每个网格最多保留一个点"""
    grid_dict = {}
    rng = random.Random(RANDOM_SEED)
    for pt in points:
        lon, lat = pt["lon"], pt["lat"]
        gx = int(math.floor(lon / grid_deg))
        gy = int(math.floor(lat / grid_deg))
        key = (gx, gy)
        if key not in grid_dict:
            grid_dict[key] = []
        grid_dict[key].append(pt)
    selected = []
    for cell_pts in grid_dict.values():
        selected.append(rng.choice(cell_pts))
    return selected


def sample_tile(tif_path: str, quota_per_class: dict[int,int], class_codes: list[int]) -> list[dict]:
    """Pass2: 库容抽样，返回点的WGS84坐标"""
    rows_out: list[dict] = []
    try:
        with rasterio.open(tif_path) as src:
            transformer = None
            if src.crs and str(src.crs).upper() != "EPSG:4326":
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

            rng = random.Random(RANDOM_SEED)
            reservoir = {c: [] for c in class_codes}
            seen = {c: 0 for c in class_codes}

            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=True)
                mask2d = np.ma.getmaskarray(arr)
                if mask2d.all():
                    continue
                rr, cc = np.where(~mask2d)
                if rr.size == 0:
                    continue
                vals = np.asarray(arr)[rr, cc].astype(np.int64)

                keep = np.isin(vals, class_codes)
                if NODATA_CODES:
                    keep &= ~np.isin(vals, list(NODATA_CODES))
                if not keep.any():
                    continue

                rr, cc, vals = rr[keep], cc[keep], vals[keep]
                if rr.size == 0:
                    continue

                rr_glob = rr + window.row_off
                cc_glob = cc + window.col_off

                for r, ccol, v in zip(rr_glob, cc_glob, vals):
                    q = quota_per_class.get(int(v), 0)
                    if q <= 0:
                        continue
                    seen[v] += 1
                    k = len(reservoir[v])
                    if k < q:
                        reservoir[v].append((int(r), int(ccol)))
                    else:
                        j = rng.randint(0, seen[v] - 1)
                        if j < q:
                            reservoir[v][j] = (int(r), int(ccol))

            # 输出点，先收集每类所有点，后进行空间均匀筛选
            for c in class_codes:
                pts_c = []
                for (r, ccol) in reservoir[c]:
                    x, y = xy(src.transform, r, ccol, offset="center")
                    if transformer:
                        lon, lat = transformer.transform(x, y)
                    else:
                        lon, lat = x, y
                    pts_c.append({
                        "tif_name": os.path.basename(tif_path),
                        "class_code": int(c),
                        "lon": float(lon),
                        "lat": float(lat)
                    })
                # 空间均匀筛选
                pts_c_selected = spatially_uniform_pick(pts_c, grid_deg=0.1)
                rows_out.extend(pts_c_selected)
    except Exception as e:
        print(f"⚠️ {os.path.basename(tif_path)}: {e}")
    return rows_out


# =========================
# 主流程
# =========================
def main():
    random.seed(RANDOM_SEED)
    tif_paths = [p for p in glob.glob(os.path.join(TIF_DIR, "*.tif")) if is_class_map(p)]
    if not tif_paths:
        raise RuntimeError("No classification GeoTIFFs found.")

    # Pass1: 并行统计
    tile_class_counts: dict[str, Counter] = {}
    global_class_counts = Counter()

    print("Pass 1/2: counting per-class pixels...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = [exe.submit(count_tile_classes, tif, CLASS_CODES) for tif in tif_paths]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            tif, counts, err = fut.result()
            if err:
                print(f"  ⚠️ skip {os.path.basename(tif)} due to {err}")
                continue
            tile_class_counts[tif] = counts
            global_class_counts.update(counts)

    # 计算每类目标样本数
    class_target: dict[int,int] = {}
    if BALANCE_BY_CLASS:
        per_class = max(MIN_PER_CLASS, TOTAL_SAMPLES // len(CLASS_CODES))
        for c in CLASS_CODES:
            class_target[c] = per_class
    else:
        denom = sum(global_class_counts[c] for c in CLASS_CODES)
        for c in CLASS_CODES:
            if global_class_counts[c] == 0:
                # 跳过不存在的类别
                continue
            prop = (global_class_counts[c] / denom) if denom > 0 else 0
            tgt = round(TOTAL_SAMPLES * prop)
            # 限制在1000到3000之间
            class_target[c] = max(1000, min(3000, tgt))

    # 把配额分配到每个tile
    tile_quota: dict[str,dict[int,int]] = {t: {c:0 for c in CLASS_CODES} for t in tif_paths}
    for c in CLASS_CODES:
        total_c = sum(tile_class_counts[t][c] for t in tif_paths)
        if total_c == 0:
            print(f"⚠️ Class {c} has 0 pixels globally. Skip.")
            continue
        if c not in class_target:
            continue
        # 初始分配
        raw = [(t, (tile_class_counts[t][c] / total_c) * class_target[c]) for t in tif_paths]
        floor_alloc = {t:int(math.floor(p)) for t,p in raw}
        remain = class_target[c] - sum(floor_alloc.values())
        residual_order = sorted(raw, key=lambda kv: kv[1]-math.floor(kv[1]), reverse=True)
        tile_alloc = floor_alloc.copy()
        for item in residual_order[:remain]:
            if isinstance(item, tuple) and len(item)==2:
                t, _ = item
                tile_alloc[t] += 1
        for t in tif_paths:
            cnt = tile_class_counts.get(t,Counter()).get(c,0)
            if cnt == 0:
                tile_quota[t][c] = 0
            else:
                tile_quota[t][c] = min(tile_alloc.get(t,0), cnt)

    # Pass2: 并行抽样
    print("Pass 2/2: sampling points...")
    rows_out_all: list[dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = [exe.submit(sample_tile, tif, tile_quota[tif], CLASS_CODES) for tif in tif_paths]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rows_out_all.extend(fut.result())

    # 保存
    df = pd.DataFrame(rows_out_all, columns=["tif_name","class_code","lon","lat"])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Done. Saved {len(df)} samples -> {OUT_CSV}")

    # 报告
    report_global = {int(c): int(global_class_counts[c]) for c in CLASS_CODES}
    report_target = {int(c): int(class_target[c]) for c in CLASS_CODES if c in class_target}
    report_actual = df["class_code"].value_counts().to_dict()

    print("\n===== Stratified Sampling Report =====")
    print("Class | Population (pixels) | Target (samples) | Actual (samples)")
    print("---------------------------------------------------------------")
    for c in CLASS_CODES:
        pop = report_global.get(c, 0)
        tgt = report_target.get(c, 0)
        act = report_actual.get(c, 0)
        print(f"{c:5d} | {pop:18d} | {tgt:17d} | {act:16d}")

    summary = {"population": report_global, "target": report_target, "actual": report_actual}
    with open("E:\HLC_data\samples\summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 额外输出CSV形式的每类统计
    class_summary_df = pd.DataFrame([
        {"class_code": c,
         "population_pixels": report_global.get(c, 0),
         "target_samples": report_target.get(c, 0),
         "actual_samples": report_actual.get(c, 0)}
        for c in CLASS_CODES
    ])
    summary_csv_path = "E:\HLC_data\samples\class_summary.csv"
    class_summary_df.to_csv(summary_csv_path, index=False)
    print(f"📊 Per-class summary saved -> {summary_csv_path}")


if __name__ == "__main__":
    main()
