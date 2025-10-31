import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix

# ======================================
# 1️⃣ Read and merge
# ======================================
csv1 = '/Users/sunzheng/Python/LC_sample/A01samples_stratified3.csv'
csv2 = '/Users/sunzheng/Python/LC_sample/A02samples_stratified3.csv'
csv3 = '/Users/sunzheng/Python/LC_sample/A03samples_stratified3.csv'

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

df_all = pd.concat([df1, df2, df3], ignore_index=True)
print(f"✅ Merged total samples: {len(df_all)}")

# ======================================
# 2️⃣ Define spatially stratified sampling
# ======================================
def stratified_spatial_sample(df_class, target_n=2000, grid_size=0.1):
    """Perform spatial stratified sampling by grid."""
    df_class = df_class.copy()
    df_class['grid_x'] = (df_class['lon'] // grid_size).astype(int)
    df_class['grid_y'] = (df_class['lat'] // grid_size).astype(int)
    df_class['grid_id'] = df_class['grid_x'].astype(str) + "_" + df_class['grid_y'].astype(str)

    sampled = (
        df_class.groupby('grid_id', group_keys=False)
        .apply(lambda x: x.sample(1, random_state=42))
    )

    if len(sampled) < target_n:
        remain = df_class.drop(sampled.index)
        add = remain.sample(
            min(target_n - len(sampled), len(remain)),
            random_state=42
        )
        sampled = pd.concat([sampled, add])
    elif len(sampled) > target_n:
        sampled = sampled.sample(target_n, random_state=42)

    return sampled

# ======================================
# 2️⃣b Regional Stratified Sampling (Africa / Amazonia / Siberia)
# ======================================
def regionally_stratified_sample(df_class, target_n=2000, grid_size=0.1):
    """Perform region-wise + spatially uniform sampling."""
    if "region" not in df_class.columns:
        print("⚠️ No region column found; fallback to global sampling.")
        return stratified_spatial_sample(df_class, target_n, grid_size)

    region_counts = df_class["region"].value_counts(normalize=True).to_dict()
    sampled_regions = []

    for region, frac in region_counts.items():
        region_subset = df_class[df_class["region"] == region]
        region_target = int(round(target_n * frac))
        if len(region_subset) <= region_target:
            sampled_regions.append(region_subset)
            continue

        # Spatial grid-based sampling within region
        region_subset = region_subset.copy()
        region_subset["grid_x"] = (region_subset["lon"] // grid_size).astype(int)
        region_subset["grid_y"] = (region_subset["lat"] // grid_size).astype(int)
        region_subset["grid_id"] = region_subset["grid_x"].astype(str) + "_" + region_subset["grid_y"].astype(str)

        sampled = (
            region_subset.groupby("grid_id", group_keys=False)
            .apply(lambda x: x.sample(1, random_state=42))
        )

        if len(sampled) < region_target:
            remain = region_subset.drop(sampled.index)
            add = remain.sample(min(region_target - len(sampled), len(remain)), random_state=42)
            sampled = pd.concat([sampled, add])
        elif len(sampled) > region_target:
            sampled = sampled.sample(region_target, random_state=42)
        sampled_regions.append(sampled)

    return pd.concat(sampled_regions, ignore_index=True)

# ======================================
# 3️⃣ Apply per class
# ======================================
target_per_class = 2000
result = []

for c, sub in df_all.groupby('class_code'):
    n = len(sub)
    if n <= target_per_class:
        print(f"Class {c}: only {n} available, keep all")
        result.append(sub)
    else:
        sampled = regionally_stratified_sample(sub, target_per_class, grid_size=0.1)
        result.append(sampled)
        print(f"Class {c}: sampled {len(sampled)} from {n}")

df_sampled = pd.concat(result, ignore_index=True)
out_csv = '/Users/sunzheng/Python/LC_sample/A123samples_stratified_balanced.csv'
df_sampled.to_csv(out_csv, index=False)
print(f"\n✅ Saved balanced dataset -> {out_csv}")

# ======================================
# 4️⃣ Summary
# ======================================
counts = df_sampled["class_code"].value_counts().sort_index()
print("\n📊 Final per-class sample counts:")
print(counts)

# ======================================
# 5️⃣ Spatial Distribution Visualization & Metrics
# ======================================

# ---- 1️⃣ 全部类别在一张图 ----
classes = sorted(df_sampled["class_code"].unique())
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
for i, c in enumerate(classes):
    subset = df_sampled[df_sampled["class_code"] == c]
    plt.scatter(subset["lon"], subset["lat"], s=5, color=colors[i], label=f"{c}")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution by Class")
plt.legend(markerscale=3, fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---- 2️⃣ 每类独立密度分布 ----
sns.set_style("white")
g = sns.FacetGrid(df_sampled, col="class_code", col_wrap=4, height=3)
g.map_dataframe(sns.kdeplot, x="lon", y="lat", fill=True, cmap="viridis", thresh=0.05)
g.set_titles("Class {col_name}")
g.set_axis_labels("Longitude", "Latitude")
plt.suptitle("Per-class Sample Density", y=1.02)
plt.show()

# ---- 3️⃣ 计算空间分布指标 ----
metrics = []
for c, sub in df_sampled.groupby('class_code'):
    coords = sub[['lon', 'lat']].values
    if len(coords) > 1:
        dmat = distance_matrix(coords, coords)
        np.fill_diagonal(dmat, np.nan)
        mean_dist = np.nanmean(dmat)
        std_dist = np.nanstd(dmat)
        cv = std_dist / mean_dist if mean_dist > 0 else np.nan
    else:
        mean_dist, std_dist, cv = np.nan, np.nan, np.nan

    metrics.append({
        "class_code": c,
        "num_points": len(coords),
        "mean_distance": round(mean_dist, 4),
        "std_distance": round(std_dist, 4),
        "coefficient_of_variation": round(cv, 4)
    })

df_metrics = pd.DataFrame(metrics)
print("\n📊 Spatial Distribution Metrics (by class):")
print(df_metrics)

# ======================================
# 6️⃣ Regional Spatial Distribution Analysis
# ======================================

# Define the three regions by bounding boxes
regions = {
    "Africa":    {"lon_min": 9.9,  "lon_max": 43.3,  "lat_min": -0.1,  "lat_max": 18.1},
    "Amazonia":  {"lon_min": -62.1, "lon_max": -42.9, "lat_min": -23.6, "lat_max": 0.0},
    "Siberia":   {"lon_min": 64.4,  "lon_max": 93.4,  "lat_min": 51.3,  "lat_max": 75.7}
}

def assign_region(row):
    for name, bbox in regions.items():
        if (bbox["lon_min"] <= row["lon"] <= bbox["lon_max"]) and (bbox["lat_min"] <= row["lat"] <= bbox["lat_max"]):
            return name
    return "Other"

df_sampled["region"] = df_sampled.apply(assign_region, axis=1)

# Compute per (region, class_code)
region_metrics = []
for (region, c), sub in df_sampled.groupby(["region", "class_code"]):
    coords = sub[["lon", "lat"]].values
    if len(coords) > 1:
        dmat = distance_matrix(coords, coords)
        np.fill_diagonal(dmat, np.nan)
        mean_dist = np.nanmean(dmat)
        std_dist = np.nanstd(dmat)
        cv = std_dist / mean_dist if mean_dist > 0 else np.nan
    else:
        mean_dist, std_dist, cv = np.nan, np.nan, np.nan

    region_metrics.append({
        "region": region,
        "class_code": c,
        "num_points": len(coords),
        "mean_distance": round(mean_dist, 4),
        "std_distance": round(std_dist, 4),
        "RSI": round(cv, 4),
        "Uniformity_Score": round(1 / (1 + cv), 4) if not np.isnan(cv) else np.nan
    })

df_region_metrics = pd.DataFrame(region_metrics)

# Normalize by global mean distance per class
global_means = df_region_metrics.groupby("class_code")["mean_distance"].mean()
df_region_metrics["NMD"] = df_region_metrics.apply(
    lambda r: round(r["mean_distance"] / global_means[r["class_code"]], 4)
    if r["class_code"] in global_means else np.nan,
    axis=1
)

# Rank within each class by uniformity
df_region_metrics["Uniformity_Rank"] = df_region_metrics.groupby("class_code")["Uniformity_Score"].rank(ascending=False).astype(int)

print("\n📈 Regional Spatial Distribution Indexes (by class & region):")
print(df_region_metrics.sort_values(["class_code", "region"]))

# ======================================
# 7️⃣ Regional Uniformity Evaluation
# ======================================
region_summary = df_sampled.groupby(["region", "class_code"]).size().unstack(fill_value=0)
print("\n🌍 Sample Distribution by Region and Class:")
print(region_summary)

# ======================================
# 7️⃣b Save "Sample Distribution by Region and Class" as image
# ======================================
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(14, max(4, len(region_summary) * 1.2)))
ax.axis('off')

# 将 DataFrame 转为表格数据
cell_text = region_summary.values.tolist()
columns = region_summary.columns.tolist()
rows = region_summary.index.tolist()

# 创建表格
tbl = ax.table(
    cellText=cell_text,
    rowLabels=rows,
    colLabels=columns,
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.5)

plt.title("Sample Distribution by Region and Class", fontsize=12, fontweight='bold')
plt.savefig("/Users/sunzheng/Python/LC_sample/sample_distribution_by_region_and_class.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved sample distribution table image -> sample_distribution_by_region_and_class.png")

metrics = []
for (region, c), sub in df_sampled.groupby(["region", "class_code"]):
    coords = sub[["lon", "lat"]].values
    if len(coords) > 1:
        dmat = distance_matrix(coords, coords)
        np.fill_diagonal(dmat, np.nan)
        mean_d = np.nanmean(dmat)
        std_d = np.nanstd(dmat)
        cv = std_d / mean_d if mean_d > 0 else np.nan
        metrics.append({
            "region": region,
            "class_code": c,
            "num_points": len(coords),
            "mean_distance": round(mean_d, 4),
            "cv": round(cv, 4),
            "Uniformity_Score": round(1 / (1 + cv), 4)
        })

df_region_eval = pd.DataFrame(metrics)
print("\n📈 Regional Uniformity Metrics:")
print(df_region_eval.sort_values(["class_code", "region"]))

# ======================================
# 8️⃣ Save metrics table as image
# ======================================
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(14, max(6, len(df_region_eval) * 0.3)))
ax.axis('off')
# Convert DataFrame to string table
cell_text = df_region_eval.round(4).values.tolist()
columns = df_region_eval.columns.tolist()
# 绘制表格
tbl = ax.table(cellText=cell_text, colLabels=columns, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)

plt.title("Regional Uniformity Metrics", fontsize=12, fontweight='bold')
plt.savefig("/Users/sunzheng/Python/LC_sample/regional_uniformity_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved metrics table image -> regional_uniformity_metrics.png")