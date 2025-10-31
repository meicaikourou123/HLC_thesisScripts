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
# 2️⃣ Statistical Summary
# ======================================
summary = df_all['class_code'].value_counts().reset_index()
summary.columns = ['class_code', 'count']
summary['percentage'] = (summary['count'] / summary['count'].sum()) * 100
print("\n📊 Class Distribution Summary:")
print(summary)

# Bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x='class_code', y='count', data=summary, palette='viridis')
plt.title("Sample Count per Class")
plt.xlabel("Class Code")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ======================================
# 3️⃣ Spatial Uniformity Metrics
# ======================================
from scipy.spatial import distance_matrix

metrics = []
for c, sub in df_all.groupby("class_code"):
    coords = sub[['lon', 'lat']].values
    if len(coords) > 1:
        dmat = distance_matrix(coords, coords)
        np.fill_diagonal(dmat, np.nan)
        mean_d = np.nanmean(dmat)
        std_d = np.nanstd(dmat)
        cv = std_d / mean_d if mean_d > 0 else np.nan
        metrics.append({
            'class_code': c,
            'num_points': len(coords),
            'mean_distance': mean_d,
            'std_distance': std_d,
            'coefficient_of_variation': cv,
            'uniformity_score': 1 / (1 + cv)
        })

df_spatial = pd.DataFrame(metrics)
print("\n📈 Spatial Uniformity Metrics per Class:")
print(df_spatial)

# ======================================
# 4️⃣ Regional Coverage
# ======================================
regions = {
    "Africa":    {"lon_min": 9.9, "lon_max": 43.3, "lat_min": -0.1, "lat_max": 18.1},
    "Amazonia":  {"lon_min": -62.1, "lon_max": -42.9, "lat_min": -23.6, "lat_max": 0.0},
    "Siberia":   {"lon_min": 64.4, "lon_max": 93.4, "lat_min": 51.3, "lat_max": 75.7}
}

def assign_region(row):
    for name, bbox in regions.items():
        if (bbox["lon_min"] <= row["lon"] <= bbox["lon_max"]) and (bbox["lat_min"] <= row["lat"] <= bbox["lat_max"]):
            return name
    return "Other"

df_all["region"] = df_all.apply(assign_region, axis=1)

region_summary = df_all.groupby(["region", "class_code"]).size().unstack(fill_value=0)
print("\n🌍 Sample Counts by Region and Class:")
print(region_summary)

# ======================================
# 5️⃣ Spatial Visualization
# ======================================
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_all, x="lon", y="lat", hue="class_code", palette="tab20", s=10)
plt.title("Spatial Distribution of Samples")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Class')
plt.tight_layout()
plt.show()