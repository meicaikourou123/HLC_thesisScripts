import pandas as pd
from count_csv_class import count_class
import numpy as np

# === 1. Load sample dataset ===
samples = pd.read_csv("/Users/sunzheng/Python/LC_sample/A123samples_stratified_merged3.csv")  # must have a 'class_code' column

# === 2. Load population summary ===
pop = pd.read_csv("/Users/sunzheng/Python/LC_sample/population_summary.csv")  # columns: class_code, population, percentage

# === 3. Calculate target sample count per class ===
# total_target = 10000  # desired total number of samples (you can adjust)
pop["raw_target"] = (pop["percentage"] )

# Apply min=50, max=500, but also check class availability in `samples`
target_list = []

for _, row in pop.iterrows():
    cls = row["class_code"]
    raw = row["raw_target"]
    available = len(samples[samples["class_code"] == cls])
    print(raw)
    n=2000
    # if available == 0:
    #     # no samples at all → skip
    #     n = 0
    # else:
    #     n=500
        # if available<1000:
        #     n=available
        # elif  available>1000:
        #     n=500
        # if cls ==70 or cls ==100:
        #     n=1000
        # if available<300:
        #     n = min(n, available)
        # # proportional number capped between 100 and 500
        # else:
        #     n=300
        # # cannot exceed available
        #     if raw <1:
        #         n=min(available, 250)
        #     elif raw>1 and raw<5:
        #         n=300
        #     elif raw>5 and raw<10:
        #         n=400
        #     elif raw > 10 and raw < 15:
        #         n = 500
        #     elif raw>15 and raw<20:
        #         n=600
        #     elif raw>20 and raw<25:
        #         n=700
        #     elif raw > 25and raw < 30:
        #         n = 800
        #     elif  raw > 30:
        #         n = 1000
            # if cls==100:
            #     n=1000
            # elif cls==90:
            #     n=500



    target_list.append({"class_code": cls, "target": n, "available": available})

target_df = pd.DataFrame(target_list)

print("📊 Target sample sizes:")
print(target_df)

# === 4. Perform stratified sampling ===
sampled_list = []

for _, row in target_df.iterrows():
    cls = row["class_code"]
    n = int(row["target"])
    subset = samples[samples["class_code"] == cls]
    if n == 0 or len(subset) == 0:
        continue
    sampled = subset.sample(n=n, random_state=42)
    sampled_list.append(sampled)

# === 5. Combine & save ===
sampled_all = pd.concat(sampled_list, ignore_index=True)
sampled_all.to_csv("/Users/sunzheng/Python/LC_sample/samples_stratified_2000.csv", index=False)
print(f"✅ Done. Total selected: {len(sampled_all)} saved to samples_stratified_limited.csv")
count_class('/Users/sunzheng/Python/LC_sample/samples_stratified_2000.csv')

