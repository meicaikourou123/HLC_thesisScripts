import pandas as pd


def count_class(csv_file):
    # === 2. 读取数据 ===
    df = pd.read_csv(csv_file)

    # 检查是否有 class_code 列
    if "class_code" not in df.columns:
        raise ValueError("⚠️ The CSV file does not contain a 'class_code' column.")

    # === 3. 统计每种 class 的数量 ===
    count_table = df["class_code"].value_counts().sort_index().reset_index()
    count_table.columns = ["class_code", "count"]

    # === 4. 打印和保存 ===
    print("📊 Number of samples per class:")
    print(count_table)

    # 保存结果
    # count_table.to_csv("class_counts.csv", index=False)
    print("✅ Saved to class_counts.csv")


# === 1. 输入文件路径 ===
csv_file = "/Users/sunzheng/Python/LC_sample/Global_HRLC_Stratified_Samples.csv"  # 改成你的CSV文件名
count_class(csv_file)
