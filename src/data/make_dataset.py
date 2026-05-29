import os
import re
import pandas as pd
from glob import glob

# 1. Setup Base Directories Robustly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw", "MetaMotion")
DATA_INTERIM_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "interim")

# Make sure the export directory exists
os.makedirs(DATA_INTERIM_DIR, exist_ok=True)

# 2. Gather All CSV Data Files
files = glob(os.path.join(DATA_RAW_DIR, "*.csv"))
print(f"📊 Found {len(files)} total data files to process.")


# 3. Robust Data Reader Function
def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        filename = os.path.basename(f)
        parts = filename.split("-")
        
        # Guard against completely malformed filenames
        if len(parts) < 3:
            continue
            
        participant = parts[0]  
        label = parts[1]  
        category = parts[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        # Assign Datetime Index Immediately to preserve temporal sequencing
        if "epoch (ms)" in df.columns:
            df.index = pd.to_datetime(df["epoch (ms)"], unit="ms")
        elif "epoch" in df.columns:
            df.index = pd.to_datetime(df["epoch"], unit="ms")
        else:
            print(f"⚠️ Skipping {filename}: Missing timestamp epoch column.")
            continue

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Sort out sensors and drop raw temporal metadata to prevent text math crashes
        cols_to_drop = ["epoch (ms)", "time (01:00)", "elapsed (s)"]
        df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        if "Accelerometer" in filename:
            df_cleaned["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df_cleaned])
        elif "Gyroscope" in filename:
            df_cleaned["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df_cleaned])

    return acc_df, gyr_df


# Load and compile the separated datasets
acc_df, gyr_df = read_data_from_files(files)

# 4. Merge Data Tracks Intelligently
if gyr_df.empty:
    print("Gyroscope dataset empty. Building Accelerometer processing pipeline.")
    data_merged = acc_df.copy()
    rename_mapping = {"x-axis (g)": "acc_x", "y-axis (g)": "acc_y", "z-axis (g)": "acc_z"}
    data_merged = data_merged.rename(columns=rename_mapping)
elif acc_df.empty:
    print("Accelerometer dataset empty. Building Gyroscope processing pipeline.")
    data_merged = gyr_df.copy()
    rename_mapping = {"x-axis (deg/s)": "gyr_x", "y-axis (deg/s)": "gyr_y", "z-axis (deg/s)": "gyr_z"}
    data_merged = data_merged.rename(columns=rename_mapping)
else:
    print("Both tracks present. Running complete multi-sensor merge alignment.")
    data_merged = pd.concat([acc_df, gyr_df], axis=1)
    rename_mapping = {
        "x-axis (g)": "acc_x", "y-axis (g)": "acc_y", "z-axis (g)": "acc_z",
        "x-axis (deg/s)": "gyr_x", "y-axis (deg/s)": "gyr_y", "z-axis (deg/s)": "gyr_z"
    }
    data_merged = data_merged.rename(columns=rename_mapping)

# Drop any side-by-side metadata column duplicates from the concatenation
data_merged = data_merged.loc[:, ~data_merged.columns.duplicated()]

# Enforce Chronological Sorting across the dataset
data_merged = data_merged.sort_index()

# 5. Build the Dynamic Aggregation Dictionary
sampling = {
    "label": "last",
    "participant": "last",
    "category": "last",
    "set": "last"
}
# Automatically add whatever sensor axes are actually present in our merged structure
for col in data_merged.columns:
    if "acc_" in col or "gyr_" in col:
        sampling[col] = "mean"

# 6. Time-Series Resampling Split By Day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D")) if not g.empty]

print("⏱️ Resampling day chunks to 200ms intervals...")
data_resampled = pd.concat([df.resample(rule="200ms").agg(sampling).dropna() for df in days])

# Force tracking index labels back to standard integer sets cleanly
data_resampled["set"] = pd.to_numeric(data_resampled["set"], errors="coerce").fillna(0).astype(int)

# 7. Export Dataset File Output
output_file = os.path.join(DATA_INTERIM_DIR, "metamotion_resampled.pkl")
data_resampled.to_pickle(output_file)
print(f"✅ Preprocessing complete! Cleaned interim dataset exported to: {output_file}")