import pandas as pd
import glob
import pyarrow.parquet as pq
import os

def merge_parquets():
    v0_files = glob.glob("processed_data/processed_v0/*.parquet")
    v1_files = glob.glob("processed_data/processed_v1/*.parquet")

    print("Found v0 files:", len(v0_files))
    print("Found v1 files:", len(v1_files))

    dfs = []

    for f in v0_files + v1_files:
        df = pd.read_parquet(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print("Merged shape:", merged.shape)

    # IMPORTANT: convert timestamp to microseconds
    ts = pd.to_datetime(merged["timestamp"])
    # ✅ If timestamp is tz-naive, localize to IST (+05:30)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Kolkata")
    # ✅ Convert everything to UTC and then drop timezone (naive UTC)
    merged["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    merged["timestamp"] = (ts.astype('int64') // 1000) # convert ns → µs
    merged["timestamp"] = merged['timestamp'].astype('datetime64[us]')

    merged = merged.dropna(subset=["rolling_avg_10", "volume_sum_10"]).reset_index(drop=True)

    print(merged.info())

    os.makedirs("processed_data/merged_files", exist_ok=True)

    out_path = "processed_data/merged_files/stock_features_all_v1.parquet"
    merged.to_parquet(out_path, index=False)

    print("✅ Saved merged file to:", out_path)

if __name__ == "__main__":
    merge_parquets()
