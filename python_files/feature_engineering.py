import os
import pandas as pd

def generate_features(input_dir: str, output_dir: str):
    """
    Reads all CSV files from input_dir, computes rolling features and target column,
    and saves processed files to output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        print(f"Processing {file_path} ...")

        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp")

        # Rolling features
        df["rolling_avg_10"] = df["close"].rolling(window=10, min_periods=1).mean()
        df["volume_sum_10"] = df["volume"].rolling(window=10, min_periods=1).sum()

        # Target: close price 5 mins later > current close
        df["future_close"] = df["close"].shift(-5)
        df["target"] = (df["future_close"] > df["close"]).astype(int)

        # Drop rows where target is NaN (because of shift)
        df = df.dropna(subset=["target"]).reset_index(drop=True)

        # Extract stock symbol from filename (e.g., AARTIIND__EQ__NSE__NSE__MINUTE.csv → AARTIIND)
        stock_symbol = file.split("__")[0]
        df["stock_symbol"] = stock_symbol

        # Save processed file
        output_file = os.path.join(output_dir, f"processed_{file}")
        df.to_csv(output_file, index=False)
        print(f"✅ Saved processed file to: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rolling features and target labels.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw CSVs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed CSVs.")
    args = parser.parse_args()

    generate_features(args.input_dir, args.output_dir)
