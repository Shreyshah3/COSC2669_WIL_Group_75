import pandas as pd
from pathlib import Path

# Find CSV files inside data/raw
raw_folder = Path("data/raw")
csv_files = list(raw_folder.glob("*.csv"))

print("CSV files found:", csv_files)

if not csv_files:
    print("ERROR: No CSV file found in data/raw/")
    raise SystemExit

# Load the first CSV
csv_file = csv_files[0]

print("\nReading:", csv_file)

df = pd.read_csv(csv_file)

print("\n========== DATASET INFORMATION ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nDuplicate rows:")
print(df.duplicated().sum())