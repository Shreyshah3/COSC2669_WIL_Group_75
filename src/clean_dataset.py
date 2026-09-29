import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_file = Path("data/raw/banking_knowledge_base_1000.csv")
processed_folder = Path("data/processed")

processed_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(input_file)

print(f"Original rows: {len(df)}")
print(f"Original columns: {df.columns.tolist()}")

# --------------------------------------------------
# 3. Standardise column names
# --------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

df = df.rename(columns={
    "section": "section",
    "question": "question",
    "answer": "answer"
})

# --------------------------------------------------
# 4. Clean text fields
# --------------------------------------------------

for column in ["section", "question", "answer"]:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# --------------------------------------------------
# 5. Remove completely empty records
# --------------------------------------------------

df = df[
    (df["section"] != "") &
    (df["question"] != "") &
    (df["answer"] != "")
]

# --------------------------------------------------
# 6. Remove duplicate questions
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["question"],
    keep="first"
)

duplicates_removed = before_duplicates - len(df)

# --------------------------------------------------
# 7. Create FAQ ID
# --------------------------------------------------

df.insert(
    0,
    "faq_id",
    range(1, len(df) + 1)
)

# --------------------------------------------------
# 8. Final column order
# --------------------------------------------------

df = df[
    ["faq_id", "section", "question", "answer"]
]

# --------------------------------------------------
# 9. Validation
# --------------------------------------------------

print("\n========== CLEANING RESULTS ==========")

print(f"Final rows: {len(df)}")
print(f"Duplicates removed: {duplicates_removed}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal columns:")
print(df.columns.tolist())

# --------------------------------------------------
# 10. Save cleaned CSV
# --------------------------------------------------

clean_csv = processed_folder / "banking_faq_clean.csv"

df.to_csv(
    clean_csv,
    index=False,
    encoding="utf-8"
)

# --------------------------------------------------
# 11. Save Excel
# --------------------------------------------------

excel_file = processed_folder / "banking_faq_clean.xlsx"

df.to_excel(
    excel_file,
    index=False,
    sheet_name="Banking FAQs"
)

print("\nFiles created:")
print(clean_csv)
print(excel_file)

print("\nFirst 5 cleaned records:")
print(df.head())

print("\nDataset cleaning completed successfully!")