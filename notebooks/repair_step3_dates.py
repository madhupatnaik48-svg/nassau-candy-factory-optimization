import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ORIGINAL_FILE = DATA_DIR / "Nassau Candy Distributor.csv"
STEP3_FILE = DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3.csv"
OUTPUT_FILE = DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv"

print("=" * 65)
print("REPAIRING STEP 3 DATE FEATURES")
print("=" * 65)

# Load files without modifying the originals
original = pd.read_csv(ORIGINAL_FILE)
step3 = pd.read_csv(STEP3_FILE)

print(f"Original records: {len(original)}")
print(f"Step 3 records:   {len(step3)}")

# Confirm both files contain Row ID
if "Row ID" not in original.columns or "Row ID" not in step3.columns:
    raise ValueError("Row ID is missing from one of the files.")

# Ensure Row IDs are unique before matching
if original["Row ID"].duplicated().any():
    raise ValueError("Duplicate Row IDs found in original data.")

if step3["Row ID"].duplicated().any():
    raise ValueError("Duplicate Row IDs found in Step 3 data.")

if set(original["Row ID"]) != set(step3["Row ID"]):
    raise ValueError("Row IDs do not match between the two files.")

# Parse the original dates explicitly as day-month-year
original["Order Date"] = pd.to_datetime(
    original["Order Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

original["Ship Date"] = pd.to_datetime(
    original["Ship Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Stop if any original date cannot be parsed
if original["Order Date"].isna().any():
    raise ValueError("Some original order dates could not be parsed.")

if original["Ship Date"].isna().any():
    raise ValueError("Some original shipping dates could not be parsed.")

# Match the original dates to Step 3 using Row ID
date_lookup = original.set_index("Row ID")[["Order Date", "Ship Date"]]

step3 = step3.drop(
    columns=["Order Date", "Ship Date"],
    errors="ignore"
)

step3["Order Date"] = step3["Row ID"].map(date_lookup["Order Date"])
step3["Ship Date"] = step3["Row ID"].map(date_lookup["Ship Date"])

# Calculate lead time from the corrected dates
step3["Shipping Lead Time Days"] = (
    step3["Ship Date"] - step3["Order Date"]
).dt.days

# Rebuild date-based features
step3["Order Year"] = step3["Order Date"].dt.year
step3["Order Month"] = step3["Order Date"].dt.month
step3["Order Quarter"] = step3["Order Date"].dt.quarter
step3["Order Day of Week"] = step3["Order Date"].dt.dayofweek
step3["Order Day of Month"] = step3["Order Date"].dt.day

# Validate lead times
calculated = (
    pd.to_datetime(step3["Ship Date"])
    - pd.to_datetime(step3["Order Date"])
).dt.days

mismatches = int(
    (calculated != step3["Shipping Lead Time Days"]).sum()
)

missing_dates = int(
    step3[["Order Date", "Ship Date"]].isna().any(axis=1).sum()
)

# Save to a NEW file; preserve the existing Step 3 file
step3.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

print("\nVALIDATION RESULTS")
print("-" * 65)
print(f"Corrected records:          {len(step3)}")
print(f"Missing order/shipping dates: {missing_dates}")
print(f"Lead-time mismatches:       {mismatches}")
print(f"Average lead time:          {step3['Shipping Lead Time Days'].mean():.2f} days")
print(f"Median lead time:           {step3['Shipping Lead Time Days'].median():.2f} days")
print(f"Minimum lead time:          {step3['Shipping Lead Time Days'].min()} days")
print(f"Maximum lead time:          {step3['Shipping Lead Time Days'].max()} days")

print("\nFirst five corrected records:")
print(
    step3[
        ["Row ID", "Order Date", "Ship Date", "Shipping Lead Time Days"]
    ].head().to_string(index=False)
)

print(f"\nCorrected file saved to:\n{OUTPUT_FILE}")
print("\nThe original files have not been overwritten.")