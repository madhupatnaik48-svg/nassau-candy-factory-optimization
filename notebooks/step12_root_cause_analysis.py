
# ============================================================
# STEP 12: DATA QUALITY ROOT CAUSE ANALYSIS
# Project: Nassau Candy Factory Optimization
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\Administrator\nassau-candy-factory-optimization"
)

DATA_DIR = PROJECT_DIR / "data"

STEP11_DIR = DATA_DIR / "step11_outputs"

OUTPUT_DIR = DATA_DIR / "step12_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_FILE = DATA_DIR / "Nassau Candy Distributor.csv"


# ------------------------------------------------------------
# 2. SET THE CORRECTED STEP 3 DATASET
# ------------------------------------------------------------

CORRECTED_FILE = (
    DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv"
)

if not CORRECTED_FILE.exists():
    raise FileNotFoundError(
        f"Corrected Step 3 dataset not found: {CORRECTED_FILE}"
    )


print("=" * 65)
print("STEP 12: ROOT CAUSE ANALYSIS")
print("=" * 65)

print("\nOriginal file:")
print(ORIGINAL_FILE)

print("\nCorrected Step 3 file:")
print(CORRECTED_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)

# ------------------------------------------------------------
# 3. LOAD DATASETS
# ------------------------------------------------------------

original = pd.read_csv(ORIGINAL_FILE)

corrected = pd.read_csv(CORRECTED_FILE)

print("\nDataset dimensions")
print("-" * 65)

print("Original records:", len(original))
print("Corrected records:", len(corrected))

print("\nOriginal columns:")
print(original.columns.tolist())

print("\nCorrected columns:")
print(corrected.columns.tolist())

# ------------------------------------------------------------
# 4. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Row ID",
    "Order Date",
    "Ship Date"
]

for name, df in [
    ("Original", original),
    ("Corrected", corrected)
]:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} dataset is missing required columns: {missing}"
        )

# ------------------------------------------------------------
# 5. CHECK ROW ID QUALITY
# ------------------------------------------------------------

row_id_results = []

for dataset_name, df in [
    ("Original", original),
    ("Corrected Step 3", corrected)
]:

    row_id_results.append({
        "Dataset": dataset_name,
        "Total Records": len(df),
        "Missing Row IDs": int(df["Row ID"].isna().sum()),
        "Duplicate Row IDs": int(
            df["Row ID"].duplicated(keep=False).sum()
        ),
        "Unique Row IDs": int(df["Row ID"].nunique())
    })

row_id_report = pd.DataFrame(row_id_results)

row_id_report.to_csv(
    OUTPUT_DIR / "row_id_quality_report.csv",
    index=False
)

print("\nROW ID QUALITY")
print(row_id_report.to_string(index=False))



# ------------------------------------------------------------
# 6. STANDARDIZE DATE VALUES FOR ANALYSIS
# ------------------------------------------------------------

date_columns = ["Order Date", "Ship Date"]

# Original dataset: DD-MM-YYYY
# Corrected Step 3 dataset: YYYY-MM-DD
#
# Parse each dataset using its actual date format.
# Do not modify the original date columns.

for column in date_columns:
    original[column + "_raw"] = original[column].astype("string")
    corrected[column + "_raw"] = corrected[column].astype("string")

    original[column + "_parsed"] = pd.to_datetime(
        original[column + "_raw"].str.strip(),
        format="%d-%m-%Y",
        errors="coerce"
    )

    corrected[column + "_parsed"] = pd.to_datetime(
        corrected[column + "_raw"].str.strip(),
        format="%Y-%m-%d",
        errors="coerce"
    )


# ------------------------------------------------------------
# 7. INVESTIGATE MISSING DATES AND PARSING FAILURES
# ------------------------------------------------------------

date_quality_results = []

for dataset_name, df in [
    ("Original", original),
    ("Corrected Step 3", corrected)
]:

    for column in date_columns:

        raw = df[column + "_raw"].str.strip()

        parsed = df[column + "_parsed"]

        truly_missing = (
            raw.isna()
            | raw.eq("")
        )

        parsing_failure = (
            ~truly_missing
            & parsed.isna()
        )

        date_quality_results.append({
            "Dataset": dataset_name,
            "Date Column": column,
            "Total Records": len(df),
            "Missing or Blank Dates": int(truly_missing.sum()),
            "Nonblank Values That Failed Parsing": int(
                parsing_failure.sum()
            ),
            "Successfully Parsed Dates": int(parsed.notna().sum())
        })

date_quality_report = pd.DataFrame(date_quality_results)

date_quality_report.to_csv(
    OUTPUT_DIR / "date_quality_root_cause_report.csv",
    index=False
)

print("\nDATE QUALITY REPORT")
print(date_quality_report.to_string(index=False))

# ------------------------------------------------------------
# 8. CALCULATE LEAD TIMES
# ------------------------------------------------------------

for df in [original, corrected]:

    df["Lead Time Days"] = (
        df["Ship Date_parsed"]
        - df["Order Date_parsed"]
    ).dt.days

# ------------------------------------------------------------
# 9. INVESTIGATE EXTREME LEAD TIMES
# ------------------------------------------------------------

lead_time_results = []

for dataset_name, df in [
    ("Original", original),
    ("Corrected Step 3", corrected)
]:

    lead = df["Lead Time Days"]

    valid = lead.dropna()

    lead_time_results.append({
        "Dataset": dataset_name,
        "Total Records": len(df),
        "Missing Lead Times": int(lead.isna().sum()),
        "Negative Lead Times": int((lead < 0).sum()),
        "Zero Day Lead Times": int((lead == 0).sum()),
        "Lead Times Over 365 Days": int((lead > 365).sum()),
        "Lead Times Over 730 Days": int((lead > 730).sum()),
        "Median Lead Time": (
            float(valid.median()) if len(valid) else np.nan
        ),
        "Maximum Lead Time": (
            float(valid.max()) if len(valid) else np.nan
        )
    })

lead_time_report = pd.DataFrame(lead_time_results)

lead_time_report.to_csv(
    OUTPUT_DIR / "lead_time_root_cause_report.csv",
    index=False
)

print("\nLEAD TIME REPORT")
print(lead_time_report.to_string(index=False))

# ------------------------------------------------------------
# 10. IDENTIFY ROW-LEVEL DATE DISCREPANCIES
# ------------------------------------------------------------

original_dates = original[
    ["Row ID", "Order Date_parsed", "Ship Date_parsed"]
].copy()

corrected_dates = corrected[
    ["Row ID", "Order Date_parsed", "Ship Date_parsed"]
].copy()

original_dates = original_dates.rename(columns={
    "Order Date_parsed": "Original Order Date",
    "Ship Date_parsed": "Original Ship Date"
})

corrected_dates = corrected_dates.rename(columns={
    "Order Date_parsed": "Corrected Order Date",
    "Ship Date_parsed": "Corrected Ship Date"
})

# Duplicate IDs can create many-to-many joins.
# Stop rather than silently produce unreliable comparisons.

if original_dates["Row ID"].duplicated().any():
    raise ValueError(
        "Original dataset contains duplicate Row IDs. "
        "Resolve the comparison grain before joining."
    )

if corrected_dates["Row ID"].duplicated().any():
    raise ValueError(
        "Corrected dataset contains duplicate Row IDs. "
        "Resolve the comparison grain before joining."
    )

comparison = original_dates.merge(
    corrected_dates,
    on="Row ID",
    how="outer",
    indicator=True,
    validate="one_to_one"
)

comparison["Issue"] = ""

comparison.loc[
    comparison["_merge"] == "left_only",
    "Issue"
] = "Row ID missing from corrected dataset"

comparison.loc[
    comparison["_merge"] == "right_only",
    "Issue"
] = "Row ID missing from original dataset"

matched = comparison["_merge"] == "both"

comparison.loc[
    matched
    & comparison["Original Order Date"].isna()
    & comparison["Corrected Order Date"].notna(),
    "Issue"
] = "Order Date missing in original but present after correction"

comparison.loc[
    matched
    & comparison["Original Order Date"].notna()
    & comparison["Corrected Order Date"].isna(),
    "Issue"
] = "Order Date became missing after correction"

comparison.loc[
    matched
    & comparison["Original Ship Date"].isna()
    & comparison["Corrected Ship Date"].notna(),
    "Issue"
] = "Ship Date missing in original but present after correction"

comparison.loc[
    matched
    & comparison["Original Ship Date"].notna()
    & comparison["Corrected Ship Date"].isna(),
    "Issue"
] = "Ship Date became missing after correction"

comparison.loc[
    matched
    & comparison["Original Order Date"].notna()
    & comparison["Corrected Order Date"].notna()
    & (
        comparison["Original Order Date"]
        != comparison["Corrected Order Date"]
    ),
    "Issue"
] = "Order Date changed after correction"

comparison.loc[
    matched
    & comparison["Original Ship Date"].notna()
    & comparison["Corrected Ship Date"].notna()
    & (
        comparison["Original Ship Date"]
        != comparison["Corrected Ship Date"]
    ),
    "Issue"
] = "Ship Date changed after correction"

comparison.to_csv(
    OUTPUT_DIR / "row_level_date_discrepancies.csv",
    index=False
)

# ------------------------------------------------------------
# 11. CREATE A ROW-LEVEL ROOT CAUSE INVESTIGATION FILE
# ------------------------------------------------------------

investigation = corrected[
    ["Row ID", "Order Date_raw", "Ship Date_raw",
     "Order Date_parsed", "Ship Date_parsed", "Lead Time Days"]
].copy()

investigation["Potential Root Cause"] = ""
investigation["Recommended Action"] = ""

missing_order = investigation["Order Date_parsed"].isna()
missing_ship = investigation["Ship Date_parsed"].isna()

investigation.loc[
    missing_order & missing_ship,
    "Potential Root Cause"
] = "Both order and ship dates are missing or unparseable"

investigation.loc[
    missing_order & ~missing_ship,
    "Potential Root Cause"
] = "Order date is missing or unparseable"

investigation.loc[
    ~missing_order & missing_ship,
    "Potential Root Cause"
] = "Ship date is missing or unparseable"

investigation.loc[
    investigation["Lead Time Days"] < 0,
    "Potential Root Cause"
] = "Ship date occurs before order date"

investigation.loc[
    investigation["Lead Time Days"] > 730,
    "Potential Root Cause"
] = "Lead time exceeds 730 days"

investigation.loc[
    investigation["Lead Time Days"].between(
        366, 730, inclusive="both"
    ),
    "Potential Root Cause"
] = "Lead time is between 366 and 730 days"

investigation.loc[
    investigation["Potential Root Cause"] == "",
    "Potential Root Cause"
] = "No issue detected by current checks"

investigation.loc[
    investigation["Potential Root Cause"].str.contains(
        "missing or unparseable|before order date|exceeds|between",
        case=False,
        regex=True
    ),
    "Recommended Action"
] = "Inspect source values and transformation logic; do not auto-correct"

investigation.loc[
    investigation["Potential Root Cause"]
    == "No issue detected by current checks",
    "Recommended Action"
] = "No action required by current checks"

investigation.to_csv(
    OUTPUT_DIR / "row_level_root_cause_investigation.csv",
    index=False
)

# ------------------------------------------------------------
# 12. SUMMARIZE ROOT CAUSE CATEGORIES
# ------------------------------------------------------------

root_cause_summary = (
    investigation["Potential Root Cause"]
    .value_counts(dropna=False)
    .rename_axis("Potential Root Cause")
    .reset_index(name="Record Count")
)

root_cause_summary.to_csv(
    OUTPUT_DIR / "root_cause_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 13. GENERATE FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 12 COMPLETED")
print("=" * 65)

print("\nPotential root causes identified:")

print(root_cause_summary.to_string(index=False))

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print("\nGenerated reports:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(" -", file.name)

print("\nIMPORTANT:")
print(
    "These are investigation findings, not proof of the underlying "
    "cause. Review the original values and transformation code "
    "before deciding whether any record should be corrected."
)

