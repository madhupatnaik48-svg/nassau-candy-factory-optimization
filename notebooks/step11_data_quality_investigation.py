"""
Step 11: Data Quality and Shipping-Date Pattern Investigation
Project: Nassau Candy Factory Optimization

Inputs:
    data/Nassau Candy Distributor.csv
    data/Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv

Outputs:
    data/step11_outputs/step11_data_quality_summary.txt
    data/step11_outputs/shipping_lead_time_distribution.csv
    data/step11_outputs/date_consistency_by_year.csv
    data/step11_outputs/potential_data_quality_flags.csv

This script investigates data quality without modifying source datasets.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ORIGINAL_FILE = DATA_DIR / "Nassau Candy Distributor.csv"

CORRECTED_FILE = (
    DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv"
)

OUTPUT_DIR = DATA_DIR / "step11_outputs"

DATE_COLS = ["Order Date", "Ship Date"]
TARGET = "Shipping Lead Time Days"


# ============================================================
# 2. DATE-PARSING FUNCTIONS
# ============================================================

def parse_original_date(series):
    """
    Parse original source dates in DD-MM-YYYY format.
    Example: 03-01-2024 means 3 January 2024.
    """
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="%d-%m-%Y",
        errors="coerce"
    )


def parse_corrected_date(series):
    """
    Parse corrected Step 3 dates in YYYY-MM-DD format.
    Example: 2024-01-03 means 3 January 2024.
    """
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="%Y-%m-%d",
        errors="coerce"
    )


def read_csv_checked(path):
    """Read a CSV file and report a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

    return pd.read_csv(path, low_memory=False)


# ============================================================
# 3. MAIN ANALYSIS
# ============================================================

def main():

    print("\nSTEP 11: DATA QUALITY INVESTIGATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    original = read_csv_checked(ORIGINAL_FILE)
    corrected = read_csv_checked(CORRECTED_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Original records: {len(original):,}")
    print(f"Corrected Step 3 records: {len(corrected):,}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_original = [
        "Row ID",
        "Order ID",
        "Order Date",
        "Ship Date"
    ]

    required_corrected = [
        "Row ID",
        "Order Date",
        "Ship Date",
        TARGET
    ]

    missing_original = [
        col for col in required_original
        if col not in original.columns
    ]

    missing_corrected = [
        col for col in required_corrected
        if col not in corrected.columns
    ]

    if missing_original:
        raise ValueError(
            f"Original dataset is missing columns: {missing_original}"
        )

    if missing_corrected:
        raise ValueError(
            f"Corrected dataset is missing columns: {missing_corrected}"
        )

    # --------------------------------------------------------
    # Parse dates using the correct format for each dataset
    # --------------------------------------------------------

    for col in DATE_COLS:
        original[col] = parse_original_date(original[col])
        corrected[col] = parse_corrected_date(corrected[col])

    # --------------------------------------------------------
    # Calculate lead times independently
    # --------------------------------------------------------

    original["_calculated_lead_time_days"] = (
        original["Ship Date"] - original["Order Date"]
    ).dt.days

    corrected["_calculated_lead_time_days"] = (
        corrected["Ship Date"] - corrected["Order Date"]
    ).dt.days

    corrected[TARGET] = pd.to_numeric(
        corrected[TARGET],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Row ID quality checks
    # --------------------------------------------------------

    original_missing_ids = int(original["Row ID"].isna().sum())
    corrected_missing_ids = int(corrected["Row ID"].isna().sum())

    original_duplicate_ids = int(
        original["Row ID"].duplicated(keep=False).sum()
    )

    corrected_duplicate_ids = int(
        corrected["Row ID"].duplicated(keep=False).sum()
    )

    original_keyed = (
        original.dropna(subset=["Row ID"])
        .drop_duplicates("Row ID")
        .set_index("Row ID")
    )

    corrected_keyed = (
        corrected.dropna(subset=["Row ID"])
        .drop_duplicates("Row ID")
        .set_index("Row ID")
    )

    common_ids = original_keyed.index.intersection(
        corrected_keyed.index
    )

    # --------------------------------------------------------
    # Compare original and corrected dates by Row ID
    # --------------------------------------------------------

    comparison = pd.DataFrame(index=common_ids)

    for col in DATE_COLS:

        comparison[f"Original {col}"] = (
            original_keyed.loc[common_ids, col]
        )

        comparison[f"Corrected {col}"] = (
            corrected_keyed.loc[common_ids, col]
        )

        comparison[f"{col} Match"] = (
            comparison[f"Original {col}"]
            == comparison[f"Corrected {col}"]
        )

    comparison["Original Calculated Lead Time Days"] = (
        original_keyed.loc[
            common_ids,
            "_calculated_lead_time_days"
        ]
    )

    comparison["Corrected Calculated Lead Time Days"] = (
        corrected_keyed.loc[
            common_ids,
            "_calculated_lead_time_days"
        ]
    )

    comparison["Stored Step 3 Lead Time Days"] = (
        corrected_keyed.loc[common_ids, TARGET]
    )

    comparison["Stored Lead Time Matches Calculated"] = (
        comparison["Stored Step 3 Lead Time Days"]
        == comparison["Corrected Calculated Lead Time Days"]
    )

    order_date_mismatches = int(
        (~comparison["Order Date Match"]).sum()
    )

    ship_date_mismatches = int(
        (~comparison["Ship Date Match"]).sum()
    )

    lead_time_mismatches = int(
        (~comparison["Stored Lead Time Matches Calculated"]).sum()
    )

    # --------------------------------------------------------
    # Missing dates and date parsing failures
    # --------------------------------------------------------

    original_missing_order = int(
        original["Order Date"].isna().sum()
    )

    original_missing_ship = int(
        original["Ship Date"].isna().sum()
    )

    corrected_missing_order = int(
        corrected["Order Date"].isna().sum()
    )

    corrected_missing_ship = int(
        corrected["Ship Date"].isna().sum()
    )

    # --------------------------------------------------------
    # Build row-level quality flags
    # --------------------------------------------------------

    flags = corrected.copy()

    flags["Flag Missing Order Date"] = (
        flags["Order Date"].isna()
    )

    flags["Flag Missing Ship Date"] = (
        flags["Ship Date"].isna()
    )

    flags["Flag Negative Lead Time"] = (
        flags["_calculated_lead_time_days"] < 0
    )

    flags["Flag Zero Lead Time"] = (
        flags["_calculated_lead_time_days"] == 0
    )

    flags["Flag Lead Time Over 365 Days"] = (
        flags["_calculated_lead_time_days"] > 365
    )

    flags["Flag Lead Time Over 730 Days"] = (
        flags["_calculated_lead_time_days"] > 730
    )

    flags["Flag Stored Lead Time Mismatch"] = (
        flags[TARGET] != flags["_calculated_lead_time_days"]
    )

    flags["Flag Missing Row ID"] = (
        flags["Row ID"].isna()
    )

    flags["Flag Duplicate Row ID"] = (
        flags["Row ID"].duplicated(keep=False)
    )

    # Map original Order IDs to corrected records.
    original_order_id_map = (
        original.dropna(subset=["Row ID"])
        .drop_duplicates("Row ID")
        .set_index("Row ID")["Order ID"]
    )

    flags["Order ID"] = flags["Row ID"].map(
        original_order_id_map
    )

    flags["Order ID Year Token"] = (
        flags["Order ID"]
        .astype("string")
        .str.extract(r"-(\d{4})-")[0]
    )

    flags["Order Date Year"] = flags["Order Date"].dt.year

    flags["Flag Order ID Year Differs From Order Date"] = (
        pd.to_numeric(
            flags["Order ID Year Token"],
            errors="coerce"
        )
        != flags["Order Date Year"]
    )

    flag_columns = [
        col for col in flags.columns
        if col.startswith("Flag ")
    ]

    flags["Number of Flags"] = (
        flags[flag_columns]
        .fillna(False)
        .astype(int)
        .sum(axis=1)
    )

    flagged = flags.loc[
        flags["Number of Flags"] > 0
    ].copy()

    # --------------------------------------------------------
    # Shipping lead-time distribution
    # --------------------------------------------------------

    lead = corrected["_calculated_lead_time_days"].dropna()

    distribution = (
        lead.value_counts()
        .sort_index()
        .rename_axis("Calculated Lead Time Days")
        .reset_index(name="Record Count")
    )

    distribution["Percent of Records"] = (
        distribution["Record Count"]
        / max(len(corrected), 1)
        * 100
    )

    # --------------------------------------------------------
    # Order year vs shipping year and lead-time bands
    # --------------------------------------------------------

    corrected["Order Year"] = (
        corrected["Order Date"].dt.year
    )

    corrected["Ship Year"] = (
        corrected["Ship Date"].dt.year
    )

    corrected["Lead Time Band"] = pd.cut(
        corrected["_calculated_lead_time_days"],
        bins=[
            -np.inf,
            0,
            30,
            90,
            365,
            730,
            np.inf
        ],
        labels=[
            "Negative",
            "0-30 days",
            "31-90 days",
            "91-365 days",
            "366-730 days",
            "Over 730 days"
        ],
        right=True
    )

    year_table = (
        corrected.groupby(
            ["Order Year", "Ship Year", "Lead Time Band"],
            dropna=False,
            observed=False
        )
        .size()
        .rename("Record Count")
        .reset_index()
    )

    # --------------------------------------------------------
    # Lead-time summary statistics
    # --------------------------------------------------------

    lead_stats = lead.describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )

    negative_lead_times = int((lead < 0).sum())
    zero_lead_times = int((lead == 0).sum())
    lead_over_365 = int((lead > 365).sum())
    lead_over_730 = int((lead > 730).sum())

    # --------------------------------------------------------
    # Save row-level quality flags
    # --------------------------------------------------------

    preferred_columns = [
        "Row ID",
        "Order ID",
        "Order Date",
        "Ship Date",
        TARGET,
        "_calculated_lead_time_days",
        "Order ID Year Token",
        "Order Date Year",
        *flag_columns,
        "Number of Flags"
    ]

    flagged[preferred_columns].to_csv(
        OUTPUT_DIR / "potential_data_quality_flags.csv",
        index=False
    )

    distribution.to_csv(
        OUTPUT_DIR / "shipping_lead_time_distribution.csv",
        index=False
    )

    year_table.to_csv(
        OUTPUT_DIR / "date_consistency_by_year.csv",
        index=False
    )

    # --------------------------------------------------------
    # Generate the text report
    # --------------------------------------------------------

    report = [
        "STEP 11: DATA QUALITY AND SHIPPING-DATE INVESTIGATION",
        "=" * 72,
        "",
        "INPUTS",
        "-" * 72,
        f"Original dataset: {ORIGINAL_FILE.relative_to(BASE_DIR)}",
        f"Corrected dataset: {CORRECTED_FILE.relative_to(BASE_DIR)}",
        f"Original records: {len(original):,}",
        f"Corrected records: {len(corrected):,}",
        f"Common Row IDs compared: {len(common_ids):,}",
        "",
        "DATE COMPLETENESS",
        "-" * 72,
        f"Original missing Order Date: {original_missing_order:,}",
        f"Original missing Ship Date: {original_missing_ship:,}",
        f"Corrected missing Order Date: {corrected_missing_order:,}",
        f"Corrected missing Ship Date: {corrected_missing_ship:,}",
        "",
        "ROW ID QUALITY",
        "-" * 72,
        f"Original missing Row IDs: {original_missing_ids:,}",
        f"Corrected missing Row IDs: {corrected_missing_ids:,}",
        f"Original duplicate Row ID rows: {original_duplicate_ids:,}",
        f"Corrected duplicate Row ID rows: {corrected_duplicate_ids:,}",
        "",
        "ORIGINAL VS CORRECTED DATA",
        "-" * 72,
        f"Order Date mismatches by Row ID: {order_date_mismatches:,}",
        f"Ship Date mismatches by Row ID: {ship_date_mismatches:,}",
        f"Stored lead-time mismatches: {lead_time_mismatches:,}",
        "",
        "CALCULATED LEAD-TIME STATISTICS (DAYS)",
        "-" * 72,
        lead_stats.to_string(
            float_format=lambda value: f"{value:.2f}"
        ),
        f"Negative lead times: {negative_lead_times:,}",
        f"Zero lead times: {zero_lead_times:,}",
        f"Lead times over 365 days: {lead_over_365:,}",
        f"Lead times over 730 days: {lead_over_730:,}",
        (
            f"Order Date range: {corrected['Order Date'].min()} "
            f"to {corrected['Order Date'].max()}"
        ),
        (
            f"Ship Date range: {corrected['Ship Date'].min()} "
            f"to {corrected['Ship Date'].max()}"
        ),
        "",
        "INTERPRETATION",
        "-" * 72,
        "1. Date formats are parsed separately for the original and corrected files.",
        "2. Date mismatches indicate differences between files, not necessarily errors.",
        "3. Lead times above 365 or 730 days are flagged for investigation.",
        "4. Long shipping durations should be confirmed with the dataset owner.",
        "5. Do not change dates or remove records merely to improve model metrics.",
        "6. Factory optimization conclusions remain provisional until date semantics are confirmed.",
        "",
        "OUTPUT FILES",
        "-" * 72,
        "data/step11_outputs/step11_data_quality_summary.txt",
        "data/step11_outputs/shipping_lead_time_distribution.csv",
        "data/step11_outputs/date_consistency_by_year.csv",
        "data/step11_outputs/potential_data_quality_flags.csv",
        "",
        "Original and corrected source files were not modified."
    ]

    report_path = (
        OUTPUT_DIR / "step11_data_quality_summary.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\nSTEP 11 COMPLETED")
    print("=" * 60)

    print(f"Original records: {len(original):,}")
    print(f"Corrected Step 3 records: {len(corrected):,}")
    print(f"Common Row IDs compared: {len(common_ids):,}")

    print(
        "Missing dates in corrected data: "
        f"{corrected_missing_order + corrected_missing_ship:,}"
    )

    print(f"Order Date mismatches: {order_date_mismatches:,}")
    print(f"Ship Date mismatches: {ship_date_mismatches:,}")
    print(f"Stored lead-time mismatches: {lead_time_mismatches:,}")

    print(f"Lead times over 365 days: {lead_over_365:,}")
    print(f"Lead times over 730 days: {lead_over_730:,}")
    print(f"Rows flagged for review: {len(flagged):,}")

    print("\nOutputs saved to:")
    print(f"  {OUTPUT_DIR}")

    print(
        "\nThis investigation does not modify the original "
        "or corrected source datasets."
    )


if __name__ == "__main__":
    main()