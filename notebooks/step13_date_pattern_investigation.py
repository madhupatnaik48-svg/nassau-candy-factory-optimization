from pathlib import Path

import pandas as pd


# ============================================================
# STEP 13: DATE PATTERN INVESTIGATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "step13_outputs"

ORIGINAL_FILE = DATA_DIR / "Nassau Candy Distributor.csv"
CORRECTED_FILE = (
    DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_original_dates(series):
    """Parse original dates stored in DD-MM-YYYY format."""
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )


def parse_corrected_dates(series):
    """Parse corrected dates stored in YYYY-MM-DD format."""
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="%Y-%m-%d",
        errors="coerce",
    )


def main():
    print("=" * 70)
    print("STEP 13: DATE PATTERN INVESTIGATION")
    print("=" * 70)

    original = pd.read_csv(ORIGINAL_FILE)
    corrected = pd.read_csv(CORRECTED_FILE)

    # Parse original dates using their original format.
    original["Order Date"] = parse_original_dates(
        original["Order Date"]
    )

    original["Ship Date"] = parse_original_dates(
        original["Ship Date"]
    )

    # Parse corrected dates using their ISO format.
    corrected["Order Date"] = parse_corrected_dates(
        corrected["Order Date"]
    )

    corrected["Ship Date"] = parse_corrected_dates(
        corrected["Ship Date"]
    )

    # Calculate lead times independently.
    original["Calculated Lead Time Days"] = (
        original["Ship Date"] - original["Order Date"]
    ).dt.days

    corrected["Calculated Lead Time Days"] = (
        corrected["Ship Date"] - corrected["Order Date"]
    ).dt.days

    # Add date components for investigation.
    for df in [original, corrected]:
        df["Order Year Investigated"] = df["Order Date"].dt.year
        df["Ship Year Investigated"] = df["Ship Date"].dt.year
        df["Order Month Investigated"] = df["Order Date"].dt.month

    # --------------------------------------------------------
    # 1. Overall summary
    # --------------------------------------------------------
    summary_rows = []

    for dataset_name, df in [
        ("Original", original),
        ("Corrected Step 3", corrected),
    ]:
        lead = df["Calculated Lead Time Days"].dropna()

        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Total Records": len(df),
                "Missing Order Dates": df["Order Date"].isna().sum(),
                "Missing Ship Dates": df["Ship Date"].isna().sum(),
                "Minimum Order Date": df["Order Date"].min(),
                "Maximum Order Date": df["Order Date"].max(),
                "Minimum Ship Date": df["Ship Date"].min(),
                "Maximum Ship Date": df["Ship Date"].max(),
                "Minimum Lead Time Days": lead.min(),
                "Median Lead Time Days": lead.median(),
                "Mean Lead Time Days": round(lead.mean(), 2),
                "Maximum Lead Time Days": lead.max(),
                "Records Over 730 Days": (lead > 730).sum(),
            }
        )

    overall_summary = pd.DataFrame(summary_rows)

    overall_summary.to_csv(
        OUTPUT_DIR / "date_pattern_overall_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Lead-time distribution
    # --------------------------------------------------------
    distribution_rows = []

    for dataset_name, df in [
        ("Original", original),
        ("Corrected Step 3", corrected),
    ]:
        lead = df["Calculated Lead Time Days"]

        distribution_rows.append(
            {
                "Dataset": dataset_name,
                "Records Under 0 Days": (lead < 0).sum(),
                "Records At 0 Days": (lead == 0).sum(),
                "Records 1-30 Days": lead.between(1, 30).sum(),
                "Records 31-90 Days": lead.between(31, 90).sum(),
                "Records 91-180 Days": lead.between(91, 180).sum(),
                "Records 181-365 Days": lead.between(181, 365).sum(),
                "Records 366-730 Days": lead.between(366, 730).sum(),
                "Records Over 730 Days": (lead > 730).sum(),
                "Missing Lead Times": lead.isna().sum(),
            }
        )

    pd.DataFrame(distribution_rows).to_csv(
        OUTPUT_DIR / "lead_time_bucket_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Shipping mode analysis
    # --------------------------------------------------------
    shipping_mode_report = (
        original.groupby("Ship Mode", dropna=False)
        .agg(
            Record_Count=("Calculated Lead Time Days", "size"),
            Minimum_Lead_Time=("Calculated Lead Time Days", "min"),
            Median_Lead_Time=("Calculated Lead Time Days", "median"),
            Mean_Lead_Time=("Calculated Lead Time Days", "mean"),
            Maximum_Lead_Time=("Calculated Lead Time Days", "max"),
            Records_Over_730_Days=(
                "Calculated Lead Time Days",
                lambda x: (x > 730).sum(),
            ),
        )
        .reset_index()
    )

    shipping_mode_report["Mean_Lead_Time"] = (
        shipping_mode_report["Mean_Lead_Time"].round(2)
    )

    shipping_mode_report.to_csv(
        OUTPUT_DIR / "lead_time_by_shipping_mode.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Order year and ship year analysis
    # --------------------------------------------------------
    order_year_report = (
        original.groupby("Order Year Investigated", dropna=False)
        .agg(
            Record_Count=("Row ID", "size"),
            Minimum_Lead_Time=(
                "Calculated Lead Time Days",
                "min",
            ),
            Median_Lead_Time=(
                "Calculated Lead Time Days",
                "median",
            ),
            Mean_Lead_Time=(
                "Calculated Lead Time Days",
                "mean",
            ),
            Maximum_Lead_Time=(
                "Calculated Lead Time Days",
                "max",
            ),
        )
        .reset_index()
    )

    order_year_report["Mean_Lead_Time"] = (
        order_year_report["Mean_Lead_Time"].round(2)
    )

    order_year_report.to_csv(
        OUTPUT_DIR / "lead_time_by_order_year.csv",
        index=False,
    )

    ship_year_report = (
        original.groupby("Ship Year Investigated", dropna=False)
        .agg(
            Record_Count=("Row ID", "size"),
            Minimum_Lead_Time=(
                "Calculated Lead Time Days",
                "min",
            ),
            Median_Lead_Time=(
                "Calculated Lead Time Days",
                "median",
            ),
            Mean_Lead_Time=(
                "Calculated Lead Time Days",
                "mean",
            ),
            Maximum_Lead_Time=(
                "Calculated Lead Time Days",
                "max",
            ),
        )
        .reset_index()
    )

    ship_year_report["Mean_Lead_Time"] = (
        ship_year_report["Mean_Lead_Time"].round(2)
    )

    ship_year_report.to_csv(
        OUTPUT_DIR / "lead_time_by_ship_year.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Monthly order analysis
    # --------------------------------------------------------
    monthly_report = (
        original.groupby("Order Month Investigated", dropna=False)
        .agg(
            Record_Count=("Row ID", "size"),
            Median_Lead_Time=(
                "Calculated Lead Time Days",
                "median",
            ),
            Mean_Lead_Time=(
                "Calculated Lead Time Days",
                "mean",
            ),
            Maximum_Lead_Time=(
                "Calculated Lead Time Days",
                "max",
            ),
        )
        .reset_index()
    )

    monthly_report["Mean_Lead_Time"] = (
        monthly_report["Mean_Lead_Time"].round(2)
    )

    monthly_report.to_csv(
        OUTPUT_DIR / "lead_time_by_order_month.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Investigate identical and repeated lead times
    # --------------------------------------------------------
    repeated_lead_times = (
        original["Calculated Lead Time Days"]
        .value_counts(dropna=False)
        .rename_axis("Lead_Time_Days")
        .reset_index(name="Record_Count")
        .sort_values("Record_Count", ascending=False)
    )

    repeated_lead_times.to_csv(
        OUTPUT_DIR / "repeated_lead_time_values.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Save row-level investigation file
    # --------------------------------------------------------
    row_report = original[
        [
            "Row ID",
            "Order ID",
            "Order Date",
            "Ship Date",
            "Ship Mode",
            "City",
            "State/Province",
            "Region",
            "Calculated Lead Time Days",
            "Order Year Investigated",
            "Ship Year Investigated",
        ]
    ].copy()

    row_report["Investigation Flag"] = "Review lead time"

    row_report.loc[
        row_report["Calculated Lead Time Days"].isna(),
        "Investigation Flag",
    ] = "Missing or unparseable date"

    row_report.loc[
        row_report["Calculated Lead Time Days"] < 0,
        "Investigation Flag",
    ] = "Ship date precedes order date"

    row_report.loc[
        row_report["Calculated Lead Time Days"].between(0, 730),
        "Investigation Flag",
    ] = "Lead time is 730 days or less"

    row_report.to_csv(
        OUTPUT_DIR / "row_level_date_pattern_investigation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Print important findings
    # --------------------------------------------------------
    print("\nOVERALL SUMMARY")
    print("-" * 70)
    print(overall_summary.to_string(index=False))

    print("\nLEAD TIME BY SHIPPING MODE")
    print("-" * 70)
    print(shipping_mode_report.to_string(index=False))

    print("\nLEAD TIME BY ORDER YEAR")
    print("-" * 70)
    print(order_year_report.to_string(index=False))

    print("\nLEAD TIME BY SHIP YEAR")
    print("-" * 70)
    print(ship_year_report.to_string(index=False))

    print("\nTOP 15 REPEATED LEAD-TIME VALUES")
    print("-" * 70)
    print(repeated_lead_times.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 13 COMPLETED")
    print("=" * 70)

    print(f"\nReports saved to:\n{OUTPUT_DIR}")

    print("\nGenerated reports:")
    for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f" - {file_path.name}")


if __name__ == "__main__":
    main()