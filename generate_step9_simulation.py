
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 9: NASSAU CANDY FACTORY REALLOCATION SIMULATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

STEP6_DIR = DATA_DIR / "step6_outputs"
STEP8_DIR = DATA_DIR / "step8_outputs"
STEP9_DIR = DATA_DIR / "step9_outputs"

HISTORICAL_FILE = DATA_DIR / "Nassau Candy Distributor.csv"

MAPPING_FILE = DATA_DIR / "product_factory_mapping.csv"

SCENARIO_FILE = (
    STEP6_DIR / "factory_scenario_summary.csv"
)

OUTPUT_FILE = (
    STEP9_DIR / "factory_reallocation_simulation.csv"
)

SUMMARY_FILE = STEP9_DIR / "step9_simulation_summary.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """Find a column without depending on capitalization."""

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def parse_dates(series):
    """
    Parse dates using the format that best matches the data.

    Explicit day-first parsing avoids the default month-first
    interpretation for dates such as 13-02-2025.
    """

    text = series.astype("string").str.strip()

    # First try day-month-year.
    parsed = pd.to_datetime(
        text,
        format="%d-%m-%Y",
        errors="coerce",
    )

    # Support dates containing slashes or timestamps if needed.
    missing = parsed.isna() & text.notna()

    if missing.any():
        fallback = pd.to_datetime(
            text.loc[missing],
            dayfirst=True,
            errors="coerce",
        )

        parsed.loc[missing] = fallback

    return parsed


# ============================================================
# MAIN SIMULATION
# ============================================================

def main():

    print("=" * 65)
    print("NASSAU CANDY - STEP 9 SIMULATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check files and create output directory
    # --------------------------------------------------------

    STEP9_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(
            f"Historical data not found:\n{HISTORICAL_FILE}"
        )

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Factory mapping not found:\n{MAPPING_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load historical data
    # --------------------------------------------------------

    df = pd.read_csv(HISTORICAL_FILE)

    mapping = pd.read_csv(MAPPING_FILE)

    print(f"\nHistorical records: {len(df):,}")
    print(f"Historical columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 3. Identify historical data columns
    # --------------------------------------------------------

    product_col = find_column(
        df,
        ["Product ID", "Product_ID", "product_id"],
    )

    order_date_col = find_column(
        df,
        ["Order Date", "Order_Date", "order_date"],
    )

    ship_date_col = find_column(
        df,
        ["Ship Date", "Ship_Date", "ship_date"],
    )

    region_col = find_column(
        df,
        ["Region", "Sales Region", "region"],
    )

    sales_col = find_column(
        df,
        ["Sales", "Revenue", "sales"],
    )

    units_col = find_column(
        df,
        ["Units", "Quantity", "units", "quantity"],
    )

    order_id_col = find_column(
        df,
        ["Order ID", "Order_ID", "order_id"],
    )

    shipping_mode_col = find_column(
        df,
        ["Ship Mode", "Shipping Mode", "ship_mode"],
    )

    if product_col is None:
        raise ValueError(
            "Product ID column was not found."
        )

    if order_date_col is None or ship_date_col is None:
        raise ValueError(
            "Order Date or Ship Date column was not found."
        )

    # --------------------------------------------------------
    # 4. Calculate historical shipping lead time
    # --------------------------------------------------------

    df[order_date_col] = parse_dates(
        df[order_date_col]
    )

    df[ship_date_col] = parse_dates(
        df[ship_date_col]
    )

    df["Historical Lead Time Days"] = (
        df[ship_date_col] - df[order_date_col]
    ).dt.total_seconds() / 86400

    # Invalid dates and negative shipping durations are rejected.
    invalid_dates = (
        df[order_date_col].isna()
        | df[ship_date_col].isna()
        | (df["Historical Lead Time Days"] < 0)
    )

    df.loc[
        invalid_dates,
        "Historical Lead Time Days",
    ] = np.nan

    valid_lead_times = df[
        "Historical Lead Time Days"
    ].dropna()

    print("\n--- DATE VALIDATION ---")

    print(
        "Valid order dates:",
        int(df[order_date_col].notna().sum()),
    )

    print(
        "Valid shipping dates:",
        int(df[ship_date_col].notna().sum()),
    )

    print(
        "Valid lead-time records:",
        len(valid_lead_times),
    )

    if len(valid_lead_times) > 0:

        print(
            "Overall average lead time:",
            round(valid_lead_times.mean(), 2),
            "days",
        )

        print(
            "Overall median lead time:",
            round(valid_lead_times.median(), 2),
            "days",
        )

    # --------------------------------------------------------
    # 5. Prepare the factory mapping
    # --------------------------------------------------------

    map_product_col = find_column(
        mapping,
        ["Product ID", "Product_ID", "product_id"],
    )

    map_factory_col = find_column(
        mapping,
        [
            "Current Factory",
            "Factory",
            "current_factory",
        ],
    )

    if map_product_col is None or map_factory_col is None:
        raise ValueError(
            "The mapping file must contain Product ID "
            "and Current Factory columns."
        )

    mapping = mapping[
        [map_product_col, map_factory_col]
    ].copy()

    mapping.columns = [
        "Product ID",
        "Current Factory",
    ]

    mapping["Product ID"] = (
        mapping["Product ID"]
        .astype("string")
        .str.strip()
    )

    mapping["Current Factory"] = (
        mapping["Current Factory"]
        .astype("string")
        .str.strip()
    )

    mapping = mapping.dropna(
        subset=["Product ID", "Current Factory"]
    )

    if mapping["Product ID"].duplicated().any():

        duplicates = mapping.loc[
            mapping["Product ID"].duplicated(keep=False),
            "Product ID",
        ].unique()

        raise ValueError(
            "Duplicate Product IDs exist in the factory mapping: "
            + ", ".join(duplicates.astype(str))
        )

    factories = sorted(
        mapping["Current Factory"].unique().tolist()
    )

    print("\n--- FACTORY MAPPING ---")

    print("Factories found:", len(factories))

    for factory in factories:
        print(" -", factory)

    # --------------------------------------------------------
    # 6. Attach current factory assignments
    # --------------------------------------------------------

    df[product_col] = (
        df[product_col]
        .astype("string")
        .str.strip()
    )

    df = df.merge(
        mapping,
        left_on=product_col,
        right_on="Product ID",
        how="left",
        validate="many_to_one",
    )

    df["Factory Mapping Status"] = np.where(
        df["Current Factory"].notna(),
        "Mapped",
        "Unmapped",
    )

    # --------------------------------------------------------
    # 7. Convert numeric fields
    # --------------------------------------------------------

    if sales_col is not None:

        df[sales_col] = pd.to_numeric(
            df[sales_col],
            errors="coerce",
        )

    if units_col is not None:

        df[units_col] = pd.to_numeric(
            df[units_col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 8. Aggregate historical product-region metrics
    # --------------------------------------------------------

    grouping_columns = [product_col]

    if region_col is not None:
        grouping_columns.append(region_col)

    aggregation = {
        "Historical Records": (
            product_col,
            "size",
        ),

        "Current Factory": (
            "Current Factory",
            "first",
        ),

        "Factory Mapping Status": (
            "Factory Mapping Status",
            lambda values: (
                "Mapped"
                if values.eq("Mapped").all()
                else "Unmapped"
            ),
        ),

        "Historical Average Lead Time Days": (
            "Historical Lead Time Days",
            "mean",
        ),

        "Historical Median Lead Time Days": (
            "Historical Lead Time Days",
            "median",
        ),

        "Historical Lead Time Records": (
            "Historical Lead Time Days",
            "count",
        ),
    }

    if sales_col is not None:

        aggregation["Historical Sales"] = (
            sales_col,
            "sum",
        )

    if units_col is not None:

        aggregation["Historical Units"] = (
            units_col,
            "sum",
        )

    if order_id_col is not None:

        aggregation["Historical Orders"] = (
            order_id_col,
            "nunique",
        )

    if shipping_mode_col is not None:

        aggregation["Observed Shipping Modes"] = (
            shipping_mode_col,
            lambda values: ", ".join(
                sorted(
                    values.dropna()
                    .astype(str)
                    .unique()
                )
            ),
        )

    result = (
        df.groupby(
            grouping_columns,
            dropna=False,
        )
        .agg(**aggregation)
        .reset_index()
    )

    result = result.rename(
        columns={
            product_col: "Product ID",
        }
    )

    if region_col is not None:

        result = result.rename(
            columns={
                region_col: "Region",
            }
        )

    else:

        result["Region"] = "All Regions"

    # --------------------------------------------------------
    # 9. Create baseline scenarios
    # --------------------------------------------------------

    baseline = result.copy()

    baseline["Proposed Factory"] = (
        baseline["Current Factory"]
    )

    baseline["Scenario Type"] = "CURRENT_BASELINE"

    baseline["Reallocation Status"] = (
        "Current assignment; historical observations"
    )

    baseline["Factory-Specific Prediction Available"] = False

    baseline["Incremental Cost Available"] = False

    baseline["Capacity Feasibility Verified"] = False

    baseline["Estimated Lead Time Improvement Days"] = np.nan

    baseline["Estimated Cost Savings"] = np.nan

    # --------------------------------------------------------
    # 10. Create alternative factory candidates
    # --------------------------------------------------------

    alternative_rows = []

    mapped = result.loc[
        result["Current Factory"].notna()
    ].copy()

    for factory in factories:

        alternative = mapped.loc[
            mapped["Current Factory"] != factory
        ].copy()

        if alternative.empty:
            continue

        alternative["Proposed Factory"] = factory

        alternative["Scenario Type"] = (
            "ALTERNATIVE_CANDIDATE"
        )

        alternative["Reallocation Status"] = (
            "UNVALIDATED: factory-specific predictions, "
            "capacity and incremental costs unavailable"
        )

        alternative[
            "Factory-Specific Prediction Available"
        ] = False

        alternative[
            "Incremental Cost Available"
        ] = False

        alternative[
            "Capacity Feasibility Verified"
        ] = False

        alternative[
            "Estimated Lead Time Improvement Days"
        ] = np.nan

        alternative[
            "Estimated Cost Savings"
        ] = np.nan

        alternative_rows.append(alternative)

    # --------------------------------------------------------
    # 11. Combine scenarios
    # --------------------------------------------------------

    if alternative_rows:

        output = pd.concat(
            [baseline] + alternative_rows,
            ignore_index=True,
        )

    else:

        output = baseline.copy()

    output["Simulation Evidence"] = (
        "Historical values represent observed operations. "
        "Alternative assignments are candidates only; "
        "no alternative factory performance has been predicted."
    )

    # --------------------------------------------------------
    # 12. Save simulation output
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 13. Create a text summary
    # --------------------------------------------------------

    summary_lines = [
        "NASSAU CANDY - STEP 9 SIMULATION SUMMARY",
        "=" * 50,
        f"Historical records loaded: {len(df):,}",
        f"Product-region baseline rows: {len(baseline):,}",
        f"Total simulation rows: {len(output):,}",
        f"Factories found: {len(factories)}",
        f"Valid lead-time records: {len(valid_lead_times):,}",
        "",
        "Factories:",
    ]

    summary_lines.extend(
        [f"- {factory}" for factory in factories]
    )

    summary_lines.extend(
        [
            "",
            "Scenario counts:",
            output["Scenario Type"]
            .value_counts()
            .to_string(),
            "",
            "Limitations:",
            "- Alternative factory assignments are not validated.",
            "- Factory-specific lead-time predictions are unavailable.",
            "- Incremental scenario costs are unavailable.",
            "- Factory capacity constraints are unavailable.",
            "- Cost savings and lead-time improvements are not estimated.",
            "",
            f"Output file: {OUTPUT_FILE}",
        ]
    )

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 14. Print completion message
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 9 SIMULATION COMPLETED")
    print("=" * 65)

    print(f"\nOutput CSV:\n{OUTPUT_FILE}")

    print(f"\nSummary:\n{SUMMARY_FILE}")

    print(f"\nSimulation rows: {len(output):,}")

    print("\nScenario counts:")

    print(
        output["Scenario Type"]
        .value_counts()
        .to_string()
    )

    print("\nMapping status:")

    print(
        output["Factory Mapping Status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nIMPORTANT: Alternative assignments are candidates, "
        "not optimized recommendations."
    )


if __name__ == "__main__":
    main()

