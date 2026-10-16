
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="ðŸ¬",
    layout="wide",
)

# Set BASE_DIR to the root directory (adjust if app.py is in a subfolder)
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

STEP8_DIR = DATA_DIR / "step8_outputs"
STEP9_DIR = DATA_DIR / "step9_outputs"
HISTORICAL_FILE = DATA_DIR / "Nassau Candy Distributor.csv"

ENGINEERED_FILE = (
    DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv"
)

SIMULATION_FILE = STEP9_DIR / "factory_reallocation_simulation.csv"
RECOMMENDATION_FILE = STEP9_DIR / "factory_reallocation_simulation.csv"
KPI_FILE = STEP9_DIR / "optimization_kpis.csv"
MODEL_FILE = STEP9_DIR / "model_comparison.csv"

READINESS_FILE = STEP8_DIR / "optimization_readiness_report.csv"
INPUT_STATUS_FILE = STEP8_DIR / "optimization_input_status.csv"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(file_path):
    """Load a CSV file if it exists."""
    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception as exc:
        st.error(f"Could not load {file_path.name}: {exc}")
        return None


def find_column(df, candidates):
    """Find a column using case-insensitive matching."""
    if df is None:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        match = lookup.get(candidate.strip().lower())

        if match is not None:
            return match

    return None


def numeric_series(df, candidates):
    """Return a numeric series if a matching column exists."""
    column = find_column(df, candidates)

    if column is None:
        return None

    return pd.to_numeric(df[column], errors="coerce")


def display_metric(label, value, help_text=None):
    """Display a metric without assuming missing values are zero."""
    st.metric(
        label=label,
        value=value,
        help=help_text,
    )


def money(value):
    """Format monetary values in US dollars."""
    if pd.isna(value):
        return "N/A"

    return f"${value:,.2f}"


def safe_mean(series):
    """Calculate a mean without treating missing observations as zero."""
    if series is None:
        return None

    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.mean())


# ============================================================
# 3. LOAD PROJECT DATA
# ============================================================

historical = load_csv(HISTORICAL_FILE)
engineered = load_csv(ENGINEERED_FILE)

simulation = load_csv(SIMULATION_FILE)
recommendations = load_csv(RECOMMENDATION_FILE)
optimization_kpis = load_csv(KPI_FILE)
model_comparison = load_csv(MODEL_FILE)

readiness = load_csv(READINESS_FILE)
input_status = load_csv(INPUT_STATUS_FILE)


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.title("ðŸ¬ Nassau Candy")

st.sidebar.caption(
    "Factory Optimization & Demand Planning System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Executive Overview",
        "Factory Reallocation Simulator",
        "Demand Planning & What-If",
        "Factory Recommendations",
        "Risk & Data Quality",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Recommendations are subject to the availability "
    "and validation of the underlying data."
)


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title("ðŸ¬ Nassau Candy Factory Optimization Dashboard")

st.markdown(
    """
    Analyze historical sales and shipping performance, explore
    alternative factory assignments, and evaluate demand-planning
    scenarios using the project's available data and model outputs.
    """
)

st.info(
    "Historical results describe recorded data. Simulations and "
    "recommendations describe hypothetical outcomes unless "
    "independently validated."
)


# ============================================================
# 6. EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    if historical is None:
        st.error(
            "The historical dataset was not found. "
            f"Expected location: {HISTORICAL_FILE}"
        )
        st.stop()

    sales = numeric_series(historical, ["Sales"])
    profit = numeric_series(
        historical,
        ["Gross Profit", "Gross Profit ($)", "Profit"],
    )
    units = numeric_series(
        historical,
        ["Units", "Quantity", "Units Sold"],
    )
    order_id = find_column(historical, ["Order ID"])
    product_id = find_column(
        historical,
        ["Product ID", "Product Name"],
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hist_len = f"{len(historical):,}" if historical is not None else "0"
        display_metric("Historical Records", hist_len)

    with col2:
        display_metric(
            "Total Sales",
            money(sales.sum()) if sales is not None else "N/A",
        )

    with col3:
        display_metric(
            "Gross Profit",
            money(profit.sum()) if profit is not None else "N/A",
        )

    with col4:
        display_metric(
            "Units Sold",
            f"{units.sum():,.0f}"
            if units is not None and units.notna().any()
            else "N/A",
        )

    st.markdown("---")

    if sales is not None:

        chart_data = historical.copy()
        chart_data["_sales"] = sales

        region_col = find_column(
            chart_data,
            ["Region"],
        )

        if region_col is not None:

            regional_sales = (
                chart_data.groupby(region_col, dropna=False)["_sales"]
                .sum()
                .reset_index()
                .sort_values("_sales", ascending=False)
            )

            fig = px.bar(
                regional_sales,
                x=region_col,
                y="_sales",
                title="Historical Sales by Region",
                labels={
                    region_col: "Region",
                    "_sales": "Sales",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        date_col = find_column(
            historical,
            ["Order Date"],
        )

        if date_col is not None:

            chart_data["_date"] = pd.to_datetime(
                chart_data[date_col],
                errors="coerce",
            )

            chart_data = chart_data.dropna(subset=["_date"])

            if not chart_data.empty:

                chart_data["_month"] = (
                    chart_data["_date"].dt.to_period("M").astype(str)
                )

                monthly = (
                    chart_data.groupby("_month")["_sales"]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    monthly,
                    x="_month",
                    y="_sales",
                    markers=True,
                    title="Historical Monthly Sales",
                    labels={
                        "_month": "Month",
                        "_sales": "Sales",
                    },
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    st.subheader("Dataset Preview")

    st.dataframe(
        historical.head(20),
        use_container_width=True,
    )


# ============================================================
# 7. FACTORY REALLOCATION SIMULATOR
# ============================================================

elif page == "Factory Reallocation Simulator":

    st.header("Factory Reallocation Simulator")

    if simulation is None:
        st.warning(
            "The Step 9 simulation file is not available yet."
        )

        st.code(str(SIMULATION_FILE))

        st.info(
            "Generate the simulation output using the existing "
            "project workflow before evaluating reallocation scenarios."
        )

    else:

        st.subheader("Simulation Output")

        st.caption(
            f"Loaded {len(simulation):,} simulation records."
        )

        st.dataframe(
            simulation.head(100),
            use_container_width=True,
        )

        st.markdown("---")

        product_col = find_column(
            simulation,
            ["Product Name", "Product ID"],
        )

        region_col = find_column(
            simulation,
            ["Region"],
        )

        ship_col = find_column(
            simulation,
            ["Ship Mode"],
        )

        filters = st.columns(3)

        filtered = simulation.copy()

        with filters[0]:

            if product_col is not None:

                options = sorted(
                    filtered[product_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected = st.selectbox(
                    "Product",
                    ["All"] + options,
                )

                if selected != "All":

                    filtered = filtered[
                        filtered[product_col].astype(str) == selected
                    ]

        with filters[1]:

            if region_col is not None:

                options = sorted(
                    filtered[region_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected = st.selectbox(
                    "Region",
                    ["All"] + options,
                )

                if selected != "All":

                    filtered = filtered[
                        filtered[region_col].astype(str) == selected
                    ]

        with filters[2]:

            if ship_col is not None:

                options = sorted(
                    filtered[ship_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected = st.selectbox(
                    "Ship Mode",
                    ["All"] + options,
                )

                if selected != "All":

                    filtered = filtered[
                        filtered[ship_col].astype(str) == selected
                    ]

        st.subheader("Filtered Simulation Results")

        st.dataframe(
            filtered,
            use_container_width=True,
        )

        st.download_button(
            "Download Filtered Simulation",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_factory_simulation.csv",
            mime="text/csv",
        )


# ============================================================
# 8. DEMAND PLANNING & WHAT-IF
# ============================================================

elif page == "Demand Planning & What-If":

    st.header("Demand Planning & What-If Analysis")

    if engineered is None:

        st.warning(
            "The engineered dataset was not found."
        )

        st.code(str(ENGINEERED_FILE))

    else:

        st.caption(
            "Explore historical units and hypothetical demand changes. "
            "These scenarios are not model forecasts unless a validated "
            "forecasting model and its outputs are connected."
        )

        units = numeric_series(
            engineered,
            ["Units", "Quantity", "Units Sold"],
        )

        if units is None or units.dropna().empty:

            st.warning(
                "A valid units column could not be identified."
            )

        else:

            valid_units = units.dropna()

            baseline_units = float(valid_units.sum())

            st.metric(
                "Historical Units in Selected Dataset",
                f"{baseline_units:,.0f}",
            )

            demand_change = st.slider(
                "Hypothetical Demand Change (%)",
                min_value=-50,
                max_value=100,
                value=10,
                step=5,
            )

            scenario_units = (
                baseline_units * (1 + demand_change / 100)
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Baseline Units",
                    f"{baseline_units:,.0f}",
                )

            with col2:

                st.metric(
                    "Hypothetical Scenario Units",
                    f"{scenario_units:,.0f}",
                    delta=f"{demand_change:+d}%",
                )

            st.warning(
                "This is a mathematical demand scenario, not a "
                "forecast. No production capacity, inventory availability, "
                "or replenishment feasibility is assumed."
            )

            scenario_table = pd.DataFrame(
                {
                    "Scenario": [
                        "Historical baseline",
                        "Hypothetical demand scenario",
                    ],
                    "Demand Change (%)": [
                        0,
                        demand_change,
                    ],
                    "Units": [
                        baseline_units,
                        scenario_units,
                    ],
                }
            )

            st.dataframe(
                scenario_table,
                use_container_width=True,
            )

            st.download_button(
                "Download Demand Scenario",
                data=scenario_table.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="demand_what_if_scenario.csv",
                mime="text/csv",
            )


# ============================================================
# 9. FACTORY RECOMMENDATIONS
# ============================================================

elif page == "Factory Recommendations":

    st.header("Factory Recommendation Dashboard")

    if recommendations is None:

        st.warning(
            "The Step 9 factory recommendation file is unavailable."
        )

        st.code(str(RECOMMENDATION_FILE))

    else:

        st.caption(
            "Review the generated recommendations and their available "
            "supporting evidence."
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
        )

        st.download_button(
            "Download Factory Recommendations",
            data=recommendations.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="factory_recommendations.csv",
            mime="text/csv",
        )

    st.markdown("---")

    if optimization_kpis is not None:

        st.subheader("Optimization KPIs")

        st.dataframe(
            optimization_kpis,
            use_container_width=True,
        )

    if model_comparison is not None:

        st.subheader("Model Comparison")

        st.dataframe(
            model_comparison,
            use_container_width=True,
        )


# ============================================================
# 10. RISK & DATA QUALITY
# ============================================================

elif page == "Risk & Data Quality":

    st.header("Risk & Data Quality")

    st.subheader("Dataset Availability")

    file_status = pd.DataFrame(
        [
            {
                "Dataset": "Historical Sales",
                "Path": str(HISTORICAL_FILE),
                "Available": HISTORICAL_FILE.exists(),
            },
            {
                "Dataset": "Feature-Engineered Data",
                "Path": str(ENGINEERED_FILE),
                "Available": ENGINEERED_FILE.exists(),
            },
            {
                "Dataset": "Step 8 Optimization Readiness",
                "Path": str(READINESS_FILE),
                "Available": READINESS_FILE.exists(),
            },
            {
                "Dataset": "Step 9 Simulation",
                "Path": str(SIMULATION_FILE),
                "Available": SIMULATION_FILE.exists(),
            },
            {
                "Dataset": "Step 9 Recommendations",
                "Path": str(RECOMMENDATION_FILE),
                "Available": RECOMMENDATION_FILE.exists(),
            },
            {
                "Dataset": "Step 9 Optimization KPIs",
                "Path": str(KPI_FILE),
                "Available": KPI_FILE.exists(),
            },
            {
                "Dataset": "Step 9 Model Comparison",
                "Path": str(MODEL_FILE),
                "Available": MODEL_FILE.exists(),
            },
        ]
    )

    st.dataframe(
        file_status,
        width='stretch',
    )

    if historical is not None:

        st.subheader("Historical Data Quality")

        quality_metrics = pd.DataFrame(
            {
                "Metric": [
                    "Row Count",
                    "Column Count",
                    "Duplicate Rows",
                    "Total Missing Cells",
                ],
                "Value": [
                    len(historical),
                    len(historical.columns),
                    int(historical.duplicated().sum()),
                    int(historical.isna().sum().sum()),
                ],
            }
        )

        st.dataframe(
            quality_metrics,
            width= 'stretch',
        )

        missing = (
            historical.isna()
            .sum()
            .rename("Missing Values")
            .reset_index()
            .rename(columns={"index": "Column"})
        )

        missing = missing[
            missing["Missing Values"] > 0
        ].sort_values(
            "Missing Values",
            ascending=False,
        )

        if not missing.empty:

            st.subheader("Missing Values by Column")

            st.dataframe(
                missing,
                width='stretch',
            )

    if readiness is not None:

        st.subheader("Step 8 Optimization Readiness Report")

        st.dataframe(
            readiness,
            width='stretch',
        )

    if input_status is not None:

        st.subheader("Step 8 Optimization Input Status")

        st.dataframe(
            input_status,
            width='stretch',
        )

    st.markdown("---")

    st.subheader("Interpretation and Limitations")

    st.markdown(
        """
        - Factory assignments must be supported by the project's
          product-to-factory mapping.
        - Predicted lead times are estimates, not guaranteed delivery times.
        - Transportation savings require validated route and cost data.
        - Capacity-constrained production recommendations require actual
          factory capacity and production data.
        - Profit-impact estimates require an explicit, validated
          relationship between the proposed change and profitability.
        - Demand scenarios must not be presented as validated forecasts.
        """
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Nassau Candy Factory Optimization & Demand Planning System | "
    "Analytical decision support â€” validate recommendations before use."
)

