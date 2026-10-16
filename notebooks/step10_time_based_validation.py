"""
Step 10: Time-Based Model Validation & Performance Analysis
Project: Nassau Candy Factory Optimization

Run this script from the project root folder:
    python step10_time_based_validation.py

Input:
    data/Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv

Outputs:
    data/step10_outputs/time_based_model_comparison.csv
    data/step10_outputs/time_based_test_predictions.csv
    data/step10_outputs/step10_validation_summary.txt

This script does not overwrite app.py, the saved model, or Step 4-Step 9 outputs.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "Nassau_Candy_Feature_Engineered_Step3_CORRECTED.csv"
OUTPUT_DIR = DATA_DIR / "step10_outputs"

TARGET = "Shipping Lead Time Days"
DATE_COL = "Order Date"

NUMERIC_FEATURES = [
    "Units",
    "Order Year",
    "Order Month",
    "Order Quarter",
    "Order Day of Week",
    "Order Day of Month",
]
CATEGORICAL_FEATURES = ["Region", "Ship Mode", "Division", "Product ID"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def parse_order_dates(series):
    """Parse day-first dates safely; support already-parsed datetime values too."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    # The source file uses DD-MM-YYYY. dayfirst=True also handles ISO strings safely.
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def build_pipeline():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        loss="squared_error",
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def metric_row(name, y_true, y_pred, split_name, train_count, test_count):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "Validation": split_name,
        "Model": name,
        "Training Records": int(train_count),
        "Test Records": int(test_count),
        "MAE (days)": float(mean_absolute_error(y_true, y_pred)),
        "RMSE (days)": rmse,
        "R²": float(r2_score(y_true, y_pred)),
    }


def main():
    if not INPUT_FILE.exists():
        print("ERROR: Corrected Step 3 file was not found:")
        print(f"  {INPUT_FILE}")
        print("\nMake sure this script is saved in your project root folder and the corrected CSV exists.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    required = [DATE_COL, TARGET, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        print("ERROR: Required columns are missing from the input file:")
        for column in missing:
            print(f"  - {column}")
        print("\nAvailable columns:")
        print(", ".join(df.columns.astype(str)))
        sys.exit(1)

    df[DATE_COL] = parse_order_dates(df[DATE_COL])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    original_rows = len(df)
    df = df.dropna(subset=[DATE_COL, TARGET]).copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < 20:
        print(f"ERROR: Only {len(df)} valid rows remain; not enough for time-based validation.")
        sys.exit(1)

    # Use a chronological 80/20 split, keeping identical order dates together.
    unique_dates = np.sort(df[DATE_COL].dropna().unique())
    split_index = int(np.floor(len(unique_dates) * 0.80))
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[split_index])

    train_df = df[df[DATE_COL] < cutoff_date].copy()
    test_df = df[df[DATE_COL] >= cutoff_date].copy()

    # Guard against empty partitions or a pathological date distribution.
    if train_df.empty or test_df.empty:
        row_split = int(np.floor(len(df) * 0.80))
        row_split = min(max(row_split, 1), len(df) - 1)
        cutoff_date = pd.Timestamp(df.iloc[row_split][DATE_COL])
        train_df = df[df[DATE_COL] < cutoff_date].copy()
        test_df = df[df[DATE_COL] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        print("ERROR: Could not create a non-empty chronological train/test split.")
        sys.exit(1)

    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].astype(float)
    X_test = test_df[FEATURES].copy()
    y_test = test_df[TARGET].astype(float)

    pipeline = build_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        pipeline.fit(X_train, y_train)

    model_predictions = pipeline.predict(X_test)
    baseline_value = float(y_train.mean())
    baseline_predictions = np.full(len(y_test), baseline_value, dtype=float)

    comparison = pd.DataFrame(
        [
            metric_row(
                "Gradient Boosting Regressor",
                y_test,
                model_predictions,
                "Chronological 80/20",
                len(train_df),
                len(test_df),
            ),
            metric_row(
                "Training-Mean Baseline",
                y_test,
                baseline_predictions,
                "Chronological 80/20",
                len(train_df),
                len(test_df),
            ),
        ]
    )

    predictions = test_df[[DATE_COL, "Region", "Ship Mode", "Division", "Product ID", "Units", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "Actual Lead Time Days"})
    predictions["Predicted Lead Time Days"] = model_predictions
    predictions["Baseline Predicted Lead Time Days"] = baseline_predictions
    predictions["Prediction Error Days"] = (
        predictions["Predicted Lead Time Days"] - predictions["Actual Lead Time Days"]
    )
    predictions["Absolute Error Days"] = predictions["Prediction Error Days"].abs()

    comparison_file = OUTPUT_DIR / "time_based_model_comparison.csv"
    predictions_file = OUTPUT_DIR / "time_based_test_predictions.csv"
    summary_file = OUTPUT_DIR / "step10_validation_summary.txt"

    comparison.to_csv(comparison_file, index=False)
    predictions.to_csv(predictions_file, index=False)

    model_row = comparison.loc[comparison["Model"] == "Gradient Boosting Regressor"].iloc[0]
    baseline_row = comparison.loc[comparison["Model"] == "Training-Mean Baseline"].iloc[0]
    model_better = model_row["MAE (days)"] < baseline_row["MAE (days)"] and model_row["RMSE (days)"] < baseline_row["RMSE (days)"]

    summary_lines = [
        "STEP 10: TIME-BASED MODEL VALIDATION & PERFORMANCE ANALYSIS",
        "=" * 68,
        "",
        f"Input file: {INPUT_FILE.relative_to(BASE_DIR)}",
        f"Rows in source file: {original_rows:,}",
        f"Rows used after excluding invalid/missing order dates or target: {len(df):,}",
        f"Training records: {len(train_df):,}",
        f"Testing records: {len(test_df):,}",
        f"Training order-date range: {train_df[DATE_COL].min().date()} to {train_df[DATE_COL].max().date()}",
        f"Testing order-date range: {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()}",
        f"Chronological cutoff date (first date in test period): {cutoff_date.date()}",
        f"Training-mean baseline prediction: {baseline_value:.2f} days",
        "",
        "MODEL PERFORMANCE",
        "-" * 68,
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "INTERPRETATION",
        "-" * 68,
        (
            "The Gradient Boosting model has lower MAE and RMSE than the training-mean baseline."
            if model_better
            else
            "The Gradient Boosting model did not beat the baseline on both MAE and RMSE. "
            "Do not treat it as reliable for future predictions without further investigation."
        ),
        "",
        "IMPORTANT LIMITATIONS",
        "-" * 68,
        "1. This is a chronological holdout test; it is more realistic than a random split, but it is only one test period.",
        "2. The supplied data has unusually long historical shipping lead times (roughly 904-1,642 days); verify the source dates and business meaning.",
        "3. The model features do not include factory identity. This model cannot directly estimate the effect of reallocating a product to a different factory.",
        "4. These results do not establish lead-time improvement, cost savings, capacity feasibility, or profit impact from reallocation.",
        "5. Do not use these results alone to make real operational decisions. Validate the source data and gather factory-specific cost, capacity, and routing information.",
        "",
        "OUTPUT FILES",
        "-" * 68,
        str(comparison_file.relative_to(BASE_DIR)),
        str(predictions_file.relative_to(BASE_DIR)),
        str(summary_file.relative_to(BASE_DIR)),
        "",
        "No existing app.py, saved model, or Step 4-Step 9 output files were modified by this script.",
    ]
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSTEP 10 COMPLETED")
    print("=" * 50)
    print(f"Training records: {len(train_df):,}")
    print(f"Testing records:  {len(test_df):,}")
    print(f"Training dates:   {train_df[DATE_COL].min().date()} to {train_df[DATE_COL].max().date()}")
    print(f"Testing dates:    {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()}")
    print("\nModel comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCreated:")
    print(f"  {comparison_file}")
    print(f"  {predictions_file}")
    print(f"  {summary_file}")
    print("\nNote: These metrics evaluate lead-time prediction only, not factory-reallocation impact.")


if __name__ == "__main__":
    main()
