import pandas as pd
import numpy as np

def inspect_raw_data(df):
    """
    FR-01: Load and inspect raw data profiling shape, dtypes, missing values, duplicates, and stats.
    """
    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.describe(include=[np.number]).to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    return profile

def remove_duplicates(df):
    """
    FR-03: Remove exact duplicate rows and log count changes.
    """
    initial_rows = len(df)
    duplicates_count = int(df.duplicated().sum())
    df_cleaned = df.drop_duplicates().copy()
    final_rows = len(df_cleaned)
    
    log = {
        "initial_rows": initial_rows,
        "duplicates_found": duplicates_count,
        "rows_removed": duplicates_count,
        "final_rows": final_rows
    }
    return df_cleaned, log

def fix_data_types_and_invalid_values(df):
    """
    FR-04: Fix data types, standardize string formatting, clean currency, parse dates, and invalidate impossible values.
    """
    df_out = df.copy()
    fixes_log = []

    # 1. Clean Sales currency strings to numeric float
    if "Sales" in df_out.columns:
        if df_out["Sales"].dtype == object:
            df_out["Sales"] = (
                df_out["Sales"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df_out["Sales"] = pd.to_numeric(df_out["Sales"], errors="coerce")
            fixes_log.append("Converted 'Sales' text currency strings to float numeric.")

    # 2. Convert Order_Date and Ship_Date to pandas datetime
    for date_col in ["Order_Date", "Ship_Date"]:
        if date_col in df_out.columns:
            df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce", format="mixed")
            fixes_log.append(f"Parsed '{date_col}' strings to datetime objects.")

    # 3. Standardize text casing and spacing in categorical columns
    string_cols = ["Segment", "Region", "Category", "Sub_Category", "Ship_Mode", "Customer_Segment", "Order_Priority"]
    for col in string_cols:
        if col in df_out.columns and df_out[col].dtype == object:
            df_out[col] = df_out[col].astype(str).str.strip().str.title()
            df_out[col] = df_out[col].replace({"Nan": np.nan, "None": np.nan, "": np.nan})
            fixes_log.append(f"Standardized whitespace and title casing for '{col}'.")

    # 4. Invalidate domain values outside valid bounds
    if "Quantity" in df_out.columns:
        invalid_qty = (df_out["Quantity"] <= 0).sum()
        df_out.loc[df_out["Quantity"] <= 0, "Quantity"] = np.nan
        if invalid_qty > 0:
            fixes_log.append(f"Invalidated {invalid_qty} negative/zero 'Quantity' observations.")

    if "Discount" in df_out.columns:
        invalid_disc = ((df_out["Discount"] < 0) | (df_out["Discount"] > 1.0)).sum()
        df_out.loc[(df_out["Discount"] < 0) | (df_out["Discount"] > 1.0), "Discount"] = np.nan
        if invalid_disc > 0:
            fixes_log.append(f"Invalidated {invalid_disc} out-of-bounds 'Discount' observations (<0 or >1.0).")

    return df_out, fixes_log

def handle_missing_values(df):
    """
    FR-02: Calculate missing counts and treat with justified imputation methods.
    """
    df_out = df.copy()
    treatment_log = {}

    # Postal_Code Imputation
    if "Postal_Code" in df_out.columns and df_out["Postal_Code"].isnull().sum() > 0:
        null_count = int(df_out["Postal_Code"].isnull().sum())
        # Impute with mode per State / City or global mode
        if "State" in df_out.columns:
            df_out["Postal_Code"] = df_out.groupby("State")["Postal_Code"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 10001))
        df_out["Postal_Code"] = df_out["Postal_Code"].fillna(10001).astype(int)
        treatment_log["Postal_Code"] = f"Imputed {null_count} missing Postal_Code entries using State mode."

    # Categorical Imputation (Mode or 'Unknown')
    for cat_col in ["Region", "Segment", "Order_Priority"]:
        if cat_col in df_out.columns and df_out[cat_col].isnull().sum() > 0:
            null_count = int(df_out[cat_col].isnull().sum())
            mode_val = df_out[cat_col].mode()[0] if not df_out[cat_col].mode().empty else "Unknown"
            df_out[cat_col] = df_out[cat_col].fillna(mode_val)
            treatment_log[cat_col] = f"Imputed {null_count} missing values with mode '{mode_val}'."

    # Quantity: Impute with median quantity
    if "Quantity" in df_out.columns and df_out["Quantity"].isnull().sum() > 0:
        null_count = int(df_out["Quantity"].isnull().sum())
        median_qty = df_out["Quantity"].median()
        df_out["Quantity"] = df_out["Quantity"].fillna(median_qty)
        treatment_log["Quantity"] = f"Imputed {null_count} missing quantities with median ({median_qty:.0f})."

    # Discount: Impute with median discount
    if "Discount" in df_out.columns and df_out["Discount"].isnull().sum() > 0:
        null_count = int(df_out["Discount"].isnull().sum())
        median_disc = df_out["Discount"].median()
        df_out["Discount"] = df_out["Discount"].fillna(median_disc)
        treatment_log["Discount"] = f"Imputed {null_count} missing discounts with median ({median_disc:.2f})."

    # Sales: Impute using median per Category / Sub_Category
    if "Sales" in df_out.columns and df_out["Sales"].isnull().sum() > 0:
        null_count = int(df_out["Sales"].isnull().sum())
        if "Sub_Category" in df_out.columns:
            df_out["Sales"] = df_out.groupby("Sub_Category")["Sales"].transform(lambda x: x.fillna(x.median()))
        df_out["Sales"] = df_out["Sales"].fillna(df_out["Sales"].median())
        treatment_log["Sales"] = f"Imputed {null_count} missing Sales entries using Sub_Category median."

    # Profit: Impute using median Profit per Sub_Category
    if "Profit" in df_out.columns and df_out["Profit"].isnull().sum() > 0:
        null_count = int(df_out["Profit"].isnull().sum())
        if "Sub_Category" in df_out.columns:
            df_out["Profit"] = df_out.groupby("Sub_Category")["Profit"].transform(lambda x: x.fillna(x.median()))
        df_out["Profit"] = df_out["Profit"].fillna(df_out["Profit"].median())
        treatment_log["Profit"] = f"Imputed {null_count} missing Profit entries using Sub_Category median."

    # Order_Date: Forward fill or backward fill if missing
    if "Order_Date" in df_out.columns and df_out["Order_Date"].isnull().sum() > 0:
        null_count = int(df_out["Order_Date"].isnull().sum())
        df_out["Order_Date"] = df_out["Order_Date"].ffill().bfill()
        treatment_log["Order_Date"] = f"Filled {null_count} missing dates using forward/backward fill."

    return df_out, treatment_log

def handle_outliers(df, numeric_cols=None, method="iqr"):
    """
    FR-05: Detect and treat extreme numeric outliers using IQR capping (Winsorization) to preserve valid row observations.
    """
    df_out = df.copy()
    outlier_log = {}

    if numeric_cols is None:
        numeric_cols = [c for c in ["Sales", "Profit", "Discount", "Quantity"] if c in df_out.columns]

    for col in numeric_cols:
        if col in df_out.columns:
            Q1 = float(df_out[col].quantile(0.25))
            Q3 = float(df_out[col].quantile(0.75))
            IQR = Q3 - Q1
            lower_bound = Q1 - 3.0 * IQR  # 3.0x IQR for conservative Winsorization
            upper_bound = Q3 + 3.0 * IQR

            outliers_count = int(((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum())
            
            # Cap extreme values at lower and upper bounds
            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])

            outlier_log[col] = {
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outliers_detected": outliers_count,
                "action": "Capped extreme values to [lower_bound, upper_bound] to retain row observations while stabilizing distributions."
            }

    return df_out, outlier_log

def validate_cleaning(df_raw, df_cleaned):
    """
    Section 8: Perform before-and-after data quality validation.
    """
    summary = {
        "Metric": ["Total Rows", "Total Columns", "Duplicate Rows", "Missing Values Total", "Date Column Dtype", "Sales Column Dtype"],
        "Before Cleaning": [
            len(df_raw),
            df_raw.shape[1],
            int(df_raw.duplicated().sum()),
            int(df_raw.isnull().sum().sum()),
            str(df_raw["Order_Date"].dtype) if "Order_Date" in df_raw.columns else "N/A",
            str(df_raw["Sales"].dtype) if "Sales" in df_raw.columns else "N/A"
        ],
        "After Cleaning": [
            len(df_cleaned),
            df_cleaned.shape[1],
            int(df_cleaned.duplicated().sum()),
            int(df_cleaned.isnull().sum().sum()),
            str(df_cleaned["Order_Date"].dtype) if "Order_Date" in df_cleaned.columns else "N/A",
            str(df_cleaned["Sales"].dtype) if "Sales" in df_cleaned.columns else "N/A"
        ]
    }
    return pd.DataFrame(summary)

def run_full_cleaning_pipeline(df_raw):
    """
    Executes end-to-end data cleaning workflow and returns cleaned dataframe and execution report.
    """
    raw_profile = inspect_raw_data(df_raw)
    df_no_dups, dup_log = remove_duplicates(df_raw)
    df_fixed, type_log = fix_data_types_and_invalid_values(df_no_dups)
    df_imputed, missing_log = handle_missing_values(df_fixed)
    df_cleaned, outlier_log = handle_outliers(df_imputed)
    validation_df = validate_cleaning(df_raw, df_cleaned)

    report = {
        "raw_profile": raw_profile,
        "duplicate_log": dup_log,
        "type_fixes": type_log,
        "missing_log": missing_log,
        "outlier_log": outlier_log,
        "validation_summary": validation_df.to_dict(orient="records")
    }

    return df_cleaned, report
