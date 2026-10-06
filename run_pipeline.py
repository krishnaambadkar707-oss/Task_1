import os
import pandas as pd
from src.dataset_loader import fetch_real_dataset
from src.cleaning import run_full_cleaning_pipeline
from src.visualization import generate_all_visualizations

def main():
    print("=" * 70)
    print("      TASK 01: END-TO-END DATA CLEANING & EDA PIPELINE")
    print("=" * 70)

    # Step 1: Fetch Real Open-Source Dataset
    raw_path = "data/raw/raw_dataset.csv"
    print("\n[Step 1/4] Acquiring real open-source dataset (Superstore Sales)...")
    df_raw = fetch_real_dataset(raw_path)

    # Step 2: Data Cleaning & Validation
    print("\n[Step 2/4] Running data cleaning pipeline (Handling missing, dups, types & outliers)...")
    df_cleaned, report = run_full_cleaning_pipeline(df_raw)

    # Save cleaned dataset
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    df_cleaned.to_csv("data/processed/cleaned_dataset.csv", index=False)
    df_cleaned.to_csv("outputs/cleaned_dataset.csv", index=False)
    print(f"[Pipeline] Cleaned dataset exported to 'data/processed/cleaned_dataset.csv' & 'outputs/cleaned_dataset.csv'")

    # Step 3: Print Data Quality Audit Metrics
    print("\n" + "=" * 70)
    print("                  BEFORE & AFTER CLEANING AUDIT REPORT")
    print("=" * 70)
    val_df = pd.DataFrame(report["validation_summary"])
    print(val_df.to_string(index=False))
    print("=" * 70)

    print("\n--- Treatment Rationale Summary ---")
    print(f"• Duplicate Rows Removed: {report['duplicate_log']['duplicates_found']}")
    for fix in report['type_fixes']:
        print(f"• Type/Formatting Fix: {fix}")
    for col, txt in report['missing_log'].items():
        print(f"• Missing Value Treatment ({col}): {txt}")
    for col, out_dict in report['outlier_log'].items():
        print(f"• Outlier Capping ({col}): Detected {out_dict['outliers_detected']} extreme bounds. Action: {out_dict['action']}")

    # Step 4: Generate Visualizations & Interactive Dashboard
    print("\n[Step 4/4] Exporting EDA charts and Plotly interactive dashboard...")
    vis_paths = generate_all_visualizations(df_cleaned)

    print("\n" + "=" * 70)
    print("SUCCESS: Pipeline executed cleanly!")
    print("Key Deliverables:")
    print(f" 1. Raw Dataset:         data/raw/raw_dataset.csv")
    print(f" 2. Cleaned Dataset:     data/processed/cleaned_dataset.csv")
    print(f" 3. Static Figures:      outputs/figures/")
    print(f" 4. Interactive Dash:    outputs/interactive_dashboard.html")
    print("=" * 70)

if __name__ == "__main__":
    main()
