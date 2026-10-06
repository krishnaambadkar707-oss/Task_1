# Implementation Plan - Task 01: Data Cleaning and Visualization

Build a complete, reproducible data cleaning and exploratory data analysis (EDA) pipeline based on **PRD – Task 01**. The project will process an open-source multi-feature dataset (E-Commerce & Retail Sales / Customer Analytics), clean raw data anomalies (missing values, duplicates, invalid data types, outliers), perform comprehensive EDA with visualizations, export interactive dashboards, and deliver professional documentation.

## User Review Required

> [!NOTE]
> **Dataset Selection**: We will use a realistic E-Commerce Retail & Customer Analytics dataset containing numerical metrics (Sales, Profit, Quantity, Discount, Shipping Cost, Customer Age) and categorical variables (Region, Category, Sub-Category, Segment, Order Priority). It includes realistic data quality anomalies (missing values, duplicate rows, date string inconsistencies, currency formatting issues, and price/profit outliers) to thoroughly satisfy all PRD cleaning requirements.

> [!TIP]
> **Deliverables**: The pipeline will output both static high-resolution figures for report inclusion (`outputs/figures/*.png`) and a standalone interactive Plotly dashboard (`outputs/interactive_dashboard.html`).

---

## Proposed Changes

### Project Structure & Setup

#### [NEW] [requirements.txt](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/requirements.txt)
- Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `jupyter`.

#### [NEW] [README.md](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/README.md)
- Complete project guide including dataset source details, environment setup instructions, data cleaning methodology, EDA key findings, visual gallery, and project limitations.

---

### Data Pipeline Core Modules (`src/`)

#### [NEW] [src/dataset_loader.py](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/src/dataset_loader.py)
- Fetches/generates raw open-source dataset into `data/raw/raw_dataset.csv`.
- Embeds realistic data quality issues (missing entries, exact duplicates, string-formatted dates/prices, extreme outliers).

#### [NEW] [src/cleaning.py](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/src/cleaning.py)
- Modular data cleaning utilities:
  - `inspect_raw_data(df)`: Data quality profiling (shape, nulls, duplicates, dtypes).
  - `handle_missing_values(df)`: Column-wise imputation & domain-justified treatments.
  - `remove_duplicates(df)`: Detection and removal of exact/repeated records.
  - `fix_data_types(df)`: Parsing datetime strings, cleaning currency/numeric strings, standardizing text casing.
  - `handle_outliers(df)`: IQR-based outlier identification, capping/winsorizing bounds, and justification logging.
  - `validate_cleaning(df_raw, df_cleaned)`: Before-vs-after audit metrics.

#### [NEW] [src/visualization.py](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/src/visualization.py)
- Plotting functions for Seaborn/Matplotlib and Plotly:
  - `plot_histograms()`: Numerical distributions (Sales, Profit, Discount, Quantity).
  - `plot_bar_charts()`: Categorical breakdowns (Sales by Region/Category/Segment).
  - `plot_boxplots()`: Outlier & variance distribution comparisons.
  - `plot_correlation_heatmap()`: Feature correlation matrix.
  - `plot_scatter_relationships()`: Pairwise numerical relationships (Sales vs Profit).
  - `generate_interactive_dashboard()`: Standalone HTML interactive Plotly dashboard.

---

### Executable Notebook & Pipeline Script

#### [NEW] [notebooks/task_01_eda.ipynb](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/notebooks/task_01_eda.ipynb)
- Interactive Jupyter Notebook containing all 13 recommended PRD sections from Project Objective to Key Findings and Conclusion.

#### [NEW] [run_pipeline.py](file:///c:/Users/krish/OneDrive/Desktop/Internship/Task%201/run_pipeline.py)
- Standalone CLI execution script that automates raw dataset generation/loading, data cleaning, validation reporting, plot export, and dataset saving.

---

## Verification Plan

### Automated Tests & Pipeline Execution
- Execute `python run_pipeline.py` to verify:
  1. Raw dataset saved to `data/raw/raw_dataset.csv`.
  2. Data cleaning pipeline logs raw vs. cleaned metrics (0 missing values, 0 duplicates, corrected dtypes).
  3. Cleaned dataset saved to `data/processed/cleaned_dataset.csv`.
  4. All static EDA figures exported to `outputs/figures/`.
  5. Interactive dashboard generated at `outputs/interactive_dashboard.html`.

### Notebook Execution
- Run notebook kernel validation to ensure `notebooks/task_01_eda.ipynb` runs top-to-bottom without errors.
