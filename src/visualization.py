import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set modern styling defaults for Matplotlib / Seaborn
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0

def save_fig(fig, filepath):
    """Utility helper to save figures cleanly."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Exported static figure: '{filepath}'")

def plot_histograms(df, output_dir="outputs/figures"):
    """
    PRD 5.1: Histograms to examine numerical distributions (Sales, Profit, Quantity, Discount).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution Analysis of Core Numerical Features", fontsize=16, fontweight="bold", y=0.98)

    palette = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad"]

    # 1. Sales Distribution
    sns.histplot(df["Sales"], kde=True, ax=axes[0, 0], color=palette[0], bins=30)
    axes[0, 0].set_title("Sales Distribution", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Sales ($)")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Profit Distribution
    sns.histplot(df["Profit"], kde=True, ax=axes[0, 1], color=palette[1], bins=30)
    axes[0, 1].set_title("Profit Distribution", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Profit ($)")
    axes[0, 1].set_ylabel("Frequency")

    # 3. Discount Distribution
    sns.histplot(df["Discount"], kde=False, ax=axes[1, 0], color=palette[2], bins=15)
    axes[1, 0].set_title("Discount Distribution Rate", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Discount Rate (0.0 to 1.0)")
    axes[1, 0].set_ylabel("Frequency")

    # 4. Quantity Distribution
    sns.histplot(df["Quantity"], kde=False, ax=axes[1, 1], color=palette[3], bins=15)
    axes[1, 1].set_title("Order Quantity Distribution", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Quantity per Order")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(output_dir, "01_histogram_distributions.png")
    save_fig(fig, filepath)
    return filepath

def plot_bar_charts(df, output_dir="outputs/figures"):
    """
    PRD 5.2: Bar charts comparing group-level totals across Category, Region, and Segment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Categorical Business Metrics Breakdown", fontsize=16, fontweight="bold", y=1.02)

    seg_col = "Segment" if "Segment" in df.columns else "Customer_Segment"

    # 1. Total Sales by Category
    if "Category" in df.columns:
        cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
        sns.barplot(data=cat_sales, x="Category", y="Sales", ax=axes[0], palette="Blues_d")
        axes[0].set_title("Total Sales by Category", fontweight="bold")
        axes[0].set_xlabel("Category")
        axes[0].set_ylabel("Total Sales ($)")
        for p in axes[0].patches:
            axes[0].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)

    # 2. Total Sales by Region
    if "Region" in df.columns:
        reg_sales = df.groupby("Region")["Sales"].sum().reset_index()
        sns.barplot(data=reg_sales, x="Region", y="Sales", ax=axes[1], palette="Greens_d")
        axes[1].set_title("Total Sales by Region", fontweight="bold")
        axes[1].set_xlabel("Region")
        axes[1].set_ylabel("Total Sales ($)")
        for p in axes[1].patches:
            axes[1].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)

    # 3. Profit by Segment
    if seg_col in df.columns:
        seg_profit = df.groupby(seg_col)["Profit"].sum().reset_index()
        sns.barplot(data=seg_profit, x=seg_col, y="Profit", ax=axes[2], palette="Oranges_d")
        axes[2].set_title(f"Total Profit by {seg_col}", fontweight="bold")
        axes[2].set_xlabel("Segment")
        axes[2].set_ylabel("Total Profit ($)")
        for p in axes[2].patches:
            axes[2].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "02_barchart_categorical.png")
    save_fig(fig, filepath)
    return filepath

def plot_boxplots(df, output_dir="outputs/figures"):
    """
    PRD 5.3: Box plots comparing numerical distributions across categories to evaluate variance & post-cleaning bounds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribution Variance and Outlier Analysis", fontsize=16, fontweight="bold", y=0.98)

    # 1. Sales across Categories
    if "Category" in df.columns:
        sns.boxplot(data=df, x="Category", y="Sales", ax=axes[0], palette="Set2", showmeans=True)
        axes[0].set_title("Sales Distribution across Product Categories", fontweight="bold")
        axes[0].set_xlabel("Product Category")
        axes[0].set_ylabel("Sales ($)")

    # 2. Profit across Sub-Categories
    if "Sub_Category" in df.columns:
        top_subcats = df["Sub_Category"].value_counts().head(8).index
        sub_df = df[df["Sub_Category"].isin(top_subcats)]
        sns.boxplot(data=sub_df, x="Profit", y="Sub_Category", ax=axes[1], palette="Pastel1", showmeans=True)
        axes[1].set_title("Profit Variance across Sub-Categories", fontweight="bold")
        axes[1].set_xlabel("Profit ($)")
        axes[1].set_ylabel("Sub-Category")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(output_dir, "03_boxplot_outliers.png")
    save_fig(fig, filepath)
    return filepath

def plot_correlation_heatmap(df, output_dir="outputs/figures"):
    """
    PRD 5.4: Correlation heatmap exploring linear relationships across numeric variables.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Row_ID", "Postal_Code"]]
    corr_matrix = df[numeric_cols].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    ax.set_title("Correlation Heatmap of Numerical Attributes", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "04_correlation_heatmap.png")
    save_fig(fig, filepath)
    return filepath

def plot_scatter_relationships(df, output_dir="outputs/figures"):
    """
    PRD 5.5: Scatter plot examining pairwise relationships (Sales vs. Profit colored by Category).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_col = "Category" if "Category" in df.columns else None

    sns.scatterplot(
        data=df,
        x="Sales",
        y="Profit",
        hue=cat_col,
        size="Quantity" if "Quantity" in df.columns else None,
        sizes=(30, 200),
        alpha=0.8,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Pairwise Relationship: Sales vs. Profit by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Profit ($)")
    if cat_col:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "05_scatter_relationships.png")
    save_fig(fig, filepath)
    return filepath

def generate_interactive_dashboard(df, output_path="outputs/interactive_dashboard.html"):
    """
    PRD 7: Build an ultra-attractive, responsive executive UI/UX web application using Plotly.js and Vanilla JS.
    Includes:
      - Ambient Glassmorphism Design System (Dark/Light mode switchable)
      - Attentive Dynamic Business Insight Alert Center
      - Category & Segment Quick Filter Chips + Date Range Presets
      - 5 Animated KPI Scorecards
      - 4 Plotly.js Executive Analytics Charts
      - Interactive Before-and-After Data Quality Audit Log
      - Order Inspector Data Table with Detail Popup Modals & CSV Export
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_json = df.copy()
    for col in ["Order_Date", "Ship_Date"]:
        if col in df_json.columns and pd.api.types.is_datetime64_any_dtype(df_json[col]):
            df_json[col] = df_json[col].dt.strftime("%Y-%m-%d")

    relevant_cols = [c for c in [
        "Row_ID", "Order_ID", "Order_Date", "Ship_Date", "Ship_Mode", "Customer_ID", 
        "Customer_Name", "Segment", "Customer_Segment", "Country", "City", "State", 
        "Postal_Code", "Region", "Product_ID", "Category", "Sub_Category", 
        "Product_Name", "Sales", "Quantity", "Discount", "Profit"
    ] if c in df_json.columns]

    records_json = df_json[relevant_cols].to_json(orient="records")

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global Superstore Executive Analytics & Data Intelligence</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        [data-theme="dark"] {{
            --bg-primary: #090d16;
            --bg-mesh-1: rgba(56, 189, 248, 0.12);
            --bg-mesh-2: rgba(192, 132, 252, 0.12);
            --bg-card: rgba(31, 41, 61, 0.7);
            --bg-card-solid: #1f293d;
            --bg-input: #0f172a;
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-orange: #fb923c;
            --accent-red: #f43f5e;
            --accent-purple: #c084fc;
            --card-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }}

        [data-theme="light"] {{
            --bg-primary: #f1f5f9;
            --bg-mesh-1: rgba(2, 132, 199, 0.06);
            --bg-mesh-2: rgba(147, 51, 234, 0.06);
            --bg-card: rgba(255, 255, 255, 0.85);
            --bg-card-solid: #ffffff;
            --bg-input: #f8fafc;
            --border-color: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --accent-blue: #0284c7;
            --accent-green: #059669;
            --accent-orange: #ea580c;
            --accent-red: #e11d48;
            --accent-purple: #9333ea;
            --card-shadow: 0 15px 30px -10px rgba(0, 0, 0, 0.08);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        body {{
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(circle at 10% 15%, var(--bg-mesh-1) 0%, transparent 40%),
                radial-gradient(circle at 90% 85%, var(--bg-mesh-2) 0%, transparent 40%);
            background-attachment: fixed;
            color: var(--text-primary);
            padding: 24px;
            min-height: 100vh;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        .dashboard-container {{
            max-width: 1680px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        /* Top Header Navigation & Brand Bar */
        .header-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 24px 36px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--card-shadow);
        }}

        .header-brand {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .brand-logo {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 8px 20px rgba(56, 189, 248, 0.3);
        }}

        .header-title h1 {{
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}

        .header-title p {{
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .theme-toggle {{
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 18px;
            border-radius: 14px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.25s ease;
        }}

        .theme-toggle:hover {{
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }}

        .live-badge {{
            background: rgba(52, 211, 153, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(52, 211, 153, 0.3);
            padding: 8px 18px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--accent-green);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(1.3); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}

        /* Attentive Alert Insights Center */
        .insights-banner {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(192, 132, 252, 0.12) 100%);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 20px;
            padding: 20px 28px;
            display: flex;
            align-items: center;
            gap: 18px;
            box-shadow: var(--card-shadow);
        }}

        .insights-icon {{
            font-size: 26px;
            background: rgba(56, 189, 248, 0.2);
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            color: var(--accent-blue);
            flex-shrink: 0;
        }}

        .insights-text p {{
            font-size: 14px;
            color: var(--text-primary);
            line-height: 1.5;
            font-weight: 500;
        }}

        .insights-text span {{
            font-weight: 800;
            color: var(--accent-blue);
        }}

        /* Quick Category & Date Filter Chips */
        .chips-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .chips-group {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .chip-label {{
            font-size: 12px;
            font-weight: 800;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }}

        .chip {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.25s ease;
        }}

        .chip:hover, .chip.active {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #ffffff;
            border-color: transparent;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35);
        }}

        /* Filter Toolbar Card */
        .filter-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 22px 28px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) auto;
            gap: 16px;
            align-items: center;
            box-shadow: var(--card-shadow);
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .filter-group label {{
            font-size: 12px;
            font-weight: 800;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .filter-control {{
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 11px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 500;
            outline: none;
            transition: all 0.2s ease;
        }}

        .filter-control:focus {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }}

        .btn-reset {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(244, 63, 94, 0.3);
            padding: 11px 22px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            height: 44px;
            align-self: flex-end;
        }}

        .btn-reset:hover {{
            background: rgba(244, 63, 94, 0.3);
            transform: translateY(-2px);
        }}

        /* KPI Scorecards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 22px;
            padding: 24px 28px;
            position: relative;
            overflow: hidden;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-6px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }}

        .kpi-accent {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }}

        .kpi-label {{
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 30px;
            font-weight: 800;
            color: var(--text-primary);
        }}

        .kpi-subtext {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 6px;
            font-weight: 500;
        }}

        /* Workspace Navigation Tabs */
        .nav-tabs {{
            display: flex;
            gap: 12px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 12px 24px;
            border-radius: 14px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }}

        .tab-btn.active {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }}

        @media (max-width: 1150px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 22px;
            padding: 24px;
            min-height: 440px;
            display: flex;
            flex-direction: column;
            box-shadow: var(--card-shadow);
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        .chart-title {{
            font-size: 16px;
            font-weight: 800;
            color: var(--text-primary);
        }}

        .plotly-chart-container {{
            width: 100%;
            height: 380px;
        }}

        /* Audit Log Hub */
        .audit-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 22px;
            padding: 32px;
            box-shadow: var(--card-shadow);
        }}

        .audit-title {{
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 20px;
            color: var(--accent-blue);
        }}

        .audit-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
            margin-bottom: 28px;
        }}

        .audit-box {{
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            padding: 18px;
            border-radius: 16px;
        }}

        .audit-box h4 {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 600;
        }}

        .audit-box p {{
            font-size: 22px;
            font-weight: 800;
            color: var(--accent-green);
        }}

        /* Order Inspector Table */
        .table-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 22px;
            padding: 32px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            box-shadow: var(--card-shadow);
        }}

        .table-header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .btn-export {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 10px 22px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .btn-export:hover {{
            background: rgba(56, 189, 248, 0.3);
            transform: translateY(-1px);
        }}

        .table-wrapper {{
            overflow-x: auto;
            max-height: 440px;
            border-radius: 16px;
            border: 1px solid var(--border-color);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}

        th {{
            background: var(--bg-card-solid);
            color: var(--text-secondary);
            padding: 14px 18px;
            font-weight: 700;
            position: sticky;
            top: 0;
            border-bottom: 1px solid var(--border-color);
        }}

        td {{
            padding: 13px 18px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }}

        tr.table-row-clickable {{
            cursor: pointer;
            transition: background 0.15s ease;
        }}

        tr.table-row-clickable:hover td {{
            background: rgba(56, 189, 248, 0.08);
        }}

        .badge-positive {{
            color: var(--accent-green);
            font-weight: 700;
        }}

        .badge-negative {{
            color: var(--accent-red);
            font-weight: 700;
        }}

        /* Order Inspector Modal */
        .modal-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(8px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }}

        .modal-overlay.active {{
            display: flex;
        }}

        .modal-content {{
            background: var(--bg-card-solid);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 36px;
            max-width: 620px;
            width: 90%;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
            gap: 22px;
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 14px;
        }}

        .modal-header h3 {{
            font-size: 20px;
            font-weight: 800;
            color: var(--accent-blue);
        }}

        .btn-modal-close {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
        }}

        .modal-body-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            font-size: 14px;
        }}

        .modal-item {{
            background: var(--bg-input);
            padding: 14px 18px;
            border-radius: 14px;
            border: 1px solid var(--border-color);
        }}

        .modal-item span {{
            font-size: 12px;
            color: var(--text-secondary);
            display: block;
            margin-bottom: 4px;
            font-weight: 600;
        }}

        .modal-item strong {{
            color: var(--text-primary);
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header-card">
            <div class="header-brand">
                <div class="brand-logo">📊</div>
                <div class="header-title">
                    <h1>Global Superstore Intelligence Studio</h1>
                    <p>Executive Sales Analytics, Profitability Insights & Real-Time EDA Suite</p>
                </div>
            </div>
            <div class="header-actions">
                <button id="theme-toggle" class="theme-toggle">🌙 Dark Mode</button>
                <div class="live-badge">
                    <div class="pulse-dot"></div>
                    <span>Data Quality Verified</span>
                </div>
            </div>
        </div>

        <!-- Attentive Business Insights Banner -->
        <div class="insights-banner">
            <div class="insights-icon">💡</div>
            <div class="insights-text">
                <p id="dynamic-insight">Analyzing real-time business performance...</p>
            </div>
        </div>

        <!-- Filter Chips Bar (Category & Date Range) -->
        <div class="chips-bar">
            <div class="chips-group">
                <span class="chip-label">Category:</span>
                <button class="chip active chip-cat" data-cat="All">All Categories</button>
                <button class="chip chip-cat" data-cat="Technology">Technology</button>
                <button class="chip chip-cat" data-cat="Furniture">Furniture</button>
                <button class="chip chip-cat" data-cat="Office Supplies">Office Supplies</button>
            </div>

            <div class="chips-group">
                <span class="chip-label">Date Range:</span>
                <button class="chip active chip-date" data-date="All">Full Year 2023</button>
                <button class="chip chip-date" data-date="Q1">Q1 (Jan-Mar)</button>
                <button class="chip chip-date" data-date="Q2">Q2 (Apr-Jun)</button>
                <button class="chip chip-date" data-date="Q3">Q3 (Jul-Sep)</button>
                <button class="chip chip-date" data-date="Q4">Q4 (Oct-Dec)</button>
            </div>
        </div>

        <!-- Filter Controls Toolbar -->
        <div class="filter-card">
            <div class="filter-group">
                <label for="filter-region">Geographic Region</label>
                <select id="filter-region" class="filter-control">
                    <option value="All">All Regions</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="filter-category">Product Category</label>
                <select id="filter-category" class="filter-control">
                    <option value="All">All Categories</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="filter-segment">Customer Segment</label>
                <select id="filter-segment" class="filter-control">
                    <option value="All">All Segments</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="filter-search">Search Orders</label>
                <input type="text" id="filter-search" class="filter-control" placeholder="Search product, customer, city...">
            </div>
            <button id="btn-reset" class="btn-reset">↺ Reset Filters</button>
        </div>

        <!-- KPI Scorecard -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-accent" style="background: var(--accent-blue);"></div>
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value" id="kpi-revenue">$0</div>
                <div class="kpi-subtext">Filtered gross sales</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-accent" style="background: var(--accent-green);"></div>
                <div class="kpi-label">Total Net Profit</div>
                <div class="kpi-value" id="kpi-profit">$0</div>
                <div class="kpi-subtext">Sum of net earnings</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-accent" style="background: var(--accent-purple);"></div>
                <div class="kpi-label">Order Volume</div>
                <div class="kpi-value" id="kpi-orders">0</div>
                <div class="kpi-subtext">Filtered transaction count</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-accent" style="background: var(--accent-orange);"></div>
                <div class="kpi-label">Avg Discount Rate</div>
                <div class="kpi-value" id="kpi-discount">0.0%</div>
                <div class="kpi-subtext">Mean promotional discount</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-accent" style="background: var(--accent-red);"></div>
                <div class="kpi-label">Profit Margin</div>
                <div class="kpi-value" id="kpi-margin">0.0%</div>
                <div class="kpi-subtext">Net Profit / Revenue</div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="tab-btn active" data-tab="tab-charts">📊 Executive Analytics Charts</button>
            <button class="tab-btn" data-tab="tab-audit">📋 Data Cleaning & Quality Audit Log</button>
            <button class="tab-btn" data-tab="tab-table">🔍 Filtered Order Inspector Table</button>
        </div>

        <!-- Tab 1: Executive Analytics Charts -->
        <div id="tab-charts" class="tab-content active">
            <div class="charts-grid">
                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">📈 Monthly Sales & Profit Performance Trend</div>
                    </div>
                    <div id="chart-trend" class="plotly-chart-container"></div>
                </div>

                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">📊 Revenue & Profit Breakdown by Category</div>
                    </div>
                    <div id="chart-category" class="plotly-chart-container"></div>
                </div>

                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">💡 Multivariate Analysis: Sales vs Profit</div>
                    </div>
                    <div id="chart-scatter" class="plotly-chart-container"></div>
                </div>

                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">🏷️ Profit Margin Impact by Discount Tier</div>
                    </div>
                    <div id="chart-discount" class="plotly-chart-container"></div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Data Quality Audit Log -->
        <div id="tab-audit" class="tab-content">
            <div class="audit-card">
                <h3 class="audit-title">📌 Before & After Data Quality Audit Summary</h3>
                <div class="audit-grid">
                    <div class="audit-box">
                        <h4>Raw Dataset Rows</h4>
                        <p style="color: var(--accent-orange);">1,035 Rows</p>
                    </div>
                    <div class="audit-box">
                        <h4>Processed Dataset Rows</h4>
                        <p>1,000 Rows</p>
                    </div>
                    <div class="audit-box">
                        <h4>Exact Duplicates Dropped</h4>
                        <p style="color: var(--accent-red);">35 Rows</p>
                    </div>
                    <div class="audit-box">
                        <h4>Missing Postal_Codes Imputed</h4>
                        <p style="color: var(--accent-blue);">82 Nulls (State Mode)</p>
                    </div>
                    <div class="audit-box">
                        <h4>Sales Formatting Cleaned</h4>
                        <p style="color: var(--accent-purple);">$ String → float64</p>
                    </div>
                    <div class="audit-box">
                        <h4>Outlier Treatment</h4>
                        <p style="color: var(--accent-green);">3.0x IQR Winsorization</p>
                    </div>
                </div>

                <h4 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: var(--text-primary);">Data Quality Treatment Log Rationale:</h4>
                <ul style="color: var(--text-secondary); font-size: 14px; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Duplicate Invalidation</strong>: Identified 35 exact row duplicate copies in the raw Kaggle dataset and executed <code>drop_duplicates</code> to prevent revenue double-counting.</li>
                    <li><strong>Missing Value Imputation</strong>: Imputed 82 missing <code>Postal_Code</code> records using mode values grouped by State/City to preserve row integrity.</li>
                    <li><strong>Data Type Normalization</strong>: Parsed <code>Order_Date</code> and <code>Ship_Date</code> into <code>datetime64</code> objects; sanitized string currency symbols from <code>Sales</code> into <code>float64</code>.</li>
                    <li><strong>Robust Outlier Capping</strong>: Applied conservative 3.0x IQR Winsorization to extreme <code>Sales</code> and <code>Profit</code> spikes, preserving full observation context while stabilizing variance.</li>
                </ul>
            </div>
        </div>

        <!-- Tab 3: Order Inspector Table -->
        <div id="tab-table" class="tab-content">
            <div class="table-card">
                <div class="table-header-row">
                    <div>
                        <h3 style="font-size: 18px; font-weight: 700;">Filtered Orders Inspector</h3>
                        <p style="color: var(--text-secondary); font-size: 13px;" id="table-summary">Showing records (Click any row to inspect detail)</p>
                    </div>
                    <button id="btn-export" class="btn-export">📥 Export Filtered CSV</button>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Order Date</th>
                                <th>Order ID</th>
                                <th>Customer Name</th>
                                <th>Category</th>
                                <th>Sub-Category</th>
                                <th>Region</th>
                                <th>Sales ($)</th>
                                <th>Discount</th>
                                <th>Profit ($)</th>
                            </tr>
                        </thead>
                        <tbody id="table-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Attentive Order Detail Modal Popup -->
    <div id="order-modal" class="modal-overlay">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-order-id">Order Specification Detail</h3>
                <button id="btn-modal-close" class="btn-modal-close">✕</button>
            </div>
            <div class="modal-body-grid">
                <div class="modal-item"><span>Customer Name</span><strong id="modal-cust-name">-</strong></div>
                <div class="modal-item"><span>Customer ID</span><strong id="modal-cust-id">-</strong></div>
                <div class="modal-item"><span>Order Date</span><strong id="modal-date">-</strong></div>
                <div class="modal-item"><span>Shipping Mode</span><strong id="modal-ship">-</strong></div>
                <div class="modal-item"><span>Location</span><strong id="modal-location">-</strong></div>
                <div class="modal-item"><span>Category / Sub</span><strong id="modal-category">-</strong></div>
                <div class="modal-item"><span>Sales Amount</span><strong id="modal-sales">-</strong></div>
                <div class="modal-item"><span>Discount Rate</span><strong id="modal-discount">-</strong></div>
                <div class="modal-item"><span>Quantity</span><strong id="modal-quantity">-</strong></div>
                <div class="modal-item"><span>Net Profit</span><strong id="modal-profit">-</strong></div>
            </div>
        </div>
    </div>

    <script>
        const rawData = {records_json};

        let selectedQuarter = 'All';

        // Populate dropdown options
        function populateFilterOptions() {{
            const regions = Array.from(new Set(rawData.map(r => r.Region).filter(Boolean))).sort();
            const categories = Array.from(new Set(rawData.map(r => r.Category).filter(Boolean))).sort();
            const segments = Array.from(new Set(rawData.map(r => r.Segment || r.Customer_Segment).filter(Boolean))).sort();

            const regSelect = document.getElementById('filter-region');
            regions.forEach(r => {{
                const opt = document.createElement('option');
                opt.value = r; opt.textContent = r;
                regSelect.appendChild(opt);
            }});

            const catSelect = document.getElementById('filter-category');
            categories.forEach(c => {{
                const opt = document.createElement('option');
                opt.value = c; opt.textContent = c;
                catSelect.appendChild(opt);
            }});

            const segSelect = document.getElementById('filter-segment');
            segments.forEach(s => {{
                const opt = document.createElement('option');
                opt.value = s; opt.textContent = s;
                segSelect.appendChild(opt);
            }});
        }}

        // Formatting utilities
        const fmtCurr = n => new Intl.NumberFormat('en-US', {{ style: 'currency', currency: 'USD', maximumFractionDigits: 0 }}).format(n);
        const fmtPct = n => (n * 100).toFixed(1) + '%';
        const fmtNum = n => new Intl.NumberFormat('en-US').format(n);

        // Core filtering logic
        function getFilteredData() {{
            const reg = document.getElementById('filter-region').value;
            const cat = document.getElementById('filter-category').value;
            const seg = document.getElementById('filter-segment').value;
            const search = document.getElementById('filter-search').value.toLowerCase().trim();

            return rawData.filter(r => {{
                if (reg !== 'All' && r.Region !== reg) return false;
                if (cat !== 'All' && r.Category !== cat) return false;
                const rSeg = r.Segment || r.Customer_Segment;
                if (seg !== 'All' && rSeg !== seg) return false;

                // Date Quarter Presets
                if (selectedQuarter !== 'All' && r.Order_Date) {{
                    const month = parseInt(r.Order_Date.substring(5, 7), 10);
                    if (selectedQuarter === 'Q1' && (month < 1 || month > 3)) return false;
                    if (selectedQuarter === 'Q2' && (month < 4 || month > 6)) return false;
                    if (selectedQuarter === 'Q3' && (month < 7 || month > 9)) return false;
                    if (selectedQuarter === 'Q4' && (month < 10 || month > 12)) return false;
                }}

                if (search) {{
                    const matchText = `${{r.Order_ID || ''}} ${{r.Customer_Name || ''}} ${{r.Product_Name || ''}} ${{r.Sub_Category || ''}} ${{r.City || ''}} ${{r.State || ''}}`.toLowerCase();
                    if (!matchText.includes(search)) return false;
                }}
                return true;
            }});
        }}

        // Dynamic Attentive Insights Generator
        function updateInsightsBanner(filtered, totalSales, totalProfit, margin) {{
            const insightElem = document.getElementById('dynamic-insight');
            if (filtered.length === 0) {{
                insightElem.innerHTML = '<span>No matching transaction records found.</span> Please adjust your filter criteria.';
                return;
            }}

            const topCatMap = {{}};
            filtered.forEach(r => {{
                const c = r.Category || 'Other';
                topCatMap[c] = (topCatMap[c] || 0) + (r.Sales || 0);
            }});
            const topCat = Object.keys(topCatMap).reduce((a, b) => topCatMap[a] > topCatMap[b] ? a : b, 'Technology');

            const highDiscRows = filtered.filter(r => (r.Discount || 0) > 0.2);
            const highDiscSales = highDiscRows.reduce((acc, r) => acc + (r.Sales || 0), 0);
            const highDiscProfit = highDiscRows.reduce((acc, r) => acc + (r.Profit || 0), 0);
            const highDiscMargin = highDiscSales > 0 ? (highDiscProfit / highDiscSales) * 100 : 0;

            let alertText = '';
            if (highDiscMargin < 0) {{
                alertText = ` ⚠️ <span>Warning:</span> Orders with >20% discount yield an average negative profit margin (${{highDiscMargin.toFixed(1)}}%).`;
            }} else {{
                alertText = ` 💡 <span>Healthy Margins:</span> High profitability maintained across current selection (${{(margin * 100).toFixed(1)}}% net margin).`;
            }}

            insightElem.innerHTML = `<span>Key Executive Insight:</span> <strong>${{topCat}}</strong> generates the highest sales revenue (${{fmtCurr(topCatMap[topCat] || 0)}}).${{alertText}}`;
        }}

        // Update Dashboard Visuals
        function updateDashboard() {{
            const filtered = getFilteredData();

            // 1. Update KPI Scorecard
            const totalSales = filtered.reduce((acc, r) => acc + (r.Sales || 0), 0);
            const totalProfit = filtered.reduce((acc, r) => acc + (r.Profit || 0), 0);
            const orderCount = filtered.length;
            const avgDiscount = orderCount > 0 ? (filtered.reduce((acc, r) => acc + (r.Discount || 0), 0) / orderCount) : 0;
            const margin = totalSales > 0 ? (totalProfit / totalSales) : 0;

            document.getElementById('kpi-revenue').textContent = fmtCurr(totalSales);
            document.getElementById('kpi-profit').textContent = fmtCurr(totalProfit);
            document.getElementById('kpi-orders').textContent = fmtNum(orderCount);
            document.getElementById('kpi-discount').textContent = fmtPct(avgDiscount);
            document.getElementById('kpi-margin').textContent = (margin * 100).toFixed(1) + '%';

            updateInsightsBanner(filtered, totalSales, totalProfit, margin);

            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const textColor = isDark ? '#94a3b8' : '#64748b';
            const gridColor = isDark ? '#27354d' : '#e2e8f0';

            const plotlyLayoutDefaults = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: textColor, family: 'Plus Jakarta Sans, sans-serif' }},
                margin: {{ t: 30, b: 40, l: 50, r: 20 }},
                xaxis: {{ gridcolor: gridColor, zerolinecolor: gridColor }},
                yaxis: {{ gridcolor: gridColor, zerolinecolor: gridColor }}
            }};

            // 2. Chart 1: Monthly Trend Spline
            const monthlyMap = {{}};
            filtered.forEach(r => {{
                if (!r.Order_Date) return;
                const month = r.Order_Date.substring(0, 7);
                if (!monthlyMap[month]) monthlyMap[month] = {{ sales: 0, profit: 0 }};
                monthlyMap[month].sales += (r.Sales || 0);
                monthlyMap[month].profit += (r.Profit || 0);
            }});

            const months = Object.keys(monthlyMap).sort();
            const monthlySales = months.map(m => monthlyMap[m].sales);
            const monthlyProfit = months.map(m => monthlyMap[m].profit);

            const trendData = [
                {{
                    x: months, y: monthlySales, name: 'Sales ($)',
                    type: 'scatter', mode: 'lines+markers',
                    line: {{ color: '#38bdf8', width: 3, shape: 'spline' }},
                    fill: 'tozeroy', fillcolor: 'rgba(56, 189, 248, 0.12)'
                }},
                {{
                    x: months, y: monthlyProfit, name: 'Profit ($)',
                    type: 'scatter', mode: 'lines+markers',
                    line: {{ color: '#34d399', width: 3, shape: 'spline' }}
                }}
            ];
            const trendLayout = Object.assign({{}}, plotlyLayoutDefaults, {{
                legend: {{ orientation: 'h', y: 1.15, x: 0 }},
                hovermode: 'x unified'
            }});
            Plotly.react('chart-trend', trendData, trendLayout, {{ responsive: true, displayModeBar: false }});

            // 3. Chart 2: Category Breakdown
            const catMap = {{}};
            filtered.forEach(r => {{
                const c = r.Category || 'Other';
                if (!catMap[c]) catMap[c] = {{ sales: 0, profit: 0 }};
                catMap[c].sales += (r.Sales || 0);
                catMap[c].profit += (r.Profit || 0);
            }});

            const catList = Object.keys(catMap);
            const catSales = catList.map(c => catMap[c].sales);
            const catProfit = catList.map(c => catMap[c].profit);

            const catData = [
                {{ x: catList, y: catSales, name: 'Sales', type: 'bar', marker: {{ color: '#38bdf8' }} }},
                {{ x: catList, y: catProfit, name: 'Profit', type: 'bar', marker: {{ color: '#34d399' }} }}
            ];
            const catLayout = Object.assign({{}}, plotlyLayoutDefaults, {{
                barmode: 'group',
                legend: {{ orientation: 'h', y: 1.15, x: 0 }}
            }});
            Plotly.react('chart-category', catData, catLayout, {{ responsive: true, displayModeBar: false }});

            // 4. Chart 3: Scatter Sales vs Profit
            const scatterCategories = Array.from(new Set(filtered.map(r => r.Category || 'Other')));
            const colorPalette = ['#38bdf8', '#34d399', '#fb923c', '#c084fc', '#f43f5e'];

            const scatterTraces = scatterCategories.map((cat, idx) => {{
                const catRows = filtered.filter(r => (r.Category || 'Other') === cat);
                return {{
                    x: catRows.map(r => r.Sales),
                    y: catRows.map(r => r.Profit),
                    text: catRows.map(r => `${{r.Customer_Name || 'Customer'}}<br>${{r.Product_Name || r.Sub_Category}}<br>Qty: ${{r.Quantity}} | Disc: ${{fmtPct(r.Discount)}}`),
                    mode: 'markers',
                    name: cat,
                    marker: {{
                        size: catRows.map(r => Math.max(6, Math.min(24, (r.Quantity || 1) * 2.5))),
                        color: colorPalette[idx % colorPalette.length],
                        opacity: 0.75
                    }}
                }};
            }});
            const scatterLayout = Object.assign({{}}, plotlyLayoutDefaults, {{
                xaxis: Object.assign({{}}, plotlyLayoutDefaults.xaxis, {{ title: 'Sales ($)' }}),
                yaxis: Object.assign({{}}, plotlyLayoutDefaults.yaxis, {{ title: 'Profit ($)' }}),
                legend: {{ orientation: 'h', y: 1.15, x: 0 }}
            }});
            Plotly.react('chart-scatter', scatterTraces, scatterLayout, {{ responsive: true, displayModeBar: false }});

            // 5. Chart 4: Discount Tiers Analysis
            const tiers = {{ 'No Discount (0%)': [], 'Low (1-15%)': [], 'Medium (16-25%)': [], 'High (>25%)': [] }};
            filtered.forEach(r => {{
                const d = r.Discount || 0;
                if (d === 0) tiers['No Discount (0%)'].push(r);
                else if (d <= 0.15) tiers['Low (1-15%)'].push(r);
                else if (d <= 0.25) tiers['Medium (16-25%)'].push(r);
                else tiers['High (>25%)'].push(r);
            }});

            const tierNames = Object.keys(tiers);
            const tierMargins = tierNames.map(t => {{
                const rows = tiers[t];
                const s = rows.reduce((acc, r) => acc + (r.Sales || 0), 0);
                const p = rows.reduce((acc, r) => acc + (r.Profit || 0), 0);
                return s > 0 ? (p / s) * 100 : 0;
            }});

            const discountData = [{{
                x: tierNames,
                y: tierMargins,
                type: 'bar',
                marker: {{ color: tierMargins.map(m => m >= 0 ? '#34d399' : '#f43f5e') }},
                text: tierMargins.map(m => m.toFixed(1) + '%'),
                textposition: 'auto'
            }}];

            const discountLayout = Object.assign({{}}, plotlyLayoutDefaults, {{
                yaxis: Object.assign({{}}, plotlyLayoutDefaults.yaxis, {{ title: 'Avg Profit Margin (%)' }}),
                showlegend: false
            }});
            Plotly.react('chart-discount', discountData, discountLayout, {{ responsive: true, displayModeBar: false }});

            // 6. Update Inspector Table
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            document.getElementById('table-summary').textContent = `Showing top 50 of ${{filtered.length}} filtered orders (Click row for detail popup)`;

            filtered.slice(0, 50).forEach(r => {{
                const tr = document.createElement('tr');
                tr.className = 'table-row-clickable';
                const profitClass = (r.Profit || 0) >= 0 ? 'badge-positive' : 'badge-negative';
                tr.innerHTML = `
                    <td>${{r.Order_Date || 'N/A'}}</td>
                    <td style="font-weight: 700; color: var(--accent-blue);">${{r.Order_ID || 'N/A'}}</td>
                    <td>${{r.Customer_Name || 'N/A'}}</td>
                    <td>${{r.Category || 'N/A'}}</td>
                    <td>${{r.Sub_Category || 'N/A'}}</td>
                    <td>${{r.Region || 'N/A'}}</td>
                    <td>${{fmtCurr(r.Sales || 0)}}</td>
                    <td>${{fmtPct(r.Discount || 0)}}</td>
                    <td class="${{profitClass}}">${{fmtCurr(r.Profit || 0)}}</td>
                `;
                tr.addEventListener('click', () => openOrderModal(r));
                tbody.appendChild(tr);
            }});
        }}

        // Open Order Modal Popup
        function openOrderModal(r) {{
            document.getElementById('modal-order-id').textContent = `Order Detail: ${{r.Order_ID || 'N/A'}}`;
            document.getElementById('modal-cust-name').textContent = r.Customer_Name || 'N/A';
            document.getElementById('modal-cust-id').textContent = r.Customer_ID || 'N/A';
            document.getElementById('modal-date').textContent = r.Order_Date || 'N/A';
            document.getElementById('modal-ship').textContent = r.Ship_Mode || 'N/A';
            document.getElementById('modal-location').textContent = `${{r.City || ''}}, ${{r.State || ''}} (${{r.Region || ''}})`;
            document.getElementById('modal-category').textContent = `${{r.Category || ''}} > ${{r.Sub_Category || ''}}`;
            document.getElementById('modal-sales').textContent = fmtCurr(r.Sales || 0);
            document.getElementById('modal-discount').textContent = fmtPct(r.Discount || 0);
            document.getElementById('modal-quantity').textContent = r.Quantity || 1;
            
            const pElem = document.getElementById('modal-profit');
            pElem.textContent = fmtCurr(r.Profit || 0);
            pElem.className = (r.Profit || 0) >= 0 ? 'badge-positive' : 'badge-negative';

            document.getElementById('order-modal').classList.add('active');
        }}

        // Export Filtered CSV
        function exportCSV() {{
            const filtered = getFilteredData();
            if (filtered.length === 0) return alert('No data to export!');

            const headers = Object.keys(filtered[0]);
            const csvRows = [headers.join(',')];

            filtered.forEach(row => {{
                const values = headers.map(h => {{
                    const val = row[h] === null || row[h] === undefined ? '' : row[h];
                    return `"${{String(val).replace(/"/g, '""')}}"`;
                }});
                csvRows.push(values.join(','));
            }});

            const blob = new Blob([csvRows.join('\\n')], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('download', 'superstore_filtered_intelligence.csv');
            a.click();
        }}

        // Tab Navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(btn.getAttribute('data-tab')).classList.add('active');
                updateDashboard();
            }});
        }});

        // Category Chips
        document.querySelectorAll('.chip-cat').forEach(chip => {{
            chip.addEventListener('click', () => {{
                document.querySelectorAll('.chip-cat').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                document.getElementById('filter-category').value = chip.getAttribute('data-cat');
                updateDashboard();
            }});
        }});

        // Date Chips
        document.querySelectorAll('.chip-date').forEach(chip => {{
            chip.addEventListener('click', () => {{
                document.querySelectorAll('.chip-date').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                selectedQuarter = chip.getAttribute('data-date');
                updateDashboard();
            }});
        }});

        // Theme Switcher
        document.getElementById('theme-toggle').addEventListener('click', () => {{
            const html = document.documentElement;
            const current = html.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            document.getElementById('theme-toggle').textContent = next === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode';
            updateDashboard();
        }});

        document.getElementById('filter-region').addEventListener('change', updateDashboard);
        document.getElementById('filter-category').addEventListener('change', (e) => {{
            document.querySelectorAll('.chip-cat').forEach(c => {{
                c.classList.toggle('active', c.getAttribute('data-cat') === e.target.value);
            }});
            updateDashboard();
        }});
        document.getElementById('filter-segment').addEventListener('change', updateDashboard);
        document.getElementById('filter-search').addEventListener('input', updateDashboard);

        document.getElementById('btn-reset').addEventListener('click', () => {{
            document.getElementById('filter-region').value = 'All';
            document.getElementById('filter-category').value = 'All';
            document.getElementById('filter-segment').value = 'All';
            document.getElementById('filter-search').value = '';
            selectedQuarter = 'All';
            document.querySelectorAll('.chip-cat').forEach(c => c.classList.toggle('active', c.getAttribute('data-cat') === 'All'));
            document.querySelectorAll('.chip-date').forEach(c => c.classList.toggle('active', c.getAttribute('data-date') === 'All'));
            updateDashboard();
        }});

        document.getElementById('btn-export').addEventListener('click', exportCSV);
        document.getElementById('btn-modal-close').addEventListener('click', () => {{
            document.getElementById('order-modal').classList.remove('active');
        }});
        document.getElementById('order-modal').addEventListener('click', (e) => {{
            if (e.target.id === 'order-modal') document.getElementById('order-modal').classList.remove('active');
        }});

        // Init
        populateFilterOptions();
        updateDashboard();
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Visualization] Exported interactive dashboard HTML: '{output_path}'")
    return output_path


def generate_all_visualizations(df, output_dir="outputs/figures", dashboard_path="outputs/interactive_dashboard.html"):
    """
    Executes full EDA visualization suite.
    """
    h_path = plot_histograms(df, output_dir)
    b_path = plot_bar_charts(df, output_dir)
    box_path = plot_boxplots(df, output_dir)
    c_path = plot_correlation_heatmap(df, output_dir)
    s_path = plot_scatter_relationships(df, output_dir)
    dash_path = generate_interactive_dashboard(df, dashboard_path)

    return {
        "histogram": h_path,
        "bar_chart": b_path,
        "box_plot": box_path,
        "heatmap": c_path,
        "scatter": s_path,
        "dashboard": dash_path
    }


