import os
import pandas as pd
import numpy as np

def clean_data():
    """
    Performs end-to-end data cleaning, tracks quality metrics, and generates a report.
    """
    # CONFIG
    RAW_PATH = "data/raw"
    PROC_PATH = "data/processed"
    REP_PATH = "reports/data_quality"
    CSV_FILENAME = "Movies_dataset.csv"
    os.makedirs(PROC_PATH, exist_ok=True)
    os.makedirs(REP_PATH, exist_ok=True)

    input_file = os.path.join(RAW_PATH, CSV_FILENAME)
    
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Source file {input_file} not found.")
        
        df = pd.read_csv(input_file)
        
        # --- DATA QUALITY TRACKING (Initial State) ---
        report = []
        report.append("=== DATA QUALITY & CLEANING REPORT ===")
        report.append(f"Initial Shape: {df.shape}")
        report.append(f"Initial Missing Values:\n{df.isnull().sum().to_string()}")
        report.append(f"Initial Duplicates: {df.duplicated().sum()}")

        # 1. Clean Column Names (FIXED LINE BELOW)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # 2. Remove Duplicates
        df = df.drop_duplicates()

        # 3. Handle Missing Values
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
            
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")

        # 4. Outlier Removal (IQR Method)
        outlier_count = 0
        for col in ["vote_average", "vote_count"]:
            if col in df.columns:
                # Ensure column is numeric before IQR
                df[col] = pd.to_numeric(df[col], errors='coerce')
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                before_count = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                outlier_count += (before_count - len(df))

        # 5. Feature Engineering
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
            df["year"] = df["release_date"].dt.year
            df["movie_age"] = 2026 - df["year"]
            
        if "vote_average" in df.columns:
            df["rating_category"] = pd.cut(
                df["vote_average"], bins=[0, 6, 7.5, 10], labels=["Low", "Medium", "High"]
            )

        # --- DATA QUALITY TRACKING (Final State) ---
        report.append("\n=== CLEANING STEPS APPLIED ===")
        report.append("- Normalized column names to snake_case using .str accessor.")
        report.append("- Imputed numeric missing values with Median.")
        report.append("- Imputed categorical missing values with 'Unknown'.")
        report.append(f"- Removed {outlier_count} outliers using IQR method.")
        report.append(f"\nFinal Shape: {df.shape}")
        report.append(f"Final Missing Values: {df.isnull().sum().sum()}")

        # Save Report
        with open(os.path.join(REP_PATH, "data_quality_report.txt"), "w") as f:
            f.write("\n".join(report))

        # Save Clean Data
        output_file = os.path.join(PROC_PATH, "clean_movies.csv")
        df.to_csv(output_file, index=False)
        print(f"Success: Data Quality Report generated in {REP_PATH}")
        return df

    except Exception as e:
        print(f"Error during data cleaning: {e}")
        raise