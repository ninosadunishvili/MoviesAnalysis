import os
import pandas as pd

def clean_data():
    # CONFIG
    DATASET_PATH = "data/raw"
    OUTPUT_PATH = "data/processed"
    CSV_FILENAME = "Movies_dataset.csv"
    CURRENT_YEAR = 2026

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1. LOAD DATA
    # The Kaggle dataset usually has a specific name, adjust if necessary
    input_file = os.path.join(DATASET_PATH, CSV_FILENAME)
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    df = pd.read_csv(input_file)
    print(f"Initial shape: {df.shape}")

    # 2. CLEAN COLUMN NAMES
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 3. REMOVE DUPLICATES
    df = df.drop_duplicates()

    # 4. HANDLE MISSING VALUES
    # Numeric columns → median
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns → "Unknown"
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # 5. FIX DATA TYPES (Synchronizing with ML script needs)
    cols_to_fix = ["vote_average", "vote_count", "popularity", "revenue"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Re-fill numeric NaNs created by coercion
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 6. OUTLIER REMOVAL (IQR METHOD)
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]

    # We apply this to vote_average and vote_count to clean the distribution for ML
    for col in ["vote_average", "vote_count"]:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)

    # 7. FEATURE ENGINEERING
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
        df["movie_age"] = CURRENT_YEAR - df["year"]

    if "vote_average" in df.columns:
        df["rating_category"] = pd.cut(
            df["vote_average"],
            bins=[0, 6, 7.5, 10],
            labels=["Low", "Medium", "High"]
        )

    # 8. SAVE CLEAN DATA
    output_file = os.path.join(OUTPUT_PATH, "clean_movies.csv")
    df.to_csv(output_file, index=False)
    print(f"Final shape: {df.shape}")
    print(f"Clean dataset saved to {output_file}")

if __name__ == "__main__":
    clean_data()