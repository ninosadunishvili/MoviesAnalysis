import os
import pandas as pd

 
# CONFIG
DATASET_PATH = "C:/Users/nagli/.cache/kagglehub/datasets/mohsin31202/top-rated-movies-dataset/versions/2"   # <-- CHANGE THIS
OUTPUT_PATH = "data/processed"
CSV_FILENAME = "Movies_dataset.csv"
CURRENT_YEAR = 2026

os.makedirs(OUTPUT_PATH, exist_ok=True)

 
# 1. LOAD DATA
df = pd.read_csv(os.path.join(DATASET_PATH, CSV_FILENAME))
 
# 2. INITIAL INSPECTION
print("Initial shape:", df.shape)
print(df.info())
print("Count of Null values before cleaning \n", df.isna().sum())

 
# 3. CLEAN COLUMN NAMES
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

 
# 4. REMOVE DUPLICATES
df = df.drop_duplicates()
 
# 5. HANDLE MISSING VALUES
# Numeric columns → median
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns → "Unknown"
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

 
# 6. FIX DATA TYPES
if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

if "votes" in df.columns:
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

# Re-fill numeric NaNs created by coercion
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

 
# 7. OUTLIER REMOVAL (IQR METHOD)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in ["rating", "votes"]:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

 
# 8. FEATURE ENGINEERING
if "year" in df.columns:
    df["movie_age"] = CURRENT_YEAR - df["year"]

if "rating" in df.columns:
    df["rating_category"] = pd.cut(
        df["rating"],
        bins=[0, 6, 7.5, 10],
        labels=["Low", "Medium", "High"]
    )

 
# 9. FINAL CHECK
print("Final shape:", df.shape)
print("Final Count of Null values \n", df.isna().sum())

 
# 10. SAVE CLEAN DATA
df.to_csv(
    os.path.join(OUTPUT_PATH, "clean_movies.csv"),
    index=False
)
print("Clean dataset saved to data/processed/clean_movies.csv")
