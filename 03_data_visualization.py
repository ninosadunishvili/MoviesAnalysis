import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/clean_movies.csv"
FIGURES_PATH = "reports/figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# VIS 1: Rating distribution
plt.figure(figsize=(8, 5))
plt.hist(df["vote_average"].dropna(), bins=20)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "rating_distribution.png"))
plt.show()

# VIS 2: Average rating by release year
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year

yearly_avg_rating = (
    df.dropna(subset=["release_year", "vote_average"])
      .groupby("release_year")["vote_average"]
      .mean()
)

plt.figure(figsize=(9, 5))
plt.plot(yearly_avg_rating.index, yearly_avg_rating.values, marker="o")
plt.title("Average Movie Rating by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "avg_rating_by_year.png"))
plt.show()

# VIS 3: Correlation heatmap of numeric features
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(ticks=range(len(numeric_df.columns)), labels=numeric_df.columns, rotation=90)
plt.yticks(ticks=range(len(numeric_df.columns)), labels=numeric_df.columns)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "correlation_heatmap.png"))
plt.show()

stats = df["vote_average"].describe()
print(stats)

q1 = stats["25%"]
q3 = stats["75%"]
iqr = q3 - q1

outliers = df[
    (df["vote_average"] < q1 - 1.5 * iqr) |
    (df["vote_average"] > q3 + 1.5 * iqr)
]

print("Number of rating outliers:", len(outliers))
