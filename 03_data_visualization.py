import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
DATA_PATH = "data/processed/clean_movies.csv"
FIGURES_PATH = "reports/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

# ===============================
# LOAD CLEAN DATA
# ===============================
df = pd.read_csv(DATA_PATH)

# ===============================
# VISUALIZATION 1: RATING DISTRIBUTION
# ===============================
plt.figure(figsize=(8, 5))
plt.hist(df["vote_average"], bins=20)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "rating_distribution.png"))
plt.show()

# ===============================
# VISUALIZATION 2: RATING VS VOTES
# ===============================
plt.figure(figsize=(8, 5))
plt.scatter(df["vote_count"], df["vote_average"], alpha=0.6, s=10)

# Linear trendline
coefficients = np.polyfit(df["vote_count"], df["vote_average"], 1)
x_vals = np.linspace(df["vote_count"].min(), df["vote_count"].max(), 200)
y_vals = np.polyval(coefficients, x_vals)
plt.plot(x_vals, y_vals)

plt.title("Rating vs Number of Votes")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "rating_vs_votes.png"))
plt.show()

# ===============================
# VISUALIZATION 3: CORRELATION HEATMAP
# ===============================
numeric_df = df.select_dtypes(include="number")
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix)
plt.colorbar(label="Correlation")

plt.xticks(
    ticks=range(len(numeric_df.columns)),
    labels=numeric_df.columns,
    rotation=90
)
plt.yticks(
    ticks=range(len(numeric_df.columns)),
    labels=numeric_df.columns
)

plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "correlation_heatmap.png"))
plt.show()
