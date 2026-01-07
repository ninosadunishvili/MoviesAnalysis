import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_visualizations():
    DATA_PATH = "data/processed/clean_movies.csv"
    FIGURES_PATH = "reports/figures"
    os.makedirs(FIGURES_PATH, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Clean data not found. Please run cleaning script first.")
        return

    df = pd.read_csv(DATA_PATH)

    # VIS 1: Rating distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["vote_average"], bins=20, kde=True, color='skyblue')
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating (Vote Average)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "rating_distribution.png"))
    plt.close()

    # VIS 2: Average rating by release year
    if "year" in df.columns:
        yearly_avg = df.groupby("year")["vote_average"].mean().sort_index()
        plt.figure(figsize=(10, 5))
        plt.plot(yearly_avg.index, yearly_avg.values, marker="o", color='green')
        plt.title("Average Movie Rating by Release Year")
        plt.xlabel("Year")
        plt.ylabel("Avg Rating")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, "avg_rating_by_year.png"))
        plt.close()

    # VIS 3: Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Movie Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "correlation_heatmap.png"))
    plt.close()

    print(f"Visualizations successfully saved to {FIGURES_PATH}")

if __name__ == "__main__":
    run_visualizations()