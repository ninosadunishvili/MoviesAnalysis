import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_visualizations():
    """
    Generates 6 mandatory visualizations for EDA and saves them to reports/figures.

    Parameters:
    -----------
    None (Loads clean_movies.csv from data/processed)

    Returns:
    --------
    None (Saves .png files to disk)

    Raises:
    -------
    IOError
        If the cleaned data file cannot be read or output directory is inaccessible.
    """
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

    # VIS 4: Count Plot (Volume of Top Rated Movies)
    # Insight: Shows the proportion of movies that actually reach 'High' status.
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="rating_category", hue="rating_category", palette="magma", legend=False)
    plt.title("Count of Movies by Rating Category")
    plt.xlabel("Rating Category")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "rating_category_counts.png"))
    plt.close()

    # VIS 5: Scatter Plot with Trend Line (Rating vs Vote Count)
    # Insight: Do the highest-rated movies have a strong consensus (many votes)?
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x="vote_average", y="vote_count",
                scatter_kws={'alpha':0.3, 'color':'teal'},
                line_kws={'color':'red'})
    plt.title("Movie Rating vs. Vote Count (Consensus)")
    plt.xlabel("Vote Average")
    plt.ylabel("Number of Votes")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "rating_vs_votes_trend.png"))
    plt.close()

    # VIS 6: Violin Plot (Movie Age by Rating Category)
    # Insight: Visualizes if 'High' rated movies are typically older classics or newer releases.
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="rating_category", y="movie_age", hue="rating_category",
                   palette="Set2", legend=False)
    plt.title("Distribution of Movie Age across Rating Categories")
    plt.xlabel("Rating Category")
    plt.ylabel("Years Since Release")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "age_rating_violin.png"))
    plt.close()


    print(f"6 Visualizations successfully saved to {FIGURES_PATH}")

if __name__ == "__main__":
    run_visualizations()