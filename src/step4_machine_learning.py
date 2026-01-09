# run_clustering_and_simple_classification.py
"""
Combined pipeline:
  - Clustering A: KMeans on (vote_average, movie_age), k=4
  - Simple classification: quantile-based rating classes (low/medium/high)
    trained with DecisionTree on (vote_count_log, movie_age)

Saves plots & summaries to reports/figures/
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "data/processed/clean_movies.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def run_ml_pipeline(input_csv: str = DATA_PATH, random_state: int = 42):
    # -----------------------
    # LOAD & FEATURE ENGINEERING
    # -----------------------
    df = pd.read_csv(input_csv)

    # movie_age: numeric years since release
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    current_year = datetime.now().year
    df["movie_age"] = current_year - df["release_date"].dt.year

    # -----------------------
    # PART 1: KMEANS CLUSTERING (vote_average + movie_age)
    # -----------------------
    df_clust = df.dropna(subset=["vote_average", "movie_age"]).copy()
    if df_clust.shape[0] > 0:
        X_clust = df_clust[["vote_average", "movie_age"]].values

        scaler = StandardScaler()
        X_clust_scaled = scaler.fit_transform(X_clust)

        k = 4
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        df_clust["cluster"] = kmeans.fit_predict(X_clust_scaled)

        # silhouette (only valid when >1 cluster)
        sil = silhouette_score(X_clust_scaled, df_clust["cluster"]) if k > 1 else None
        print(f"KMeans (k={k}) silhouette score: {sil if sil is not None else 'N/A'}")

        # Plot: movie_age vs vote_average colored by cluster
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_clust, x="movie_age", y="vote_average", hue="cluster", palette="tab10", s=30, edgecolor="k", linewidth=0.2)
        plt.title("KMeans Clusters: Movie Age vs Vote Average")
        plt.xlabel("Movie Age (years)")
        plt.ylabel("Vote Average")
        plt.legend(title="cluster", loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "clusters_age_rating.png"))
        plt.show()
    else:
        print("No data available for clustering (missing vote_average or movie_age).")

    # -----------------------
    # PART 2: SIMPLE CLASSIFICATION (quantile classes)
    # -----------------------
    # Need vote_average, vote_count, movie_age for this simple classifier
    df_clf = df.dropna(subset=["vote_average", "vote_count", "movie_age"]).copy()
    if df_clf.shape[0] < 50:
        print("Not enough data for classification after dropping missing values.")
        return

    # Create balanced classes using quantiles (3 classes: low/medium/high)
    # duplicates='drop' prevents errors if many identical values at edges
    df_clf["rating_class"] = pd.qcut(df_clf["vote_average"], q=3, labels=["low", "medium", "high"], duplicates="drop")

    # If qcut dropped duplicate bins and produced fewer classes, inform user
    classes_present = df_clf["rating_class"].unique().tolist()
    print("\nRating classes created (counts):")
    print(df_clf["rating_class"].value_counts())

    # Features: vote_count_log and movie_age (do NOT use vote_average to avoid leakage)
    df_clf["vote_count_log"] = np.log1p(df_clf["vote_count"].astype(float))
    feature_cols = ["vote_count_log", "movie_age"]
    X = df_clf[feature_cols].values
    y = df_clf["rating_class"].values

    # Train/test split (stratify by class to keep balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nDecision Tree Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    labels_order = ["low", "medium", "high"]
    # keep only labels that exist in y_test
    labels_in_test = [l for l in labels_order if l in y_test]
    cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_in_test, yticklabels=labels_in_test)
    plt.title("Confusion Matrix: Rating Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rating_classification_confusion_matrix.png"))
    plt.show()

    # Feature importances (from Decision Tree)
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nFeature importances:")
        print(fi.round(4))
        plt.figure(figsize=(5, 2.5))
        sns.barplot(x=fi.values, y=fi.index)
        plt.title("Decision Tree Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "rating_feature_importances.png"))
        plt.show()

    # Save a small summary CSV
    summary_df = pd.DataFrame({
        "metric": ["accuracy"],
        "value": [acc]
    })
    summary_df.to_csv(os.path.join(OUT_DIR, "rating_classification_summary.csv"), index=False)
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    run_ml_pipeline(DATA_PATH)
