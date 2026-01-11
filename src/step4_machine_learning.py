import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
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


def run_ml_pipeline(input_csv: str = DATA_PATH, random_state: int = 42, top_n_genres: int = 8):
    """
    Executes KMeans clustering and Decision Tree classification.
    Produces evaluation metrics and confusion matrix plots.

    Parameters:
    -----------
    input_csv : str
        Path to the cleaned dataset.
    random_state : int, optional
        Seed for reproducibility (default is 42).
    top_n_genres : int, optional
        Number of top genres to one-hot encode for the classifier.

    Returns:
    --------
    dict
        A dictionary containing key model metrics (e.g., accuracy, silhouette).
    """
    try:
        
        # LOAD & FEATURE ENGINEERING
        
        df = pd.read_csv(input_csv)
        if len(df) < 50:
            raise ValueError("Insufficient data for ML training.")

        # movie_age: numeric years since release
        df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
        current_year = datetime.now().year
        df["movie_age"] = current_year - df["release_date"].dt.year

        
        # PART 1: KMEANS CLUSTERING (vote_average + movie_age)
        
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
            sns.scatterplot(
                data=df_clust,
                x="movie_age",
                y="vote_average",
                hue="cluster",
                palette="tab10",
                s=30,
                linewidth=0.2,
            )
            plt.title("KMeans Clusters: Movie Age vs Vote Average")
            plt.xlabel("Movie Age (years)")
            plt.ylabel("Vote Average")
            plt.legend(title="cluster", loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "clusters_age_rating.png"))
            plt.show()
        else:
            print("No data available for clustering (missing vote_average or movie_age).")
            sil = None

        
        # PART 2: SIMPLE CLASSIFICATION (quantile classes) WITH GENRES
        
        # Need vote_average, vote_count, movie_age for this classifier
        df_clf = df.dropna(subset=["vote_average", "vote_count", "movie_age"]).copy()
        if df_clf.shape[0] < 50:
            print("Not enough data for classification after dropping missing values.")
            return None

        # Create balanced classes using quantiles (3 classes: low/medium/high)
        df_clf["rating_class"] = pd.qcut(
            df_clf["vote_average"], q=3, labels=["low", "medium", "high"], duplicates="drop"
        )

        # If qcut dropped duplicate bins and produced fewer classes, inform user
        print("\nRating classes created (counts):")
        print(df_clf["rating_class"].value_counts())

        # Features: vote_count_log and movie_age (do NOT use vote_average to avoid leakage)
        df_clf["vote_count_log"] = np.log1p(df_clf["vote_count"].astype(float))
        X_num = df_clf[["vote_count_log", "movie_age"]]

        
        # GENRE ENCODING (TOP N)
        # Parse genres into lists, handle missing
        genres_series = (
            df_clf["genre"].fillna("").astype(str).apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])
        )

        # Find top N genres (frequency)
        all_genres = pd.Series([g for sub in genres_series for g in sub])
        if len(all_genres) > 0:
            top_genres = all_genres.value_counts().head(top_n_genres).index.tolist()
        else:
            top_genres = []

        if top_genres:
            mlb = MultiLabelBinarizer(classes=top_genres)
            genre_dummies = pd.DataFrame(
                mlb.fit_transform(genres_series.apply(lambda g: [x for x in g if x in top_genres])),
                columns=[f"genre_{g}" for g in mlb.classes_],
                index=df_clf.index,
            )
        else:
            genre_dummies = pd.DataFrame(index=df_clf.index)

        # Combine features
        X = pd.concat([X_num.reset_index(drop=True), genre_dummies.reset_index(drop=True)], axis=1)
        y = df_clf["rating_class"].reset_index(drop=True)

        print(f"\nClassifier features: {list(X.columns)}")

        # Drop rows with any NaNs in X
        valid_idx = X.dropna().index
        X = X.loc[valid_idx].reset_index(drop=True)
        y = y.loc[valid_idx].reset_index(drop=True)

        if X.shape[0] < 30:
            print("Not enough data after preprocessing to train.")
            return None

        # Train/test split (stratify by class to keep balance)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

        # Train Decision Tree
        clf = DecisionTreeClassifier(max_depth=5, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        print(f"\nDecision Tree Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        labels_order = ["low", "medium", "high"]
        labels_in_test = [l for l in labels_order if l in y_test.values]
        cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_in_test, yticklabels=labels_in_test)
        plt.title("Confusion Matrix: Rating Classification")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "rating_classification_confusion_matrix.png"))
        plt.show()

        # Feature importances (from Decision Tree)
        if hasattr(clf, "feature_importances_"):
            fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("\nFeature importances:")
            print(fi.round(4))
            plt.figure(figsize=(6, max(2.5, 0.25 * len(fi))))
            sns.barplot(x=fi.values, y=fi.index)
            plt.title("Decision Tree Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "rating_feature_importances.png"))
            plt.show()

        # Save a small summary CSV
        summary_df = pd.DataFrame({"metric": ["accuracy", "silhouette"], "value": [acc, sil if sil is not None else np.nan]})
        summary_df.to_csv(os.path.join(OUT_DIR, "rating_classification_summary.csv"), index=False)
        print(f"\nSaved outputs to {OUT_DIR}")

        # Return metrics for programmatic use
        return {"accuracy": float(acc), "silhouette": float(sil) if sil is not None else None}

    except Exception as e:
        print(f"ML Pipeline Error: {e}")
        return None


if __name__ == "__main__":
    run_ml_pipeline(DATA_PATH)
