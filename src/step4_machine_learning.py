import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def run_ml_pipeline(input_csv, top_n_genres=8, random_state=42):
    """
    Pipeline:
      - Clustering A: KMeans on vote_average + movie_age (k=4)
      - Regression (Method 2): predict vote_average using numeric + genre dummies
        Models: LinearRegression, Ridge, RandomForest, (XGBoost if available)
      Outputs: metrics CSV + plots saved under reports/figures/
    """
    os.makedirs("reports/figures", exist_ok=True)

    # LOAD
    df = pd.read_csv(input_csv)

    # FEATURE: movie_age
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    current_year = datetime.now().year
    df["movie_age"] = current_year - df["release_date"].dt.year

    # -------------------------
    # PART 1: CLUSTERING A
    # -------------------------
    df_clust = df.dropna(subset=["vote_average", "movie_age"]).copy()
    X_clust = df_clust[["vote_average", "movie_age"]].values

    scaler_clust = StandardScaler()
    X_clust_scaled = scaler_clust.fit_transform(X_clust)

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    df_clust["cluster"] = kmeans.fit_predict(X_clust_scaled)

    # silhouette may fail if cluster count is 1; assume k>1 here
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X_clust_scaled, df_clust["cluster"])
    print(f"KMeans (k={k}) silhouette score: {sil:.4f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_clust, x="movie_age", y="vote_average", hue="cluster", palette="tab10", s=40)
    plt.title("KMeans Clusters: Movie Age vs Vote Average")
    plt.xlabel("Movie Age (years)")
    plt.ylabel("Vote Average")
    plt.tight_layout()
    plt.savefig("reports/figures/clusters_age_rating.png")
    plt.show()

    # -------------------------
    # PART 2: REGRESSION
    # -------------------------
    # Candidate features:
    # - vote_count_log
    # - movie_age
    # - popularity (if present)
    # - runtime (if present)
    # - top-N genre dummies

    # Prepare base dataframe: need target and vote_count
    df_reg = df.copy()
    df_reg = df_reg.dropna(subset=["vote_average", "vote_count", "movie_age"]).copy()

    # Numeric features
    df_reg["vote_count_log"] = np.log1p(df_reg["vote_count"].astype(float))

    numeric_features = ["vote_count_log", "movie_age"]
    if "popularity" in df_reg.columns:
        numeric_features.append("popularity")
        # ensure float
        df_reg["popularity"] = pd.to_numeric(df_reg["popularity"], errors="coerce")
    if "runtime" in df_reg.columns:
        numeric_features.append("runtime")
        df_reg["runtime"] = pd.to_numeric(df_reg["runtime"], errors="coerce")

    # Genre one-hot (top N)
    genres_series = df_reg["genre"].fillna("").astype(str).apply(lambda s: [g.strip() for g in s.split(",") if g.strip() != ""])
    all_genres = pd.Series([g for sub in genres_series for g in sub])
    if len(all_genres) > 0:
        top_genres = list(all_genres.value_counts().head(top_n_genres).index)
    else:
        top_genres = []

    def keep_top_genres(gen_list):
        return [g for g in gen_list if g in top_genres]

    if top_genres:
        from sklearn.preprocessing import MultiLabelBinarizer
        genre_lists = genres_series.apply(keep_top_genres)
        mlb = MultiLabelBinarizer(classes=top_genres)
        genre_dummies = pd.DataFrame(mlb.fit_transform(genre_lists),
                                     columns=[f"genre__{c}" for c in mlb.classes_],
                                     index=df_reg.index)
    else:
        genre_dummies = pd.DataFrame(index=df_reg.index)

    # Combine features
    X_df = pd.concat([df_reg[numeric_features].reset_index(drop=True),
                      genre_dummies.reset_index(drop=True)], axis=1)
    y = df_reg["vote_average"].reset_index(drop=True)

    # Drop rows with any remaining NaNs in X_df
    valid_idx = X_df.dropna().index
    X_df = X_df.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=random_state
    )

    # Scale numeric columns (keep dummies as-is)
    num_cols = [c for c in X_df.columns if c in numeric_features]
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    X_train_prepared = np.hstack([X_train_num, X_train.drop(columns=num_cols).values]) if X_train.shape[1] > len(num_cols) else X_train_num
    X_test_prepared = np.hstack([X_test_num, X_test.drop(columns=num_cols).values]) if X_test.shape[1] > len(num_cols) else X_test_num

    # Models to run
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state)
    }

    # Add XGBoost if available
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0, objective="reg:squarederror")

    results = []
    for name, model in models.items():
        model.fit(X_train_prepared, y_train)
        preds = model.predict(X_test_prepared)
        rmse = sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        print(f"\n{name} -- RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Save a scatter plot (pred vs actual) for the RandomForest and XGBoost and Linear (all)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, preds, s=20, alpha=0.6)
        # perfect line
        minv = min(y_test.min(), preds.min())
        maxv = max(y_test.max(), preds.max())
        plt.plot([minv, maxv], [minv, maxv], color="red", linewidth=1)
        plt.xlabel("Actual vote_average")
        plt.ylabel("Predicted vote_average")
        plt.title(f"{name}: Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(f"reports/figures/{name}_actual_vs_predicted.png")
        plt.show()

        # For tree-based models, plot feature importances (if available)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_names = num_cols + [c for c in X_df.columns if c not in num_cols]
            fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
            plt.figure(figsize=(6, max(3, 0.3 * len(fi))))
            sns.barplot(x=fi.values, y=fi.index)
            plt.title(f"{name} Feature Importances")
            plt.tight_layout()
            plt.savefig(f"reports/figures/{name}_feature_importances.png")
            plt.show()

    # Save results summary
    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv("reports/figures/regression_results_summary.csv", index=False)
    print("\nSaved regression summary to reports/figures/regression_results_summary.csv")
    print(results_df)

if __name__ == "__main__":
    run_ml_pipeline("data/processed/clean_movies.csv")
