import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, confusion_matrix

def run_ml_pipeline(input_csv):
    """
    Implements Clustering to group similar movies and Classification 
    to predict high-rated films.
    """
    # 1. LOAD DATA
    df = pd.read_csv(input_csv)
    
    # 2. PREPROCESSING & SCALING
    # Scaling is mandatory for K-Means (Clustering) to perform correctly
    features = ['popularity', 'vote_count']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # --- TASK 1: CLUSTERING (K-MEANS) ---
    # Technical Req 2.3.2: Grouping similar data
    print("Running K-Means Clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Evaluation for Clustering (Requirement 2.3.3)
    sil_score = silhouette_score(X_scaled, df['cluster'])
    print(f"Clustering Silhouette Score: {sil_score:.4f}")

    # Visualization of Clusters (Requirement 2.2.2 - Multivariate)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='popularity', y='vote_count', hue='cluster', palette='viridis')
    plt.title("Movie Clusters based on Popularity and Vote Count")
    plt.savefig("reports/figures/clustering_plot.png")
    plt.show()

    # --- TASK 2: CLASSIFICATION (DECISION TREE) ---
    # Goal: Predict if a movie is "Top Tier" (Rating > 8.0)
    # Technical Req 2.3.2: Predicting categories
    print("\nRunning Classification...")
    df['is_top_tier'] = (df['vote_average'] >= 8.2).astype(int)
    
    X = df[['popularity', 'vote_count', 'cluster']] # Use cluster as a feature!
    y = df['is_top_tier']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    # 3. EVALUATION METRICS (Requirement 2.3.3)
    print("--- Classification Results ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix (Requirement 4.2 - Excellent Level)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    # Ensure the directory for reports exists
    import os
    os.makedirs("reports/figures/", exist_ok=True)
    run_ml_pipeline("data/processed/clean_movies.csv")