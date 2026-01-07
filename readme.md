# Top-Rated Movie Insights: An End-to-End Analysis
**Course:** Data Science with Python  
**Team Members:** Gujabidze Lizi, Sadunishvili Nino

---

## 1. Problem Statement & Objectives
The film industry is a multi-billion dollar business where audience perception is key. This project aims to analyze the factors that contribute to a movie's "success" as defined by its audience rating.

**Primary Objectives:**
*   Clean and preprocess a raw dataset of top-rated movies.
*   Identify correlations between popularity, vote counts, and final ratings.
*   Build and compare machine learning models to predict a movie's average score based on engagement metrics.

## 2. Dataset Description
The data used in this project is the **Top Rated Movies Dataset**, sourced from Kaggle via the `kagglehub` library.
*   **Source:** [Kaggle - mohsin31202/top-rated-movies-dataset](https://www.kaggle.com/datasets/mohsin31202/top-rated-movies-dataset)
*   **Features:** Includes `original_title`, `original_language`, `genre`, `popularity`, `vote_count`, `vote_average`, and `release_date`.

## 3. Installation & Setup
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ninosadunishvili/MoviesAnalysis.git
   cd MovieAnalysis
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Acquire the data:**
Run the import script to download the raw CSV:
    ```bash
    python 01_import_dataset.py
    ```

## 4. Repository Structure
   Following the course guidelines:
```text
   project-name/
   |-- data/
   |   |-- raw/          # Original, immutable data
   |   +-- processed/    # Cleaned data (clean_movies.csv)
   |-- src/              # Source code for cleaning and models
   |-- reports/          # Generated visualizations and metrics
   |-- README.md
   |-- requirements.txt
   +-- .gitignore
```

## 5. Usage
Data Import: Run 01_import_dataset.py to fetch the data.
   
Cleaning: Run 02_data_cleaning.py to handle missing values, outliers, and feature engineering.
   
Visualization: Run 03_data_visualization.py to generate statistical plots.
   
Machine Learning: Run 04_machine_learning.py to train and evaluate models.

## 6. Results Summary
   (To be completed upon final analysis)
   EDA Findings: Initial analysis shows [e.g., a strong correlation between vote count and rating].
   ML Performance: Compared Linear Regression and Decision Tree models. The [Model Name] performed best with an R-squared of [X.XX].
---
