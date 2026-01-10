import os
import sys

# Ensure src is in the path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src import step1_import_dataset as step1
from src import step2_data_cleaning as step2
from src import step3_data_visualization as step3
from src import step4_machine_learning as step4

def main():
    """
    Main entry point for the End-to-End Data Science Pipeline.
    Coordinates data acquisition, cleaning, EDA, and Machine Learning.
    """
    print("=== MOVIE INSIGHTS PROJECT PIPELINE ===")

    # 1. DATA ACQUISITION
    try:
        print("\n[Stage 1: Downloading Data]")
        step1.download_data()
    except Exception as e:
        print(f"Skipping Stage 1: {e}")

    # 2. CLEANING & QUALITY REPORT
    try:
        print("\n[Stage 2: Data Cleaning & Quality Reporting]")
        step2.clean_data()
    except Exception as e:
        print(f"CRITICAL ERROR in Stage 2: {e}")
        return # Stop if cleaning fails

    # 3. VISUALIZATION
    try:
        print("\n[Stage 3: Exploratory Data Analysis]")
        step3.run_visualizations()
    except Exception as e:
        print(f"Stage 3 Warning: {e}")

    # 4. MACHINE LEARNING
    try:
        print("\n[Stage 4: Machine Learning Implementation]")
        step4.run_ml_pipeline("data/processed/clean_movies.csv")
    except Exception as e:
        print(f"Stage 4 Warning: {e}")

    print("\n=== PIPELINE RUN FINISHED ===")

if __name__ == "__main__":
    main()