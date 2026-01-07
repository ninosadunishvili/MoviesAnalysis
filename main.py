import os
import sys

# Add src to path if running from root
sys.path.append(os.path.join(os.getcwd(), "src"))

from src import (
    step1_import_dataset as step1,
    step2_data_cleaning as step2,
    step3_data_visualization as step3,
    step4_machine_learning as step4
)

def main():
    print("--- STARTING MOVIE DATA PIPELINE ---")
    
    # Step 1: Download
    try:
        step1.download_data()
    except Exception as e:
        print(f"Extraction Step failed: {e}. Ensure Kaggle API is configured.")

    # Step 2: Cleaning
    print("\n--- STEP 2: DATA CLEANING ---")
    step2.clean_data()

    # Step 3: Visualization
    print("\n--- STEP 3: DATA VISUALIZATION ---")
    step3.run_visualizations()

    # Step 4: Machine Learning
    print("\n--- STEP 4: MACHINE LEARNING ---")
    # File 04 already has the run_ml_pipeline function
    step4.run_ml_pipeline("data/processed/clean_movies.csv")

    print("\n--- PIPELINE COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()