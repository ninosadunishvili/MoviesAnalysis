Data Cleaning & Preprocessing

The raw movie dataset was cleaned and prepared to ensure accuracy and consistency before analysis and modeling.
Initial inspection was performed to understand the dataset structure, data types, missing values, and duplicates.

Column names were standardized by converting them to lowercase and replacing spaces with underscores. Duplicate records were removed to prevent biased analysis.

Missing values were handled based on data type:

Numerical columns were filled using the median to reduce the impact of outliers.

Categorical columns were filled with "Unknown" to preserve records.

Key columns such as release year, ratings, and vote counts were converted to numeric types where necessary. Outliers in selected numerical features were detected and removed using the Interquartile Range (IQR) method.

Additional features were created, including movie age and rating category, to enhance analysis and modeling.

The final cleaned dataset is saved as:

data/processed/clean_movies.csv


and is used for all subsequent exploratory analysis and machine learning steps.