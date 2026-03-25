import os
import pandas as pd
import numpy as np


# --------------------------------------------------
# Safe Folder Creation
# --------------------------------------------------
def ensure_folder(path):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------
# Load Processed Data
# --------------------------------------------------
def load_processed_data():

    path = "data/online_retail_customer_churn.csv"

    if not os.path.exists(path):
        raise FileNotFoundError("Processed data not found. Run data_preprocessing.py first.")

    df = pd.read_csv(path)
    print("Processed dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
def engineer_features(df):

    print("\nStarting Feature Engineering...")

    df = df.copy()

    # Example 1: Total Spending (if related columns exist)
    spending_cols = [col for col in df.columns if "spend" in col.lower() or "amount" in col.lower()]

    if len(spending_cols) > 1:
        df["Total_Spending"] = df[spending_cols].sum(axis=1)
        print("Created: Total_Spending")

    # Example 2: Average Value per Feature
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        df["Numeric_Mean"] = df[numeric_cols].mean(axis=1)
        print("Created: Numeric_Mean")

    # Example 3: Interaction Feature (first two numeric columns)
    if len(numeric_cols) >= 2:
        col1 = numeric_cols[0]
        col2 = numeric_cols[1]
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        print(f"Created Interaction Feature: {col1}_x_{col2}")

    print("Feature Engineering Completed.")
    print("New Shape:", df.shape)

    return df


# --------------------------------------------------
# Save Engineered Data
# --------------------------------------------------
def save_engineered_data(df):

    ensure_folder("data/engineered")

    path = "data/online_retail_customer_churn.csv"
    df.to_csv(path, index=False)

    print("Engineered data saved at:", path)


# --------------------------------------------------
# Run Independently
# --------------------------------------------------
if __name__ == "__main__":

    print("----- STARTING FEATURE ENGINEERING -----")

    df = load_processed_data()

    df_engineered = engineer_features(df)

    save_engineered_data(df_engineered)

    print("----- FEATURE ENGINEERING COMPLETED SUCCESSFULLY -----")
