import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Dataset

def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv("data/online_retail_customer_churn.csv")
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    return df

# Handle Missing Values

def handle_missing_values(df):

    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]

    print(f"Removed {before - after} rows with missing values.")
    return df


# Encode Categorical Columns

def encode_categorical(df):

    df_encoded = df.copy()

    categorical_cols = df_encoded.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    print(f"Encoded {len(categorical_cols)} categorical columns.")
    return df_encoded


# Scale Features
def scale_features(df):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    print("Feature scaling completed.")
    return df_scaled


# Safe Folder Creation
def ensure_processed_folder():

    data_folder = "data"
    processed_folder = os.path.join(data_folder, "processed")

    # If 'processed' exists but is a file → remove it
    if os.path.exists(processed_folder) and not os.path.isdir(processed_folder):
        os.remove(processed_folder)

    os.makedirs(processed_folder, exist_ok=True)

    return processed_folder

# Full Pipeline
def preprocess_pipeline(input_path):

    print("----- STARTING DATA PREPROCESSING -----")

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df_scaled = scale_features(df)

    processed_folder = ensure_processed_folder()
    save_path = os.path.join(processed_folder, "processed_data.csv")

    df_scaled.to_csv(save_path, index=False)

    print(f"Processed data saved at: {save_path}")
    print("----- PREPROCESSING COMPLETED SUCCESSFULLY -----")

    return df_scaled

# Run Independently (For VS Code Testing)
if __name__ == "__main__":

    dataset_path = "data/online_retail_customer_churn.csv"

    preprocess_pipeline(dataset_path)
