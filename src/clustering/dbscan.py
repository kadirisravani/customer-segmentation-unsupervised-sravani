import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Safe Folder Creation

def ensure_folder(path):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)

# Load Engineered Data

def load_data():
    path = "data/online_retail_customer_churn.csv"

    if not os.path.exists(path):
        raise FileNotFoundError("Engineered data not found. Run feature_engineering.py first.")

    df = pd.read_csv(path)
    print("Engineered data loaded successfully.")
    print("Shape:", df.shape)

    return df

# Apply DBSCAN

def apply_dbscan(X, eps=0.5, min_samples=5):

    print("\nApplying DBSCAN...")

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    unique_clusters = set(labels)
    print("DBSCAN completed.")
    print("Number of clusters found:", len(unique_clusters - {-1}))
    print("Number of noise points:", list(labels).count(-1))

    return labels

# PCA Visualization

def visualize_clusters(X, labels):

    ensure_folder("results")
    ensure_folder("results/dbscan")

    print("Applying PCA for visualization...")

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("DBSCAN Clustering (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("results/dbscan/dbscan_clusters.png")
    plt.close()

    print("Cluster visualization saved at results/dbscan/dbscan_clusters.png")

# Save Results
def save_results(df, labels):

    ensure_folder("results")
    ensure_folder("results/dbscan")

    df["DBSCAN_Cluster"] = labels
    df.to_csv("results/dbscan/dbscan_results.csv", index=False)

    print("DBSCAN results saved at results/dbscan/dbscan_results.csv")


# Run Independently

if __name__ == "__main__":

    print("----- STARTING DBSCAN CLUSTERING -----")

    df = load_data()

    # Keep only numeric data
    X = df.select_dtypes(include=np.number)

    # Scale data (IMPORTANT for DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = apply_dbscan(X_scaled)

    visualize_clusters(X_scaled, labels)

    save_results(df, labels)

    print("----- DBSCAN COMPLETED SUCCESSFULLY -----")
