import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
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

# Apply GMM

def apply_gmm(X, n_clusters=3):

    print("\nApplying Gaussian Mixture Model...")

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X)

    labels = gmm.predict(X)

    print("GMM clustering completed.")
    print("Number of clusters:", n_clusters)
    print("Cluster distribution:\n", pd.Series(labels).value_counts())

    return labels

# PCA Visualization
def visualize_clusters(X, labels):

    ensure_folder("results")
    ensure_folder("results/gmm")

    print("\nApplying PCA for visualization...")

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("GMM Clustering (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("results/gmm/gmm_clusters.png")
    plt.close()

    print("Cluster visualization saved at results/gmm/gmm_clusters.png")


# Save Results

def save_results(df, labels):

    ensure_folder("results")
    ensure_folder("results/gmm")

    df["GMM_Cluster"] = labels
    df.to_csv("results/gmm/gmm_results.csv", index=False)

    print("GMM results saved at results/gmm/gmm_results.csv")

# Run Independently

if __name__ == "__main__":

    print("----- STARTING GMM CLUSTERING -----")

    df = load_data()

    # Select only numeric columns
    X = df.select_dtypes(include=np.number)

    # Scale features (IMPORTANT for GMM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = apply_gmm(X_scaled, n_clusters=3)

    visualize_clusters(X_scaled, labels)

    save_results(df, labels)

    print("----- GMM COMPLETED SUCCESSFULLY -----")
