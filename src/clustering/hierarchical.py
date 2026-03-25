import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Apply Hierarchical Clustering

def apply_hierarchical(X, n_clusters=3):

    print("\nApplying Hierarchical Clustering...")

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    print("Hierarchical clustering completed.")
    print("Number of clusters:", n_clusters)
    print("Cluster distribution:\n", pd.Series(labels).value_counts())

    return labels

# Save Dendrogram

def save_dendrogram(X):

    ensure_folder("results")
    ensure_folder("results/hierarchical")

    print("\nGenerating dendrogram...")

    linked = linkage(X, method='ward')

    plt.figure(figsize=(8, 5))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    plt.savefig("results/hierarchical/dendrogram.png")
    plt.close()

    print("Dendrogram saved at results/hierarchical/dendrogram.png")


# PCA Visualization

def visualize_clusters(X, labels):

    ensure_folder("results")
    ensure_folder("results/hierarchical")

    print("\nApplying PCA for visualization...")

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("Hierarchical Clustering (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("results/hierarchical/hierarchical_clusters.png")
    plt.close()

    print("Cluster plot saved at results/hierarchical/hierarchical_clusters.png")

# Save Results

def save_results(df, labels):

    ensure_folder("results")
    ensure_folder("results/hierarchical")

    df["Hierarchical_Cluster"] = labels
    df.to_csv("results/hierarchical/hierarchical_results.csv", index=False)

    print("Hierarchical results saved at results/hierarchical/hierarchical_results.csv")

# Run Independently

if __name__ == "__main__":

    print("----- STARTING HIERARCHICAL CLUSTERING -----")

    df = load_data()

    # Select only numeric columns
    X = df.select_dtypes(include=np.number)

    # Scale features (IMPORTANT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply clustering
    labels = apply_hierarchical(X_scaled, n_clusters=3)

    # Save dendrogram
    save_dendrogram(X_scaled)

    # Save cluster visualization
    visualize_clusters(X_scaled, labels)

    # Save results
    save_results(df, labels)

    print("----- HIERARCHICAL CLUSTERING COMPLETED SUCCESSFULLY -----")
