import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

# Elbow Method

def elbow_method(X):

    ensure_folder("results")
    ensure_folder("results/kmeans")

    print("\nRunning Elbow Method...")

    inertia_values = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertia_values.append(model.inertia_)

    plt.figure()
    plt.plot(range(1, 11), inertia_values, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.savefig("results/kmeans/elbow_plot.png")
    plt.close()

    print("Elbow plot saved at results/kmeans/elbow_plot.png")

# Apply KMeans

def apply_kmeans(X, n_clusters=3):

    print("\nApplying KMeans...")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    print("KMeans clustering completed.")
    print("Number of clusters:", n_clusters)
    print("Cluster distribution:\n", pd.Series(labels).value_counts())

    silhouette = silhouette_score(X, labels)
    print("Silhouette Score:", round(silhouette, 4))

    return labels


# PCA Visualization
def visualize_clusters(X, labels):

    ensure_folder("results")
    ensure_folder("results/kmeans")

    print("\nApplying PCA for visualization...")

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("KMeans Clustering (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("results/kmeans/kmeans_clusters.png")
    plt.close()

    print("Cluster plot saved at results/kmeans/kmeans_clusters.png")

# Save Results
def save_results(df, labels):

    ensure_folder("results")
    ensure_folder("results/kmeans")

    df["KMeans_Cluster"] = labels
    df.to_csv("results/kmeans/kmeans_results.csv", index=False)

    print("KMeans results saved at results/kmeans/kmeans_results.csv")
# Run Independently
if __name__ == "__main__":

    print("----- STARTING KMEANS CLUSTERING -----")

    df = load_data()

    # Select only numeric columns
    X = df.select_dtypes(include=np.number)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    elbow_method(X_scaled)

    # Apply KMeans (choose 3 clusters based on elbow)
    labels = apply_kmeans(X_scaled, n_clusters=3)

    # Visualize clusters
    visualize_clusters(X_scaled, labels)

    # Save results
    save_results(df, labels)

    print("----- KMEANS CLUSTERING COMPLETED SUCCESSFULLY -----")

