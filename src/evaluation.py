import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# --------------------------------------------------
# SAFE FOLDER CREATION (Windows Safe)
# --------------------------------------------------
def ensure_folder(path):

    # If path exists and is a FILE → remove it
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)

    os.makedirs(path, exist_ok=True)


def create_results_folders():
    ensure_folder("results")
    ensure_folder("results/metrics")
    ensure_folder("results/cluster_plots")


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
# Elbow Method
# --------------------------------------------------
def elbow_method(X, max_k=10):

    print("\nRunning Elbow Method...")

    wcss = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        wcss.append(model.inertia_)

    create_results_folders()

    plt.figure()
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig("results/cluster_plots/elbow_plot.png")
    plt.close()

    print("Elbow plot saved in results/cluster_plots/")

    return wcss


# --------------------------------------------------
# Evaluate Clustering
# --------------------------------------------------
def evaluate_clustering(X, k=3):

    print("\nRunning KMeans Evaluation...")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print("\n===== CLUSTER EVALUATION RESULTS =====")
    print("Number of Clusters:", k)
    print("Silhouette Score:", round(sil, 4))
    print("Davies-Bouldin Index:", round(db, 4))

    create_results_folders()

    with open("results/metrics/evaluation_metrics.txt", "w") as f:
        f.write("Cluster Evaluation Results\n")
        f.write("--------------------------\n")
        f.write(f"Number of Clusters: {k}\n")
        f.write(f"Silhouette Score: {sil}\n")
        f.write(f"Davies-Bouldin Index: {db}\n")

    print("Metrics saved in results/metrics/")
    print("====================================")

    return sil, db


# --------------------------------------------------
# Run Independently
# --------------------------------------------------
if __name__ == "__main__":

    print("----- STARTING CLUSTER EVALUATION -----")

    df = load_processed_data()
    X = df.values

    elbow_method(X)
    evaluate_clustering(X, k=3)

    print("----- EVALUATION COMPLETED SUCCESSFULLY -----")
