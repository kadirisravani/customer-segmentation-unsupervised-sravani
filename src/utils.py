import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# --------------------------------------------------
# Safe Folder Creation (Windows Safe)
# --------------------------------------------------
def ensure_folder(path):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)


def create_results_folders():
    ensure_folder("results")
    ensure_folder("results/pca_outputs")
    ensure_folder("results/cluster_plots")


# --------------------------------------------------
# Load Engineered Data
# --------------------------------------------------
def load_engineered_data():

    path = "data/online_retail_customer_churn.csv"

    if not os.path.exists(path):
        raise FileNotFoundError("Engineered data not found. Run feature_engineering.py first.")

    df = pd.read_csv(path)
    print("Engineered dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


# --------------------------------------------------
# Apply PCA
# --------------------------------------------------
def apply_pca(df, n_components=2):

    print("\nApplying PCA...")

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)

    print("PCA completed.")
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    return components


# --------------------------------------------------
# Save PCA Plot
# --------------------------------------------------
def save_pca_plot(components, labels=None):

    create_results_folders()

    plt.figure()

    if labels is not None:
        plt.scatter(components[:, 0], components[:, 1], c=labels)
    else:
        plt.scatter(components[:, 0], components[:, 1])

    plt.title("PCA Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("results/pca_outputs/pca_plot.png")
    plt.close()

    print("PCA plot saved in results/pca_outputs/")


# --------------------------------------------------
# Save Generic Cluster Plot
# --------------------------------------------------
def save_cluster_plot(components, labels):

    create_results_folders()

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)

    plt.title("Cluster Visualization (PCA Reduced)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig("results/cluster_plots/cluster_plot.png")
    plt.close()

    print("Cluster plot saved in results/cluster_plots/")


# --------------------------------------------------
# Run Independently (Testing Mode)
# --------------------------------------------------
if __name__ == "__main__":

    print("----- STARTING PCA TEST -----")

    df = load_engineered_data()

    components = apply_pca(df)

    save_pca_plot(components)

    print("----- UTILS TEST COMPLETED SUCCESSFULLY -----")
