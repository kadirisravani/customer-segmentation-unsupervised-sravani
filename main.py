# CUSTOMER SEGMENTATION - COMPLETE WORKING FILE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def main():

    print("Loading Dataset...")

    # Change path if needed
    df = pd.read_csv("data/online_retail_customer_churn.csv")

    print("Dataset Shape:", df.shape)

    # Preprocessing


    df = df.dropna()

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    # Keep numeric data
    df_numeric = df.select_dtypes(include=[np.number])

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # KMEANS CLUSTERING

    print("\nRunning KMeans...")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print("\n===== Evaluation =====")
    print("Clusters:", len(set(labels)))
    print("Silhouette Score:", round(sil, 4))
    print("Davies-Bouldin Index:", round(db, 4))

    # PCA Visualization

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("PCA Cluster Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


    # Cluster Summary
  
  
    df['Cluster'] = labels

    print("\nCluster Summary (Mean Values):")
    print(df.groupby('Cluster').mean(numeric_only=True))


    # Feature Importance

    print("\nCalculating Feature Importance...")

    X = df.drop('Cluster', axis=1)
    y = df['Cluster']

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 5 Important Features:")
    print(importance.head())


    # CLV Calculation (Advanced)
   
   
    print("\nCalculating Customer Lifetime Value...")

    spend_col = None
    freq_col = None

    for col in df.columns:
        if 'spend' in col.lower() or 'amount' in col.lower():
            spend_col = col
        if 'frequency' in col.lower() or 'transaction' in col.lower():
            freq_col = col

    if spend_col and freq_col:
        df['CLV'] = df[spend_col] * df[freq_col]
    else:
        df['CLV'] = df_numeric.mean(axis=1)

    df['CLV_Group'] = pd.qcut(df['CLV'], 3,
                              labels=['Low', 'Medium', 'High'])

    print("\nCLV Group Distribution:")
    print(df['CLV_Group'].value_counts())

    print("\nProject Completed Successfully!")


if __name__ == "__main__":
    main()
