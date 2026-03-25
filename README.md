1️⃣ Project Title

Customer Segmentation Using Unsupervised Machine Learning

2️⃣ Problem Statement

Businesses need to understand customer behavior to improve marketing strategies, reduce churn, and increase revenue.
This project applies unsupervised machine learning algorithms to segment customers based on purchasing behavior, spending patterns, and engagement features.

The objective is to:
Identify distinct customer groups
Compare clustering algorithms
Extract meaningful business insights
Recommend actionable strategies for each segment

3️⃣ Dataset Description

Dataset Name: online_retail_customer_churn.csv
The dataset contains customer transaction and behavioral data.

Key Features Used:
CustomerID – Unique customer identifier
Recency – Days since last purchase
Frequency – Number of transactions
Monetary – Total spending
Tenure – Customer duration
AvgOrderValue – Average purchase value
Churn (if present) – Customer churn status
Feature Engineering Performed:
RFM Features (Recency, Frequency, Monetary)
Average Order Value
Behavioral ratios
Feature scaling using StandardScaler

4️⃣ Algorithms Used
We implemented and compared the following clustering algorithms:

🔹 KMeans Clustering
Partitional clustering
Uses centroid-based grouping
Evaluated using Silhouette Score & Elbow Method

🔹 DBSCAN
Density-based clustering
Detects noise and outliers
Does not require predefined cluster count

🔹 Hierarchical Clustering
Agglomerative clustering
Dendrogram visualization used
Good for hierarchical customer grouping

5️⃣ How to Run the Project
Step 1: Clone the repository
Step 2: Install dependencies
Step 3: Run the project

6️⃣ Key Results
 Number of Clusters Found
Algorithm	Number of Clusters
KMeans	4
DBSCAN	3 (+ noise)
Hierarchical	4


Best Algorithm :

KMeans performed best based on:
Higher Silhouette Score
Clear cluster separation
Better business interpretability

 Business Insights :
 Cluster 1 – High-Value Premium Customers
High frequency
High spending
Low recency
Strategy: Loyalty rewards, premium offers

Cluster 2 – Regular Customers
Moderate spending
Moderate frequency
Strategy: Upselling and cross-selling

 Cluster 3 – Low-Value Customers
Low spending
Low frequency
Strategy: Discount campaigns

 Cluster 4 – At-Risk Customers
High recency (inactive)
Previously moderate spenders
Strategy: Retargeting & win-back campaigns

7️⃣ Sample Visualizations
📊 Elbow Method (KMeans)
 screenshot


📊 Cluster Visualization (PCA 2D Projection)
Add screenshot here

📊 Hierarchical Dendrogram
Add screenshot here

📊 DBSCAN Cluster Plot
Add screenshot here

📂 Project Structure
├── data/
│   └── online_retail_customer_churn.csv
├── src/
│   ├── preprocessing.py
│   ├── kmeans.py
│   ├── dbscan.py
│   ├── hierarchical.py
│   └── visualization.py
├── screenshots/
├── main.py
├── requirements.txt
└── README.md
🔮 Future Improvements

Add Gaussian Mixture Model (GMM)

Deploy as a Streamlit dashboard

Automate optimal cluster selection

Integrate with marketing automation tools

✅ Conclusion

This project successfully demonstrates how unsupervised learning can uncover hidden customer segments and generate actionable business insights.

By comparing multiple clustering algorithms, we identified KMeans as the most interpretable and business-friendly approach for customer segmentation.