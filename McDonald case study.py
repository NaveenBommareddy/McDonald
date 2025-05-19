# McDonald's Market Segmentation Analysis in Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Step 1: Load Dataset (replace with real path)
df = pd.read_csv("mcdonalds.csv")  # Assumes CSV version of R 'mcdonalds' data

# Step 2: Convert Yes/No to 1/0 for binary image attributes
binary_df = df.iloc[:, :11].applymap(lambda x: 1 if x == "Yes" else 0)

# Step 3: PCA for Visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(binary_df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(pca_result[:,0], pca_result[:,1], alpha=0.5)
plt.title("PCA of Brand Image Attributes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Step 4: K-means Clustering
kmeans = KMeans(n_clusters=4, random_state=1234)
df["Segment_KMeans"] = kmeans.fit_predict(X_scaled)

# Step 5: Hierarchical Clustering
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# Step 6: Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=123)
df["Segment_GMM"] = gmm.fit_predict(X_scaled)

# Step 7: Logistic Regression to Predict Segment
X = binary_df
y = df["Segment_KMeans"]
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, y)

# Step 8: Decision Tree for Interpretation
tree = DecisionTreeClassifier(max_depth=4, random_state=123)
tree.fit(X, y)

plt.figure(figsize=(14,6))
plot_tree(tree, feature_names=binary_df.columns, class_names=True, filled=True)
plt.title("Decision Tree to Predict Segments")
plt.show()

# Step 9: Visualize Segment Profiles
sns.boxplot(x="Segment_KMeans", y="Like", data=df)
plt.title("Like Scores by Segment")
plt.show()

# Optional: Silhouette Score for Clustering Evaluation
score = silhouette_score(X_scaled, df["Segment_KMeans"])
print(f"Silhouette Score: {score:.2f}")