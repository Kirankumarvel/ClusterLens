import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import umap.umap_ as UMAP 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

# Generate synthetic data with four clusters in a 3D space
centers = [
    [2, -6, -6],
    [-1, 9, 4],
    [-8, 7, 2],
    [4, 7, 9]
]
cluster_std = [1, 1, 2, 3.5]

# Create blobs and return the data and labels
X, labels_ = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)

# Display the data in an interactive Plotly 3D scatter plot
df = pd.DataFrame(X, columns=['X', 'Y', 'Z'])
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=labels_.astype(str), opacity=0.7, 
                    color_discrete_sequence=px.colors.qualitative.G10, title="3D Scatter Plot of Four Blobs")
fig.update_traces(marker=dict(size=5, line=dict(width=1, color='black')), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800)  # Remove color bar, resize plot
fig.show()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Plot the 2D t-SNE projection result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D t-SNE Projection of 3D Data", fontsize=14)
ax.set_xlabel("t-SNE Component 1", fontsize=12)
ax.set_ylabel("t-SNE Component 2", fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(scatter, ax=ax, label="Cluster Label")
plt.tight_layout()
plt.show()

# Apply UMAP to reduce the dimensionality to 2D
umap_model = UMAP(n_components=2, random_state=42, min_dist=0.5, spread=1, n_jobs=1)
X_umap = umap_model.fit_transform(X_scaled)

# Plot the 2D UMAP projection result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D UMAP Projection of 3D Data", fontsize=14)
ax.set_xlabel("UMAP Component 1", fontsize=12)
ax.set_ylabel("UMAP Component 2", fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(scatter, ax=ax, label="Cluster Label")
plt.tight_layout()
plt.show()

# Apply PCA to reduce the dimensionality to 2D
pca_model = PCA(n_components=2)
X_pca = pca_model.fit_transform(X_scaled)

# Plot the 2D PCA projection result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D PCA Projection of 3D Data", fontsize=14)
ax.set_xlabel("PCA Component 1", fontsize=12)
ax.set_ylabel("PCA Component 2", fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(scatter, ax=ax, label="Cluster Label")
plt.tight_layout()
plt.show()
