import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pandas as pd

# Load embeddings
gene_name = 'lys4'
embeddings = np.load(f'/home/s233201/esm_runs/embeddings/{gene_name}.npy')

# Check if embeddings were loaded and reshape if needed
if embeddings.size == 0:
    raise ValueError("Empty embeddings array loaded")
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(-1, 1)

clusterer = DBSCAN(eps=0.002, min_samples=10, metric='cosine')
cluster_labels = clusterer.fit_predict(embeddings)

# Print clustering statistics
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Total number of points: {len(cluster_labels)}")
print(f"Percentage of points clustered: {(1 - n_noise/len(cluster_labels))*100:.2f}%")

# Save cluster assignments to CSV
cluster_df = pd.DataFrame({
    'sequence_idx': range(len(cluster_labels)),
    'cluster': cluster_labels
})
cluster_df.to_csv(f'/home/s233201/esm_runs/clusters/{gene_name}_clusters.csv', index=False)

# Plot clusters
# Reduce dimensionality for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Create scatter plot of clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=cluster_labels, cmap='Spectral',
                     alpha=0.6, s=50)
plt.colorbar(scatter)
plt.title(f'HDBSCAN Clustering of {gene_name} (PCA projection)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(f'/home/s233201/esm_runs/plots/dbscan_clusters_{gene_name}.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# # Plot cluster probabilities
# plt.hist(clusterer.probabilities_, bins=50)
# plt.xlabel('Cluster membership probability')
# plt.ylabel('Number of points')
# plt.title('HDBSCAN Cluster Membership Probabilities')
# plt.savefig(f'/home/s233201/esm_runs/plots/hdbscan_probabilities_{gene_name}.png', dpi=300, bbox_inches='tight')
# plt.close()
