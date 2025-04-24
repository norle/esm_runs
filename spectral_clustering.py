import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import pandas as pd
import umap

# Load embeddings
gene_name = 'lys2'
embeddings = np.load(f'/home/s233201/esm_runs/embeddings/{gene_name}.npy')

# Check if embeddings were loaded and reshape if needed
if embeddings.size == 0:
    raise ValueError("Empty embeddings array loaded")
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(-1, 1)

# First UMAP reduction for clustering
print('Running UMAP for clustering')
reducer = umap.UMAP(n_neighbors=100, min_dist=0, n_components=5, metric='cosine', random_state=42)
embeddings_reduced = reducer.fit_transform(embeddings)

# Perform spectral clustering
print('Running Spectral Clustering')
n_clusters = 7  # adjust this based on your needs
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
cluster_labels = spectral.fit_predict(embeddings_reduced)

# Second UMAP reduction for visualization
print('Running UMAP for visualization')
reducer_2d = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
embeddings_2d = reducer_2d.fit_transform(embeddings)

# Create visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=cluster_labels, cmap='viridis', s=5, alpha=0.6)
plt.colorbar(scatter)
plt.title(f'Spectral Clustering of {gene_name} (2D UMAP projection)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

# Save the plot
plt.savefig(f'/home/s233201/esm_runs/plots/{gene_name}_spectral_clusters.png')
plt.close()

# Save cluster assignments to CSV
cluster_df = pd.DataFrame({
    'sequence_idx': range(len(cluster_labels)),
    'cluster': cluster_labels
})
cluster_df.to_csv(f'/home/s233201/esm_runs/clusters/{gene_name}_spectral_clusters.csv', index=False)

# Print clustering statistics
print(f"Number of clusters: {n_clusters}")
print(f"Total number of points: {len(cluster_labels)}")
