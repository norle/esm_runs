import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, DBSCAN
import pandas as pd
import umap
from multiprocessing import Pool, cpu_count
from Bio import SeqIO

def get_accession_numbers(gene_name):
    """Extract accession numbers from FASTA file."""
    accessions = []
    with open(f'/home/s233201/esm_runs/inputs/{gene_name.upper()}.fasta', 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            accessions.append(record.id)
    return accessions

def run_hdbscan_analysis(gene_name):
    embeddings = np.load(f'/home/s233201/esm_runs/embeddings/{gene_name}.npy')

    # Get accession numbers from FASTA file
    accession_numbers = get_accession_numbers(gene_name)
    
    # Check if embeddings were loaded and reshape if needed
    if embeddings.size == 0:
        raise ValueError("Empty embeddings array loaded")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    # Identify unique embeddings and keep track of indices
    unique_embeddings, unique_indices, inverse_indices = np.unique(embeddings, axis=0, return_index=True, return_inverse=True)

    print('Running UMAP')
    # Perform UMAP dimensionality reduction on unique embeddings
    reducer = umap.UMAP(n_neighbors=100, min_dist=0, n_components=10, metric='cosine')
    embeddings_reduced = reducer.fit_transform(unique_embeddings)

    print('Running HDBSCAN')
    # Initialize and fit HDBSCAN on unique embeddings
    clusterer = HDBSCAN(min_cluster_size=5, min_samples=20, metric='l2', cluster_selection_epsilon=0.5, cluster_selection_method='leaf')
    unique_cluster_labels = clusterer.fit_predict(embeddings_reduced)

    # Map cluster labels back to all embeddings using inverse indices
    cluster_labels = unique_cluster_labels[inverse_indices]

    # Print clustering statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"Number of unique sequences: {len(unique_embeddings)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Total number of points: {len(cluster_labels)}")
    print(f"Percentage of points clustered: {(1 - n_noise/len(cluster_labels))*100:.2f}%")

    # Save cluster assignments to CSV with accession numbers
    cluster_df = pd.DataFrame({
        'accession': accession_numbers,
        'cluster': cluster_labels
    })
    cluster_df.to_csv(f'/home/s233201/esm_runs/clusters/5_20_05_leaf/{gene_name}_clusters.csv', index=False)

if __name__ == '__main__':
    #gene_names = ['aco2']
    gene_names = ['lys1','lys2','lys4','lys9','lys12','lys20','aro8','aco2']

    if len(gene_names) > 1:
        # Use maximum available cores or number of genes, whichever is smaller
        n_processes = min(len(gene_names), cpu_count())
        print(f"Running analysis in parallel with {n_processes} processes")
        
        with Pool(processes=n_processes) as pool:
            pool.map(run_hdbscan_analysis, gene_names)
    else:
        # Run sequentially if only one gene
        run_hdbscan_analysis(gene_names[0])
    
    print("Finished clustering for all genes")