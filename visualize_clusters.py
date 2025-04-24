import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from Bio import SeqIO

# Define phylum colors using ColorBrewer
PHYLUM_SHAPES = {
    'Ascomycota': 'o',    # Circle
    'Basidiomycota': 's', # Square
    'Mucoromycota': 'D',  # Diamond
    'Zoopagomycota': '^', # Triangle up
    'Chytridiomycota': 'v',# Triangle down
    'Blastocladiomycota': '<',# Triangle left
    'Cryptomycota': '>'   # Triangle right
}

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    embeddings = np.load(embedding_file)
    print(f"Loaded embedding array shape: {embeddings.shape}")
    
    # Generate sequential IDs since we don't have protein IDs anymore
    protein_ids = [f"protein_{i+1}" for i in range(len(embeddings))]
    
    return embeddings, protein_ids

def load_taxa_info(taxa_file):
    """Load taxa information from CSV file."""
    df = pd.read_csv(taxa_file)
    return dict(zip(df.Accession, df.Phylum))

def get_fasta_accessions(fasta_file):
    """Get accessions in order from FASTA file."""
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Assuming the accession is the first part of the header before any spaces
        acc = record.id.split()[0]
        # Remove version number if present
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

def load_cluster_info(cluster_file):
    """Load cluster assignments from CSV file."""
    df = pd.read_csv(cluster_file)
    return df['cluster'].values

def create_umap_plot(embeddings, protein_ids, output_dir, gene_name, phyla=None, clusters=None):
    """Create and save UMAP plot of embeddings."""
    print(f"Input embedding matrix shape: {embeddings.shape}")
    
    # Ensure embeddings are 2D
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Create UMAP projection
    umap = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=100,
    )
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    umap_coords = umap.fit_transform(distance_matrix)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if clusters is not None:
        # Get unique clusters excluding -1 (noise)
        unique_clusters = sorted(set(clusters[clusters != -1]))
        # Create colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = {c: colors[i] for i, c in enumerate(unique_clusters)}
        
        # Plot noise points first (grey)
        noise_mask = clusters == -1
        plt.scatter(umap_coords[noise_mask, 0], umap_coords[noise_mask, 1],
                   alpha=0.6, color='grey', label='Noise', s=7)
        
        # Plot clustered points
        for cluster in unique_clusters:
            mask = clusters == cluster
            plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       alpha=0.6, label=f'Cluster {cluster}',
                       color=cluster_colors[cluster], s=7)
        
        plt.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  fontsize=12,
                  markerscale=2,
                  frameon=True)
    else:
        plt.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=10)
    
    plt.title('UMAP Projection of Protein Embeddings\n(based on cosine similarity)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}_clusters.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def run_for_gene(gene_name):

    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    cluster_file = f'/home/s233201/esm_runs/clusters/5_20_05_leaf/{gene_name.lower()}_clusters.csv'
    output_dir = '/home/s233201/esm_runs/plots/clusters/5_20_05_leaf'
    
    # Load embeddings and clusters
    print("Loading embeddings and clusters...")
    embeddings, protein_ids = load_embeddings(embedding_file)
    clusters = load_cluster_info(cluster_file)
    
    print("Creating UMAP plot...")
    create_umap_plot(embeddings, protein_ids, output_dir, gene_name, clusters=clusters)
    print(f"Plot saved to {output_dir}/embeddings_umap_{gene_name.lower()}_clusters.png")

def main():
    # Paths
    gene_names = ['LYS1', 'LYS2', 'LYS4', 'LYS9', 'LYS12', 'LYS20', 'ARO8', 'ACO2']
    
    if len(gene_names) > 1:
        # Use parallel processing for multiple genes
        from multiprocessing import Pool, cpu_count
        num_cores = min(cpu_count(), len(gene_names))
        print(f"Processing {len(gene_names)} genes using {num_cores} cores...")
        
        with Pool(num_cores) as pool:
            pool.map(run_for_gene, gene_names)
    else:
        # Single gene, run sequentially
        run_for_gene(gene_names[0])

if __name__ == "__main__":
    main()
