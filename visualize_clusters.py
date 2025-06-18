import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from Bio import SeqIO
from multiprocessing import Pool, cpu_count

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

def compute_umap_for_gene(gene_name):
    """Compute UMAP coordinates for a single gene (for parallel processing)."""
    print(f"Processing {gene_name}...")
    
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    cluster_file = f'/home/s233201/esm_runs/clusters/5_20_05_leaf/{gene_name.lower()}_clusters.csv'
    
    # Load embeddings and clusters
    embeddings, protein_ids = load_embeddings(embedding_file)
    clusters = load_cluster_info(cluster_file)
    
    print(f"Computing UMAP for {gene_name} - Input embedding matrix shape: {embeddings.shape}")
    
    # Ensure embeddings are 2D
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
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
    
    return {
        'gene_name': gene_name,
        'umap_coords': umap_coords,
        'clusters': clusters
    }

def plot_umap_on_axis(result, ax):
    """Plot UMAP results on given axis."""
    gene_name = result['gene_name']
    umap_coords = result['umap_coords']
    clusters = result['clusters']
    
    if clusters is not None:
        # Get unique clusters excluding -1 (noise)
        unique_clusters = sorted(set(clusters[clusters != -1]))
        # Create colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = {c: colors[i] for i, c in enumerate(unique_clusters)}
        
        # Plot noise points first (grey)
        noise_mask = clusters == -1
        ax.scatter(umap_coords[noise_mask, 0], umap_coords[noise_mask, 1],
                   alpha=0.6, color='grey', s=7)
        
        # Plot clustered points
        for cluster in unique_clusters:
            mask = clusters == cluster
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       alpha=0.6, color=cluster_colors[cluster], s=7)
    else:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=10)
    
    ax.set_title(f'{gene_name}', fontsize=12)
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)

def create_umap_plot(embeddings, protein_ids, gene_name, ax, phyla=None, clusters=None):
    """Create UMAP plot of embeddings on given axes."""
    print(f"Processing {gene_name} - Input embedding matrix shape: {embeddings.shape}")
    
    # Ensure embeddings are 2D
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
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
    
    if clusters is not None:
        # Get unique clusters excluding -1 (noise)
        unique_clusters = sorted(set(clusters[clusters != -1]))
        # Create colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = {c: colors[i] for i, c in enumerate(unique_clusters)}
        
        # Plot noise points first (grey)
        noise_mask = clusters == -1
        ax.scatter(umap_coords[noise_mask, 0], umap_coords[noise_mask, 1],
                   alpha=0.6, color='grey', s=7)
        
        # Plot clustered points
        for cluster in unique_clusters:
            mask = clusters == cluster
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       alpha=0.6, color=cluster_colors[cluster], s=7)
    else:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=10)
    
    ax.set_title(f'{gene_name}', fontsize=12)
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)

def run_for_gene(gene_name):
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    cluster_file = f'/home/s233201/esm_runs/clusters/5_20_05_leaf/{gene_name.lower()}_clusters.csv'
    output_dir = '/home/s233201/esm_runs/plots/clusters/5_20_05_leaf'
    
    # Load embeddings and clusters
    print("Loading embeddings and clusters...")
    embeddings, protein_ids = load_embeddings(embedding_file)
    clusters = load_cluster_info(cluster_file)
    
    print("Creating UMAP plot...")
    create_umap_plot(embeddings, protein_ids, gene_name, clusters=clusters)
    print(f"Plot saved to {output_dir}/embeddings_umap_{gene_name.lower()}_clusters.png")

def main():
    # Paths
    gene_names = ['LYS1', 'LYS2', 'LYS4', 'LYS9', 'LYS12', 'LYS20', 'ARO8', 'ACO2']
    output_dir = '/home/s233201/esm_runs/plots/clusters/5_20_05_leaf'
    
    # Use parallel processing for UMAP computations
    num_cores = min(cpu_count(), len(gene_names))
    print(f"Computing UMAP for {len(gene_names)} genes using {num_cores} cores...")
    
    with Pool(num_cores) as pool:
        results = pool.map(compute_umap_for_gene, gene_names)
    
    # Create 3x3 subplot figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('UMAP Projections of Protein Embeddings\n(based on cosine similarity)', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot results on subplots
    for i, result in enumerate(results):
        print(f"Plotting {result['gene_name']}...")
        plot_umap_on_axis(result, axes_flat[i])
    
    # Hide the last subplot if there are fewer than 9 genes
    if len(gene_names) < 9:
        axes_flat[len(gene_names)].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'all_genes_umap_clusters_3x3.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Combined plot saved to {output_dir}/all_genes_umap_clusters_3x3.png")

if __name__ == "__main__":
    main()
