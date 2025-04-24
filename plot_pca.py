import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from Bio import SeqIO

# Define phylum colors using ColorBrewer
PHYLUM_COLORS = {
    'Ascomycota': '#377eb8',    # Blue
    'Basidiomycota': '#e41a1c',       # Red
    'Mucoromycota': '#4daf4a',     # Green
    'Zoopagomycota': '#984ea3',    # Purple
    'Chytridiomycota': '#ff7f00',  # Orange
    'Blastocladiomycota': '#ffff33',# Yellow
    'Cryptomycota': '#a65628'      # Brown
}

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    embeddings = np.load(embedding_file)
    print(f"Loaded embedding array shape: {embeddings.shape}")
    
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
        acc = record.id.split()[0]
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

def create_pca_plot(embeddings, protein_ids, output_dir, gene_name, phyla=None):
    """Create and save PCA plot of embeddings."""
    print(f"Input embedding matrix shape: {embeddings.shape}")
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if phyla is not None:
        for phylum in PHYLUM_COLORS.keys():
            if phylum in set(phyla):
                mask = [p == phylum for p in phyla]
                plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                           alpha=0.6, label=phylum, 
                           color=PHYLUM_COLORS[phylum],
                           s=7)
        plt.legend(bbox_to_anchor=(1.05, 1), 
                  loc='upper left',
                  fontsize=12,
                  markerscale=2,
                  frameon=True)
    else:
        plt.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.6, s=10)
    
    plt.title('PCA of Protein Embeddings')
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}% variance explained)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}% variance explained)')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'embeddings_pca_{gene_name.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Paths
    gene_name = 'LYS1'
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta'
    output_dir = '/home/s233201/esm_runs/plots'
    
    # Load data
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    phyla = [taxa_dict[acc] for acc in fasta_accessions]
    
    print("Loading embeddings...")
    embeddings, protein_ids = load_embeddings(embedding_file)
    print(f"Loaded {len(protein_ids)} sequences with embedding shape: {embeddings.shape}")
    
    print("Creating PCA plot...")
    create_pca_plot(embeddings, protein_ids, output_dir, gene_name, phyla=phyla)
    print(f"Plot saved to {output_dir}/embeddings_pca_{gene_name.lower()}.png")

if __name__ == "__main__":
    main()
