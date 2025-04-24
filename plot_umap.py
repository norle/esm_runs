import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from Bio import SeqIO
import multiprocessing
import pickle  # Add at top with other imports

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

def create_umap_plot(embeddings, protein_ids, output_dir, gene_name, phyla=None, ax=None, save_format='png'):
    """Create and save UMAP plot of embeddings."""
    print(f"Input embedding matrix shape: {embeddings.shape}")
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    umap = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=100
    )
    
    distance_matrix = 1 - similarity_matrix
    umap_coords = umap.fit_transform(distance_matrix)
    
    # If no axis provided, create new figure
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    if phyla is not None:
        for phylum in PHYLUM_COLORS.keys():
            if phylum in set(phyla):
                mask = [p == phylum for p in phyla]
                ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                        alpha=0.6, label=phylum, 
                        color=PHYLUM_COLORS[phylum],
                        s=7)
        # Only add legend for the last subplot when plotting separately
        if ax.is_last_row():
            ax.legend(bbox_to_anchor=(1.05, 1), 
                     loc='upper left',
                     fontsize=12,
                     markerscale=2,
                     frameon=True)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=10)
    
    ax.set_title(f'{gene_name}')
    
    # Only save if we're not using subplots
    if ax is None:
        os.makedirs(output_dir, exist_ok=True)
        if save_format == 'png':
            plt.savefig(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
        elif save_format == 'fig':
            with open(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}.fig.pickle'), 'wb') as f:
                pickle.dump(plt.gcf(), f)
        plt.close()

def main(gene_name, use_subplots=False, fig=None, ax=None, save_format='png'):
    # Paths
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta'
    output_dir = '/home/s233201/esm_runs/plots'
    
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    phyla = [taxa_dict[acc] for acc in fasta_accessions]
    
    print("Loading embeddings...")
    embeddings, protein_ids = load_embeddings(embedding_file)
    print(f"Loaded {len(protein_ids)} sequences with embedding shape: {embeddings.shape}")
    
    print("Creating UMAP plot...")
    create_umap_plot(embeddings, protein_ids, output_dir, gene_name, 
                    phyla=phyla, ax=ax, save_format=save_format)
    
    if not use_subplots:
        ext = 'png' if save_format == 'png' else 'fig.pickle'
        print(f"Plot saved to {output_dir}/embeddings_umap_{gene_name.lower()}.{ext}")

def process_gene_data(gene_name):
    """Process single gene and return UMAP coordinates and phyla"""
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta'
    
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    phyla = [taxa_dict[acc] for acc in fasta_accessions]
    
    print(f"Processing {gene_name}...")
    embeddings, _ = load_embeddings(embedding_file)
    
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    umap = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=100
    )
    
    umap_coords = umap.fit_transform(distance_matrix)
    return gene_name, umap_coords, phyla

if __name__ == "__main__":
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    use_subplots = True  # Toggle for subplot vs separate plots
    save_format = 'png'  # Toggle between 'png' or 'fig' for saving format
    
    if use_subplots:
        # Process all genes in parallel
        num_cores = multiprocessing.cpu_count()
        print(f"Running on {num_cores} cores")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(process_gene_data, gene_names)
        
        # Calculate the figure size to ensure square subplots
        n_rows = 2
        n_cols = 4
        subplot_size = 4  # Size of each subplot in inches
        fig_width = subplot_size * n_cols
        fig_height = subplot_size * n_rows
        
        # Create figure with square subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = axes.ravel()
        
        # Find global limits for consistent scaling
        all_coords = np.vstack([coords for _, coords, _ in results])
        min_val = all_coords.min()
        max_val = all_coords.max()
        plot_range = max_val - min_val
        center = (max_val + min_val) / 2
        
        # Plot results with consistent square dimensions
        for idx, (gene_name, umap_coords, phyla) in enumerate(results):
            ax = axes[idx]
            for phylum in PHYLUM_COLORS.keys():
                if phylum in set(phyla):
                    mask = [p == phylum for p in phyla]
                    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                             alpha=0.8,
                             label=phylum,
                             color=PHYLUM_COLORS[phylum],
                             s=12,
                             zorder=2)
            
            # Set consistent square limits
            ax.set_xlim(center - plot_range/2, center + plot_range/2)
            ax.set_ylim(center - plot_range/2, center + plot_range/2)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Set title
            ax.set_title(f'{gene_name}', fontsize=16, pad=5)
        
        # Adjust layout with enough spacing
        plt.tight_layout()
        
        if save_format == 'png':
            plt.savefig('/home/s233201/esm_runs/plots/embeddings_umap_all.png',
                       dpi=300, bbox_inches='tight')
        elif save_format == 'fig':
            with open('/home/s233201/esm_runs/plots/embeddings_umap_all.fig.pickle', 'wb') as f:
                pickle.dump(fig, f)
        plt.close()
    else:
        # Process genes in parallel for separate plots
        num_cores = multiprocessing.cpu_count()
        print(f"Running on {num_cores} cores")
        with multiprocessing.Pool(processes=num_cores) as pool:
            from functools import partial
            main_with_format = partial(main, save_format=save_format)
            pool.map(main_with_format, gene_names)
