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

# Add special organisms for labeling
SPECIAL_ORGANISMS = {
    "GCF_000146045": "Saccharomyces cerevisiae",
    "GCA_000230395": "Aspergillus niger",
    "GCF_000002655": "Aspergillus fumigatus",
    "GCF_000182895": "Coprinopsis cinerea",
    "GCF_000149305": "Rhizopus delemar",
    "GCF_028827035": "Penicillium chrysogenum"
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

def add_organism_labels(ax, umap_coords, accessions, fontsize=6):
    """Add labels with lines pointing to special organisms."""
    # Get axis limits to ensure labels stay within bounds
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = x_max - x_min
    height = y_max - y_min
    
    # Reduced fontsize for labels
    fontsize = fontsize * 0.8
    
    for i, acc in enumerate(accessions):
        # Strip version number if present
        base_acc = acc.split('.')[0]
        if base_acc in SPECIAL_ORGANISMS:
            # Get the coordinates for this organism
            x, y = umap_coords[i, 0], umap_coords[i, 1]
            
            # Determine whether to place label above or below based on position in plot
            # Alternate between top and bottom placement for better distribution
            idx = list(SPECIAL_ORGANISMS.keys()).index(base_acc)
            if idx % 2 == 0:  # Even indices: place label above the point
                offset_dist = height * 0.1  # 10% of plot height
                offset_y = offset_dist
                va_setting = 'bottom'
            else:  # Odd indices: place label below the point
                offset_dist = -height * 0.1
                offset_y = offset_dist
                va_setting = 'top'
            
            # Keep x-coordinate the same for straight vertical lines
            offset_x = 0
            
            # Make sure label stays within axis limits with a small margin
            margin = min(width, height) * 0.05
            label_x = x  # Keep x-coordinate aligned with point
            label_y = min(max(y + offset_y, y_min + margin), y_max - margin)
            
            # Add a straight line pointing to the organism
            ax.annotate(
                SPECIAL_ORGANISMS[base_acc],
                xy=(x, y),  # Position of the organism
                xytext=(label_x, label_y),  # Position of the text label
                fontsize=fontsize,
                color='black',
                ha='center',
                va=va_setting,
                arrowprops=dict(
                    arrowstyle="-",  # Simple line instead of arrow
                    connectionstyle="arc3,rad=0",  # Straight line (rad=0)
                    color='black',
                    lw=0.5,
                    alpha=0.7
                ),
                bbox=None,  # No background box
                zorder=5  # Place above points but below other elements
            )

def create_umap_plot(embeddings, protein_ids, output_dir, gene_name, phyla=None, ax=None, save_format='png', fasta_accessions=None):
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
    
    # Add labels for special organisms if accessions are provided
    if fasta_accessions is not None:
        add_organism_labels(ax, umap_coords, fasta_accessions, fontsize=8 if ax is None else 6)
    
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
    embedding_file = f'/home/s233201/esm_runs/embeddings/filtered_embeddings/{gene_name.lower()}.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/ordered_taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    output_dir = '/home/s233201/esm_runs/plots'
    
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    phyla = [taxa_dict[acc] for acc in fasta_accessions]
    
    print("Loading embeddings...")
    embeddings, protein_ids = load_embeddings(embedding_file)
    print(f"Loaded {len(protein_ids)} sequences with embedding shape: {embeddings.shape}")
    
    print("Creating UMAP plot...")
    create_umap_plot(embeddings, protein_ids, output_dir, gene_name, 
                    phyla=phyla, ax=ax, save_format=save_format,
                    fasta_accessions=fasta_accessions)  # Pass accessions for labeling
    
    if not use_subplots:
        ext = 'png' if save_format == 'png' else 'fig.pickle'
        print(f"Plot saved to {output_dir}/embeddings_umap_{gene_name.lower()}.{ext}")

def process_gene_data(gene_name):
    """Process single gene and return UMAP coordinates and phyla"""
    embedding_file = f'/home/s233201/esm_runs/embeddings_new/{gene_name}_embeddings.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    
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
    return gene_name, umap_coords, phyla, fasta_accessions  # Return accessions too

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
        
        # Calculate the figure size for a 3x3 grid with wider aspect ratio
        n_rows = 3
        n_cols = 3
        subplot_size = 5  # Increased from 4 to 5
        subplot_aspect_ratio = 1.2  # Make plots wider (width = height * 1.2)
        fig_width = subplot_size * n_cols * subplot_aspect_ratio
        fig_height = subplot_size * n_rows
        
        # Create figure with wider subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = axes.ravel()
        
        # Plot results with adjusted dimensions for the first 8 cells
        for idx, (gene_name, umap_coords, phyla, fasta_accessions) in enumerate(results):
            if idx >= 8:  # Only plot in the first 8 cells
                continue
                
            ax = axes[idx]
            for phylum in PHYLUM_COLORS.keys():
                if phylum in set(phyla):
                    mask = [p == phylum for p in phyla]
                    scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                             alpha=0.7,  # Slightly increased alpha
                             label=phylum,
                             color=PHYLUM_COLORS[phylum],
                             s=1.0,  # Doubled point size from 0.5 to 1.0
                             zorder=2)
            
            # Calculate the data limits for this subplot
            x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
            y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
            
            # Calculate ranges and centers
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            
            # Apply the aspect ratio to the plot limits
            max_range = max(x_range, y_range) * 1.1  # 10% padding
            x_half_range = max_range * subplot_aspect_ratio / 2
            y_half_range = max_range / 2
            
            # Set limits with appropriate aspect ratio
            ax.set_xlim(x_center - x_half_range, x_center + x_half_range)
            ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
            
            # Label special organisms with arrows - use larger fontsize
            add_organism_labels(ax, umap_coords, fasta_accessions, fontsize=12)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Add a subtle grid for better readability
            ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
            
            # Set title with larger font
            ax.set_title(f'{gene_name}', fontsize=18, pad=10, fontweight='bold')
        
        # Create an attractive legend in the last (9th) cell
        legend_ax = axes[8]
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        for spine in legend_ax.spines.values():
            spine.set_visible(False)
        
        # Add a title to the legend subplot
        legend_ax.text(0.5, 0.95, "Legend", fontsize=20, fontweight='bold', 
                      ha='center', va='top', transform=legend_ax.transAxes)
        
        # Add phylum legend with larger markers
        phylum_legend = []
        for phylum, color in PHYLUM_COLORS.items():
            phylum_legend.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=14, 
                          label=phylum)
            )
        

        # Add the phylum legend in the upper part
        legend1 = legend_ax.legend(
            handles=phylum_legend,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.85),
            fontsize=13,
            title="Phylum",
            title_fontsize=16,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        
        # Add the first legend back
        legend_ax.add_artist(legend1)
        
        # Adjust layout with more space between subplots
        plt.tight_layout(pad=2.5)
        
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
