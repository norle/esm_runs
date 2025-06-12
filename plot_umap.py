import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
import os
import pandas as pd
from Bio import SeqIO
import multiprocessing
import pickle
from adjustText import adjust_text  # Add this import

# Define phylum colors using ColorBrewer
PHYLUM_COLORS = {
    'Ascomycota': '#377eb8',    # Blue
    'Basidiomycota': '#e41a1c',       # Red
    'Mucoromycota': '#4daf4a',     # Green
    'Zoopagomycota': '#984ea3',    # Purple
    'Chytridiomycota': '#ff7f00',  # Orange
    'Blastocladiomycota': '#ffff33',# Yellow
    #'Cryptomycota': '#a65628'      # Brown
}

# # Add special organisms for labeling
# SPECIAL_ORGANISMS = {
#     "GCF_000146045.2": "Saccharomyces cerevisiae",
#     "GCA_000230395.2": "Aspergillus niger",
#     "GCF_000002655.1": "Aspergillus fumigatus",
#     "GCF_000182895.1": "Coprinopsis cinerea",
#     "GCF_000149305.1": "Rhizopus delemar",
#     "GCF_028827035.1": "Penicillium chrysogenum"
# }
# Add special organisms for labeling
SPECIAL_ORGANISMS = {
    "GCF_000146045": "S. cerevisiae",
    "GCA_000230395": "A. niger",
    "GCF_000002655": "A. fumigatus",
    "GCF_000182895": "C. cinerea",
    "GCF_000149305": "R. delemar",
    "GCF_028827035": "P. chrysogenum"
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
    texts = []
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    for i, acc in enumerate(accessions):
        base_acc = acc.split('.')[0]
        if base_acc in SPECIAL_ORGANISMS:
            x, y = umap_coords[i, 0], umap_coords[i, 1]
            
            # Create text annotation without arrow initially
            text = ax.text(x, y, SPECIAL_ORGANISMS[base_acc],
                         fontsize=fontsize,
                         ha='center',
                         va='bottom',
                         color='black',
                         zorder=5)
            texts.append(text)
    
    # Adjust text positions to avoid overlaps with arrows for all labels
    adjust_text(texts,
               ax=ax,
               arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.7),
               expand_points=(1.5, 1.5),
               force_points=(0.1, 0.1),
               force_text=(0.5, 0.5),
               lim=500)  # Increase iterations for better placement

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

def load_outlier_accessions():
    """Load outlier accessions from the text file."""
    with open('/home/s233201/outliers_set.txt', 'r') as f:
        return {line.strip().split('.')[0] for line in f}

def calculate_embedding_diversity(embeddings, metric='cosine'):
    """
    Calculate diversity metrics for protein embeddings.
    
    Args:
        embeddings: numpy array of shape (n_proteins, embedding_dim)
        metric: distance metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        dict with diversity metrics
    """
    if metric == 'cosine':
        # Calculate cosine distances (1 - cosine similarity)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
    else:
        distance_matrix = pairwise_distances(embeddings, metric=metric)
    
    # Remove diagonal (self-distances)
    n = distance_matrix.shape[0]
    upper_triangle = distance_matrix[np.triu_indices(n, k=1)]
    
    diversity_metrics = {
        'mean_distance': np.mean(upper_triangle),
        'std_distance': np.std(upper_triangle),
        'median_distance': np.median(upper_triangle),
        'max_distance': np.max(upper_triangle),
        'min_distance': np.min(upper_triangle),
        'cv_distance': np.std(upper_triangle) / np.mean(upper_triangle) if np.mean(upper_triangle) > 0 else 0,
        'iqr_distance': np.percentile(upper_triangle, 75) - np.percentile(upper_triangle, 25),
        'dispersion_index': np.var(upper_triangle) / np.mean(upper_triangle) if np.mean(upper_triangle) > 0 else 0
    }
    
    return diversity_metrics

def calculate_phylum_diversity(embeddings, phyla, metric='cosine'):
    """
    Calculate diversity metrics within and between phylums.
    
    Returns:
        dict with within-phylum and between-phylum diversity
    """
    unique_phyla = list(set(phyla))
    phylum_indices = {phylum: [i for i, p in enumerate(phyla) if p == phylum] 
                      for phylum in unique_phyla}
    
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
    else:
        distance_matrix = pairwise_distances(embeddings, metric=metric)
    
    within_phylum_diversity = {}
    for phylum, indices in phylum_indices.items():
        if len(indices) > 1:
            phylum_distances = distance_matrix[np.ix_(indices, indices)]
            upper_triangle = phylum_distances[np.triu_indices(len(indices), k=1)]
            within_phylum_diversity[phylum] = {
                'mean_distance': np.mean(upper_triangle),
                'std_distance': np.std(upper_triangle),
                'n_sequences': len(indices)
            }
    
    # Calculate between-phylum diversity
    between_phylum_diversity = {}
    for i, phylum1 in enumerate(unique_phyla):
        for phylum2 in unique_phyla[i+1:]:
            indices1 = phylum_indices[phylum1]
            indices2 = phylum_indices[phylum2]
            between_distances = distance_matrix[np.ix_(indices1, indices2)].flatten()
            between_phylum_diversity[f"{phylum1}_vs_{phylum2}"] = {
                'mean_distance': np.mean(between_distances),
                'std_distance': np.std(between_distances)
            }
    
    return within_phylum_diversity, between_phylum_diversity

def process_gene_data(gene_name):
    """Process single gene and return UMAP coordinates, phyla, and diversity metrics"""
    embedding_file = f'/home/s233201/esm_runs/embeddings_new/{gene_name}_embeddings.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    umap_cache_file = f'/home/s233201/esm_runs/umap_cache/{gene_name}_umap.npz'
    
    print(f"Processing {gene_name}...")
    
    # Load outliers
    outliers = load_outlier_accessions()
    
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    
    # Filter out outliers
    valid_indices = [i for i, acc in enumerate(fasta_accessions) if acc.split('.')[0] not in outliers]
    filtered_accessions = [fasta_accessions[i] for i in valid_indices]
    phyla = [taxa_dict[acc] for acc in filtered_accessions if acc in taxa_dict]
    
    # Try to load cached UMAP coordinates
    if os.path.exists(umap_cache_file):
        print(f"Loading cached UMAP coordinates for {gene_name}")
        cached_data = np.load(umap_cache_file, allow_pickle=True)
        
        # Check if diversity metrics are in cache
        if 'diversity_metrics' in cached_data.files:
            diversity_metrics = cached_data['diversity_metrics'].item()
            within_phylum_div = cached_data['within_phylum_diversity'].item()
            between_phylum_div = cached_data['between_phylum_diversity'].item()
            return gene_name, cached_data['umap_coords'], phyla, filtered_accessions, diversity_metrics, within_phylum_div, between_phylum_div
        else:
            # Cache exists but without diversity metrics, need to recalculate
            print(f"Cache exists but missing diversity metrics for {gene_name}, recalculating...")
    
    # If no cache exists or cache is incomplete, proceed with full processing
    embeddings, _ = load_embeddings(embedding_file)
    # Filter embeddings
    embeddings = embeddings[valid_indices]
    
    # Calculate diversity metrics
    diversity_metrics = calculate_embedding_diversity(embeddings, metric='cosine')
    within_phylum_div, between_phylum_div = calculate_phylum_diversity(embeddings, phyla, metric='cosine')
    
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
    
    # Cache the results including diversity metrics
    os.makedirs(os.path.dirname(umap_cache_file), exist_ok=True)
    np.savez(umap_cache_file, 
             umap_coords=umap_coords,
             diversity_metrics=diversity_metrics,
             within_phylum_diversity=within_phylum_div,
             between_phylum_diversity=between_phylum_div)
    
    return gene_name, umap_coords, phyla, filtered_accessions, diversity_metrics, within_phylum_div, between_phylum_div

def save_diversity_results(results, output_file):
    """Save diversity metrics to CSV file."""
    diversity_data = []
    
    for result in results:
        if len(result) == 7:  # Full result with diversity metrics
            gene_name, _, _, _, diversity_metrics, within_phylum_div, between_phylum_div = result
        else:  # Old cached result without diversity metrics
            print(f"Warning: No diversity metrics for {result[0]}, skipping...")
            continue
            
        row = {'gene': gene_name}
        row.update(diversity_metrics)
        
        # Add within-phylum diversity averages
        if within_phylum_div:
            within_means = [metrics['mean_distance'] for metrics in within_phylum_div.values()]
            within_stds = [metrics['std_distance'] for metrics in within_phylum_div.values()]
            row['avg_within_phylum_mean'] = np.mean(within_means)
            row['avg_within_phylum_std'] = np.mean(within_stds)
        
        diversity_data.append(row)
    
    df = pd.DataFrame(diversity_data)
    df.to_csv(output_file, index=False)
    print(f"Diversity metrics saved to {output_file}")

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
        
        # Save diversity metrics to CSV
        save_diversity_results(results, '/home/s233201/esm_runs/plots/protein_diversity_metrics.csv')
        
        # Print diversity summary - handle both result formats
        print("\nDiversity Summary (mean pairwise cosine distance):")
        print("-" * 60)
        valid_results = [r for r in results if len(r) == 7]
        for gene_name, _, _, _, diversity_metrics, _, _ in valid_results:
            print(f"{gene_name:8}: {diversity_metrics['mean_distance']:.4f} ± {diversity_metrics['std_distance']:.4f}")
        
        # Sort by diversity for ranking
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x[4]['mean_distance'], reverse=True)
            print(f"\nMost diverse protein: {sorted_results[0][0]} (diversity: {sorted_results[0][4]['mean_distance']:.4f})")
            print(f"Least diverse protein: {sorted_results[-1][0]} (diversity: {sorted_results[-1][4]['mean_distance']:.4f})")
        
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
        for idx, result in enumerate(results):
            if idx >= 8:  # Only plot in the first 8 cells
                continue
            
            if len(result) == 7:  # Full result with diversity metrics
                gene_name, umap_coords, phyla, fasta_accessions, diversity_metrics, _, _ = result
                has_diversity = True
            else:  # Old cached result without diversity metrics
                gene_name, umap_coords, phyla, fasta_accessions = result
                has_diversity = False
                
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
            
            # Set title with diversity information if available
            if has_diversity:
                diversity_score = diversity_metrics['mean_distance']
                ax.set_title(f'{gene_name}\nDiversity: {diversity_score:.3f}', 
                            fontsize=16, pad=10, fontweight='bold')
            else:
                ax.set_title(f'{gene_name}', fontsize=16, pad=10, fontweight='bold')
        
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