import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import rgb2hex
import numba as nb
from numba import types
from numba.typed import Dict, List
import matplotlib.colors as mcolors
import os
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import multiprocessing as mp
from functools import partial
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('agg')  # Add this at the top after import


@nb.njit
def get_colors_numba(rows, cols, phyla_indices, pair_to_color_idx):
    """Numba-accelerated function to determine colors based on phylum pairs"""
    n = len(rows)
    color_indices = np.empty(n, dtype=np.int64)
    
    for k in range(n):
        i, j = rows[k], cols[k]
        phylum_i = phyla_indices[i]
        phylum_j = phyla_indices[j]
        
        # Ensure consistent ordering for pair lookup
        if (phylum_i > phylum_j):
            pair_key = (phylum_j, phylum_i)
        else:
            pair_key = (phylum_i, phylum_j)
            
        color_indices[k] = pair_to_color_idx[pair_key]
        
    return color_indices


def plot_dms(dm1, dm2, dm1_name='dm1', dm2_name='dm2', gene_name='gene'):
    # Align both dms - use only first 14 chars
    dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:13]
    dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:13]
    
    dm1_order = dm1.iloc[:, 0].tolist()
    
    # Create mapping for reordering dm2
    dm2_order = dm2.iloc[:, 0].tolist()
    order_mapping = [dm2_order.index(x) for x in dm1_order]
    
    # Reorder rows of dm2
    dm2 = dm2.iloc[order_mapping]
    
    # Reorder columns of dm2 (excluding first column)
    dm2_cols = dm2.columns.tolist()
    dm2 = dm2[[dm2_cols[0]] + [dm2_cols[i+1] for i in order_mapping]]

    # Continue with numerical part extraction
    accessions = dm1.iloc[:, 0].tolist()  # Save accessions before removing the column
    dm1 = dm1.iloc[:, 1:]
    dm2 = dm2.iloc[:, 1:]

    # Convert to numpy arrays
    dm1_array = dm1.to_numpy()
    dm2_array = dm2.to_numpy()
    
    # Load taxa information to get phyla - use first 14 chars for matching
    taxa_df = pd.read_csv('/home/s233201/esm_runs/inputs/taxa.csv')
    taxa_dict = dict(zip(taxa_df['Accession'], taxa_df['Phylum']))
    
    # Map accessions to phyla
    phyla = [taxa_dict.get(acc, "Unknown") for acc in accessions]
    
    # Get unique phylum pairs and assign colors
    unique_phyla = sorted(set(phyla))
    phyla_pairs = list(itertools.combinations_with_replacement(unique_phyla, 2))
    
    # Create mapping from phylum name to index for numba
    phylum_to_idx = {phylum: i for i, phylum in enumerate(unique_phyla)}
    phyla_indices = np.array([phylum_to_idx[p] for p in phyla], dtype=np.int64)
    
    # TOL 27 color palette (colorblind-friendly)
    tol_27_colors = [
        '#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499',
        '#882255', '#6699CC', '#661100', '#DD6677', '#AA4466', '#4477AA', '#228833',
        '#CCBB44', '#EE8866', '#BBCC33', '#AAAA00', '#EEDD88', '#FFAABB', '#77AADD',
        '#99DDFF', '#44BB99', '#DDDDDD', '#000000', '#FFFFFF', '#BBBBBB'
    ]
    
    # Create color list for phylum pairs, cycling if needed
    color_list = []
    for i in range(len(phyla_pairs)):
        color_idx = i % len(tol_27_colors)
        color_list.append(tol_27_colors[color_idx])
    
    # Create numba-compatible mapping from pair indices to color indices
    pair_to_color_idx = Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.int64
    )
    
    for i, pair in enumerate(phyla_pairs):
        idx_pair = (phylum_to_idx[pair[0]], phylum_to_idx[pair[1]])
        pair_to_color_idx[idx_pair] = i
    
    print(f"Number of unique phyla: {len(unique_phyla)}")
    print(f"Number of possible phylum pairs: {len(phyla_pairs)}")

    # Get upper triangle indices with the original i, j positions
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    
    # Extract distances using these indices
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]
    
    # Use numba-accelerated function to get color indices
    color_indices = get_colors_numba(rows, cols, phyla_indices, pair_to_color_idx)
    
    # Map color indices back to actual colors
    colors = [color_list[idx] for idx in color_indices]
    
    print(dm1_array.shape)
    print(dm2_array.shape)
    print(dm1_flat.shape)
    print(dm2_flat.shape)
    
    # Create figure with gridspec - adjusted height ratios for bigger histograms
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(3, 3, 
                 width_ratios=[1, 1, 0.3], 
                 height_ratios=[0.5, 1, 1],  # Changed from [0.3, 1, 1] to [0.5, 1, 1]
                 hspace=0.05, 
                 wspace=0.05)
    
    # Create axis for scatter plot and histograms
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_hist_x = fig.add_subplot(gs[0, :-1])
    ax_hist_y = fig.add_subplot(gs[1:, -1])
    
    # Plot scatter - reduced point size and alpha
    scatter = ax_scatter.scatter(dm1_flat, dm2_flat, s=0.5, c=colors, alpha=0.3)
    
    # Plot histograms with log scale
    bins = min(50, len(dm1_flat) // 100)  # Adaptive bin size
    ax_hist_x.hist(dm1_flat, bins=bins, color='gray', alpha=0.3, log=True)
    ax_hist_y.hist(dm2_flat, bins=bins, orientation='horizontal', color='gray', alpha=0.3, log=True)
    
    # Set labels
    ax_scatter.set_xlabel(dm1_name)
    ax_scatter.set_ylabel(dm2_name)
    
    # Remove histogram axis labels
    ax_hist_x.xaxis.set_visible(False)
    ax_hist_y.yaxis.set_visible(False)
    
    # Align histogram axes with scatter plot
    ax_hist_x.set_xlim(ax_scatter.get_xlim())
    ax_hist_y.set_ylim(ax_scatter.get_ylim())
    
    # Add title to main plot
    fig.suptitle('Distance Matrix Comparison Colored by Phylum Pairs')
    
    # Simplified legend with new position - inside plot
    if len(phyla_pairs) > 10:
        selected_pairs = phyla_pairs[:10]  # Show only first 10 pairs
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                label=f"{pair[0]}-{pair[1]}",
                                markerfacecolor=color_list[i], markersize=4)
                          for i, pair in enumerate(selected_pairs)]
    else:
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                label=f"{pair[0]}-{pair[1]}",
                                markerfacecolor=color_list[i], markersize=4)
                          for i, pair in enumerate(phyla_pairs)]
    
    # Move legend inside plot with increased size
    ax_scatter.legend(handles=legend_elements, title="Phylum Pairs", 
                     loc='upper right',
                     prop={'size': 10}, title_fontsize=12,  # Increased font sizes by 1.5
                     bbox_to_anchor=(0.98, 0.98))
    
    # Fast save with lower dpi and compression
    plt.savefig(f'esm_runs/plots/covars/dm_comparison{gene_name}.png', 
                dpi=300, bbox_inches='tight', 
                pad_inches=0.1)
    plt.close('all')  # Ensure all figures are closed


def process_gene(gene, accessions, to_remove, to_remove_indices):
    """Process a single gene and create its plot"""
    print(f"Processing gene: {gene}")
    
    # Load ESM embeddings
    embed = np.load(f'/home/s233201/esm_runs/embeddings/{gene}.npy')
    if embed.ndim == 1:
        embed = embed.reshape(-1, 1)
        
    # Load phylogenetic distance matrix
    phylo = pd.read_csv(f'/home/s233201/full_dist_mats/full_mat_{gene.upper()}.csv', 
                       sep='\s+', header=None, skiprows=1)
    
    # Get accessions from phylo matrix and filter out unwanted ones
    phylo_accessions = phylo.iloc[:, 0].values
    keep_indices = [i for i, acc in enumerate(phylo_accessions) if acc not in to_remove]
    phylo = phylo.iloc[keep_indices]
    phylo = phylo[[0] + [i+1 for i in keep_indices]]
    
    # Create DataFrame with accessions and embeddings
    embed_df = pd.DataFrame(embed)

    # Remove rows corresponding to removed accessions
    embed_df = embed_df.drop(index=to_remove_indices)
    embed_df.index = accessions

    # Reindex embeddings to match filtered phylo matrix order
    embed_df = embed_df.reindex(phylo.iloc[:, 0].values)
    
    # Convert to distance matrix using cosine distance
    from scipy.spatial.distance import pdist, squareform
    embed_dist = pd.DataFrame(
        squareform(pdist(embed_df.values, metric='cosine')),
        index=phylo.iloc[:, 0].values,
        columns=phylo.iloc[:, 0].values
    )
    
    # Create DataFrames for plot_dms
    dm1 = pd.DataFrame(embed_dist)
    dm1.insert(0, 'accession', dm1.index)
    
    dm1_name = f'{gene} ESM distances'
    dm2_name = f'{gene} phylogenetic distances'
    
    # Create directory for covariance plots
    os.makedirs('figures/covars', exist_ok=True)
    
    plot_dms(dm1, phylo, dm1_name, dm2_name, gene)
    print(f"Completed processing gene: {gene}")

if __name__ == '__main__':
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    
    with open('/home/s233201/esm_runs/inputs/all_missing.txt', 'r') as f:
        to_remove = [line.strip() for line in f.readlines()]

    # Get reference accession order
    accessions = pd.read_csv('/home/s233201/esm_runs/clusters/phylo_clusters/lys20_dm_clusters.csv')['accession'].values
    
    # Get indices of accessions to remove
    to_remove_indices = [i for i, acc in enumerate(accessions) if acc in to_remove]
    # Remove accessions that are in the to_remove list
    accessions = [acc for acc in accessions if acc not in to_remove]
    
    # Set up multiprocessing
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    print(f"Using {num_processes} processes")
    
    # Create partial function with fixed arguments
    process_gene_partial = partial(
        process_gene,
        accessions=accessions,
        to_remove=to_remove,
        to_remove_indices=to_remove_indices
    )
    
    # Create pool and process genes in parallel
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_gene_partial, gene_names)

