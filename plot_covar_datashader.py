import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datashader as ds
from datashader.colors import colormap_select
import colorcet as cc
import os
from functools import partial
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform
matplotlib.use('agg')

def plot_dms_datashader(dm1, dm2, dm1_name='dm1', dm2_name='dm2', gene_name='gene', ax=None, fig=None):
    # Ensure the first column is the index for alignment
    if dm1.columns[0] != 'accession':
        dm1 = dm1.set_index(dm1.columns[0])
    else:
        dm1 = dm1.set_index('accession')

    if dm2.columns[0] != 'accession':
        dm2 = dm2.set_index(dm2.columns[0])
    else:
        dm2 = dm2.set_index('accession')

    # Find common accessions
    common_accessions = dm1.index.intersection(dm2.index)

    # Reindex both dataframes (rows and columns) to align them
    dm1_aligned = dm1.loc[common_accessions, common_accessions]
    dm2_aligned = dm2.loc[common_accessions, common_accessions]

    # Get numerical arrays and flatten upper triangle
    dm1_array = dm1_aligned.to_numpy()
    dm2_array = dm2_aligned.to_numpy()
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]

    df = pd.DataFrame({
        'x': dm1_flat,
        'y': dm2_flat
    })

    min_val_x = df['x'].min()
    max_val_x = df['x'].max()
    min_val_y = df['y'].min()
    max_val_y = df['y'].quantile(0.99999)  # Use 99th percentile instead of scaling max value
    
    # Filter points exceeding the y-axis limit
    df = df[df['y'] <= max_val_y]
    
    # Calculate aspect ratio based on data ranges
    y_range = max_val_y - min_val_y
    x_range = max_val_x - min_val_x
    aspect_ratio = x_range / y_range
    
    # Create main subplot for datashader plot with data-driven x range
    canvas = ds.Canvas(plot_width=1000, plot_height=int(1000/aspect_ratio),
                      x_range=(min_val_x, max_val_x),
                      y_range=(min_val_y, max_val_y))
    
    # Create density plot
    agg = canvas.points(df, 'x', 'y')
    img = ds.tf.shade(agg, cmap=cc.fire)
    img = ds.tf.set_background(img, 'white')
    
    # Convert to matplotlib figure and set aspect to auto
    ax.imshow(img.to_pil(), extent=[min_val_x, max_val_x, min_val_y, max_val_y], aspect='auto')
    
    # Increase font sizes
    ax.set_xlabel(f'ESM-C distance', fontsize=20)
    ax.set_ylabel(f'Phylogenetic distance', fontsize=20)
    ax.set_title(gene_name, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    return agg.values.max()

def process_gene(gene):
    """Process a single gene and return the processed data"""
    print(f"Processing gene: {gene}")

    # Load ESM embeddings
    embed = np.load(f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_embeddings.npy')
    if embed.ndim == 1:
        embed = embed.reshape(-1, 1)

    # Load phylogenetic distance matrix
    phylo_raw = pd.read_csv(f'/home/s233201/full_dist_mats/new/full_mat_{gene.upper()}.csv',
                       sep='\s+', header=None, skiprows=1)

    # Get accessions from phylo matrix and set as index/columns
    phylo_accessions = phylo_raw.iloc[:, 0].values
    phylo = pd.DataFrame(phylo_raw.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)

    # Load accessions used for embeddings (assuming these match the order in the .npy file)
    # It's crucial that the order of accessions here matches the order of embeddings in the .npy file
    # If not, load the accessions corresponding to the .npy file correctly.
    # For this example, assuming 'lys20_dm_clusters.csv' contains the correct ordered accessions for the embeddings.
    # A safer approach would be to save accessions alongside embeddings.
    embed_accessions = []
    with open(f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_ids.txt', 'r') as f:
        embed_accessions = [line.strip() for line in f.readlines()[:embed.shape[0]]]  # Ensure we only take as many accessions as embeddings

    # Create DataFrame with accessions and embeddings
    embed_df = pd.DataFrame(embed, index=embed_accessions)

    # Convert to distance matrix using cosine distance
    embed_dist = pd.DataFrame(
        squareform(pdist(embed_df.values, metric='cosine')),
        index=embed_accessions,
        columns=embed_accessions
    )

    # Reset index to add 'accession' column for plot_dms_datashader compatibility
    dm1 = embed_dist.reset_index().rename(columns={'index': 'accession'})
    dm2 = phylo.reset_index().rename(columns={'index': 'accession'})

    dm1_name = f'{gene} ESM distances'
    dm2_name = f'{gene} phylogenetic distances'

    return dm1, dm2, dm1_name, dm2_name, gene

if __name__ == '__main__':
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    
    # Create a large figure with 4x2 subplots and space for colorbar
    fig = plt.figure(figsize=(16, 24))  # Adjusted for 4x2 layout
    gs = plt.GridSpec(4, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1, 1, 1])
    
    # Process all genes first
    all_results = []
    for gene in gene_names:
        result = process_gene(gene)
        all_results.append(result)
    
    # Plot all genes and track max density
    max_density = 0
    for idx, (dm1, dm2, dm1_name, dm2_name, gene) in enumerate(all_results):
        row = idx // 2  # Changed from 4 to 2 for new layout
        col = idx % 2   # Changed from 4 to 2 for new layout
        ax = fig.add_subplot(gs[row, col])
        max_val = plot_dms_datashader(dm1, dm2, dm1_name, dm2_name, gene, ax=ax, fig=fig)
        max_density = max(max_density, max_val)
    
    # Add single colorbar for all plots with larger font
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_density)
    fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fire', cc.fire)
    sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=norm)
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(sm, cax=cax, label='Density')
    cbar.ax.tick_params(labelsize=16)  # Increase colorbar tick font size
    cbar.set_label('Density', size=20)  # Increase colorbar label font size
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs('esm_runs/plots/covars_datashader', exist_ok=True)
    plt.savefig('esm_runs/plots/covars_datashader/all_genes_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
