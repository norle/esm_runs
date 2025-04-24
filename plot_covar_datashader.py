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
    # Align both dms - use only first 13 chars
    dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:13]
    dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:13]
    
    # Get numerical arrays and flatten upper triangle
    dm1_array = dm1.iloc[:, 1:].to_numpy()
    dm2_array = dm2.iloc[:, 1:].to_numpy()
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
    embed = np.load(f'/home/s233201/esm_runs/embeddings/{gene}.npy')
    if embed.ndim == 1:
        embed = embed.reshape(-1, 1)
        
    # Load phylogenetic distance matrix
    phylo = pd.read_csv(f'/home/s233201/full_dist_mats/full_mat_{gene.upper()}.csv',
                       sep='\s+', header=None, skiprows=1)
    
    # Get accessions from phylo matrix
    phylo_accessions = phylo.iloc[:, 0].values
    
    # Create DataFrame with accessions and embeddings
    accessions = pd.read_csv('/home/s233201/esm_runs/clusters/phylo_clusters/lys20_dm_clusters.csv')['accession'].values
    embed_df = pd.DataFrame(embed)
    embed_df.index = accessions
    embed_df = embed_df.reindex(phylo.iloc[:, 0].values)
    
    # Convert to distance matrix using cosine distance
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
    
    return dm1, phylo, dm1_name, dm2_name, gene

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
