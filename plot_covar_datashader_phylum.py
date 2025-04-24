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
import itertools
from matplotlib.colors import rgb2hex
import numba as nb
from numba import types
from numba.typed import Dict
import matplotlib.colors as mcolors
import holoviews as hv
from holoviews.operation.datashader import datashade, shade
hv.extension('bokeh')
matplotlib.use('agg')

def get_phylum_colors():
    """Return TOL 27 color palette for phylum pairs"""
    return [
        '#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499',
        '#882255', '#6699CC', '#661100', '#DD6677', '#AA4466', '#4477AA', '#228833',
        '#CCBB44', '#EE8866', '#BBCC33', '#AAAA00', '#EEDD88', '#FFAABB', '#77AADD',
        '#99DDFF', '#44BB99', '#DDDDDD', '#000000', '#F0E442', '#BBBBBB'
    ]


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
        if phylum_i > phylum_j:
            pair_key = (phylum_j, phylum_i)
        else:
            pair_key = (phylum_i, phylum_j)
            
        color_indices[k] = pair_to_color_idx[pair_key]
        
    return color_indices

def plot_dms_datashader(dm1, dm2, dm1_name='dm1', dm2_name='dm2', gene_name='gene', ax=None, fig=None, taxa_df=None):
    """Create a datashader plot comparing two distance matrices with phylum-based coloring."""
    print(f"Starting plot for {gene_name}...")
    
    # Align both dms - use only first 13 chars
    dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:13]
    dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:13]
    print(f"  - Aligned distance matrices for {gene_name}")
    
    # Get numerical arrays and flatten upper triangle
    dm1_array = dm1.iloc[:, 1:].to_numpy()
    dm2_array = dm2.iloc[:, 1:].to_numpy()
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]
    print(f"  - Extracted {len(dm1_flat)} pairwise distances for {gene_name}")
    
    # Create a DataFrame with the coordinates and organism IDs for coloring
    accessions = dm1.iloc[:, 0].values
    org_ids_rows = [accessions[i] for i in rows]
    org_ids_cols = [accessions[j] for j in cols]
    
    # Create DataFrame more efficiently
    print(f"  - Creating dataframe for {gene_name}...")
    df = pd.DataFrame({
        'x': dm1_flat,
        'y': dm2_flat,
        'org_id_row': org_ids_rows,
        'org_id_col': org_ids_cols
    })

    # OPTIMIZATION: Downsample if too many points to improve speed
    n_points = len(df)
    sample_size = min(n_points, 100000)  # Cap at 100k points for performance
    if n_points > sample_size:
        print(f"  - Downsampling from {n_points} to {sample_size} points for performance")
        df = df.sample(sample_size, random_state=42)
    
    min_val_x = df['x'].min()
    max_val_x = df['x'].max()
    min_val_y = df['y'].min()
    max_val_y = df['y'].quantile(0.99999)  # Use 99th percentile instead of scaling max value
    
    # Filter points exceeding the y-axis limit
    df = df[df['y'] <= max_val_y]
    print(f"  - Set plot range: x=[{min_val_x:.4f}, {max_val_x:.4f}], y=[{min_val_y:.4f}, {max_val_y:.4f}]")
    
    # Add phylum information
    if taxa_df is not None:
        print(f"  - Adding phylum information for {gene_name}...")
        
        # OPTIMIZATION: Convert taxa_df index to dictionary for faster lookups
        acc_to_phylum = dict(zip(taxa_df['Accession'].str[:13], taxa_df['Phylum']))
        
        # OPTIMIZATION: Use vectorized operations when possible
        print(f"  - Mapping phyla to {len(df)} data points...")
        df['phylum_row'] = pd.Series(df['org_id_row'].map(acc_to_phylum)).fillna('Unknown')
        df['phylum_col'] = pd.Series(df['org_id_col'].map(acc_to_phylum)).fillna('Unknown')
        
        # OPTIMIZATION: Vectorized phylum pair creation
        print(f"  - Creating phylum pairs...")
        df['phylum_pair'] = df.apply(
            lambda x: f"{min(x['phylum_row'], x['phylum_col'])}_{max(x['phylum_row'], x['phylum_col'])}", 
            axis=1
        )
        
        # Convert to categorical and count unique pairs
        df['phylum_pair'] = df['phylum_pair'].astype('category')
        unique_pair_count = len(df['phylum_pair'].cat.categories)
        print(f"  - Found {unique_pair_count} unique phylum pairs for {gene_name}")
        
        # Get unique phylum pairs and map to colors
        unique_pairs = sorted(df['phylum_pair'].cat.categories)
        phylum_colors = get_phylum_colors()
        color_lookup = {pair: phylum_colors[i % len(phylum_colors)] 
                      for i, pair in enumerate(unique_pairs)}
        
        # Calculate aspect ratio based on data ranges
        y_range = max_val_y - min_val_y
        x_range = max_val_x - min_val_x
        aspect_ratio = x_range / y_range
        
        # OPTIMIZATION: Reduce resolution for faster rendering
        plot_width = 800  # Reduced from 1000
        plot_height = int(plot_width/aspect_ratio)
        print(f"  - Creating canvas with dimensions {plot_width}x{plot_height}...")
        
        # Create image with fast method
        print(f"  - Rendering datashader plot for {gene_name}...")
        start_time = pd.Timestamp.now()
        
        # Use direct datashader approach - faster than the try/except block
        canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                          x_range=(min_val_x, max_val_x),
                          y_range=(min_val_y, max_val_y))
        
        # OPTIMIZATION: Use spread_points for faster aggregation with colors
        df_subset = df[['x', 'y', 'phylum_pair']]  # Use only needed columns
        
        agg = canvas.points(df_subset, 'x', 'y', ds.count_cat('phylum_pair'))
        img = ds.tf.shade(agg, color_key=color_lookup)
        img = ds.tf.set_background(img, 'white')
        
        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        print(f"  - Rendering completed in {duration:.2f} seconds")
        
        # Display the image
        ax.imshow(img.to_pil(), extent=[min_val_x, max_val_x, min_val_y, max_val_y], aspect='auto')
        
        # OPTIMIZATION: Only compute legend for the top pairs to save time
        print(f"  - Creating legend with top phylum pairs...")
        top_pairs = df['phylum_pair'].value_counts().nlargest(8).index.tolist()  # Reduced from 10
        legend_colors = {pair: color_lookup[pair] for pair in top_pairs}
        
        # Parse pair names for display
        def format_pair_name(pair_str):
            parts = pair_str.split('_')
            if len(parts) == 2:
                return f"{parts[0]} - {parts[1]}"
            return pair_str
            
        handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in legend_colors.values()]
        labels = [format_pair_name(pair) for pair in legend_colors.keys()]
        
        # Only add legend to first plot
        if ax == fig.axes[0]:
            ax.legend(handles, labels, title="Top Phylum Pairs", 
                     fontsize=8, title_fontsize=10, 
                     loc='upper left', bbox_to_anchor=(1, 1))
    else:
        print(f"  - No taxonomy data provided, creating density plot instead...")
        # Fallback to original density plot if no phylum information
        # Calculate aspect ratio based on data ranges
        y_range = max_val_y - min_val_y
        x_range = max_val_x - min_val_x
        aspect_ratio = x_range / y_range
        
        # OPTIMIZATION: Reduce resolution for faster rendering
        plot_width = 800  # Reduced from 1000
        plot_height = int(plot_width/aspect_ratio)
        
        canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
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
    
    print(f"Completed plot for {gene_name}\n")
    return 1

def process_gene(gene, taxa_df):
    """Process a single gene and return the processed data"""
    print(f"\nProcessing gene: {gene}")
    start_time = pd.Timestamp.now()
    
    # Load ESM embeddings
    print(f"  - Loading embeddings for {gene}...")
    embed = np.load(f'/home/s233201/esm_runs/embeddings/{gene}.npy')
    if embed.ndim == 1:
        embed = embed.reshape(-1, 1)
        
    # Load phylogenetic distance matrix
    print(f"  - Loading phylogenetic distances for {gene}...")
    phylo = pd.read_csv(f'/home/s233201/full_dist_mats/full_mat_{gene.upper()}.csv',
                       sep='\s+', header=None, skiprows=1)
    
    # Get accessions from phylo matrix
    phylo_accessions = phylo.iloc[:, 0].values
    print(f"  - Found {len(phylo_accessions)} sequences for {gene}")
    
    # Create DataFrame with accessions and embeddings
    accessions = pd.read_csv('/home/s233201/esm_runs/clusters/phylo_clusters/lys20_dm_clusters.csv')['accession'].values
    embed_df = pd.DataFrame(embed)
    embed_df.index = accessions
    embed_df = embed_df.reindex(phylo.iloc[:, 0].values)
    
    # Convert to distance matrix using cosine distance
    print(f"  - Computing ESM distance matrix for {gene}...")
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
    
    end_time = pd.Timestamp.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Completed processing for {gene} in {duration:.2f} seconds")
    
    return dm1, phylo, dm1_name, dm2_name, gene

if __name__ == '__main__':
    print("Starting phylum-colored distance matrix comparison script")
    start_time = pd.Timestamp.now()
    
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    print(f"Analyzing {len(gene_names)} genes: {', '.join(gene_names)}")
    
    # Create a large figure with 4x2 subplots and space for colorbar
    print("Creating figure...")
    fig = plt.figure(figsize=(20, 24))  # Increased width to accommodate legend
    gs = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

    print("Loading taxonomy data...")
    taxa_df = pd.read_csv('/home/s233201/esm_runs/inputs/taxa.csv')
    print(f"Found {len(taxa_df)} taxonomy records")
    
    # Process all genes first
    print("\nPre-processing all genes...")
    all_results = []
    for i, gene in enumerate(gene_names):
        print(f"\nProcessing gene {i+1}/{len(gene_names)}: {gene}")
        result = process_gene(gene, taxa_df)
        all_results.append(result)
    
    # Plot all genes with phylum-based coloring
    print("\nGenerating plots for all genes...")
    for idx, (dm1, dm2, dm1_name, dm2_name, gene) in enumerate(all_results):
        row = idx // 2
        col = idx % 2
        print(f"\nPlotting gene {idx+1}/{len(all_results)}: {gene} (position: row {row+1}, col {col+1})")
        ax = fig.add_subplot(gs[row, col])
        plot_dms_datashader(dm1, dm2, dm1_name, dm2_name, gene, ax=ax, fig=fig, taxa_df=taxa_df)
    
    # Adjust layout and save
    print("\nFinalizing figure...")
    plt.tight_layout()
    
    # Create output directory if needed
    output_dir = 'esm_runs/plots/covars_datashader'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/all_genes_comparison_phylum.png'
    print(f"Saving figure to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    end_time = pd.Timestamp.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nScript completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")