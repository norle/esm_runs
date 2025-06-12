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
import scipy.stats as stats  # Add at the top with other imports
matplotlib.use('agg')

def plot_dms_datashader(dm1, dm2, dm1_name='dm1', dm2_name='dm2', gene_name='gene', ax=None, fig=None, x_range=None, y_range=None):
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

    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(df['x'], df['y'])
    
    # Format p-value string
    if p_value == 0.0:
        p_value_str = "p < 1e-16"
    elif p_value < 0.001:
        p_value_str = f"p = {p_value:.2e}"
    else:
        p_value_str = f"p = {p_value:.3f}"

    # Use provided ranges or calculate from data
    if x_range is None:
        min_val_x = df['x'].min()
        max_val_x = df['x'].max()
    else:
        min_val_x, max_val_x = x_range
        
    if y_range is None:
        min_val_y = df['y'].min()
        max_val_y = df['y'].quantile(0.99999)
    else:
        min_val_y, max_val_y = y_range
    
    # Filter points exceeding the axis limits
    df = df[(df['y'] <= max_val_y) & (df['x'] <= max_val_x) & (df['x'] >= min_val_x) & (df['y'] >= min_val_y)]
    
    # Create datashader canvas
    canvas = ds.Canvas(plot_width=100, plot_height=100,
                      x_range=(min_val_x, max_val_x),
                      y_range=(min_val_y, max_val_y))
    
    # Create density plot
    agg = canvas.points(df, 'x', 'y')
    img = ds.tf.shade(agg, cmap=cc.fire)
    img = ds.tf.set_background(img, 'white')
    
    # Convert to matplotlib figure
    ax.imshow(img.to_pil(), extent=[min_val_x, max_val_x, min_val_y, max_val_y], aspect='auto')
    
    # Set consistent axis limits
    ax.set_xlim(min_val_x, max_val_x)
    ax.set_ylim(min_val_y, max_val_y)
    
    # Increase font sizes
    ax.set_xlabel(f'ESM-C distance', fontsize=16)
    ax.set_ylabel(f'Phylogenetic distance', fontsize=16)
    ax.set_title(gene_name.upper(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add correlation text to plot
    correlation_text = f"r = {correlation:.3f}\n{p_value_str}"
    ax.text(0.95, 0.95, correlation_text,
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    return agg.values.max(), df

def process_gene(gene):
    """Process a single gene and return the processed data"""
    print(f"Processing gene: {gene}")

    # Load ESM embeddings
    embed = np.load(f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_embeddings.npy')
    if embed.ndim == 1:
        embed = embed.reshape(-1, 1)

    # Load phylogenetic distance matrix
    phylo_raw = pd.read_csv(f'/home/s233201/full_dist_mats/clean/full_mat_{gene.upper()}.csv',
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

def plot_single_gene(args):
    """Plot a single gene subplot and return the result"""
    dm1, dm2, dm1_name, dm2_name, gene, x_range, y_range, subplot_params = args
    
    # Create a temporary figure for this subplot
    temp_fig, temp_ax = plt.subplots(figsize=(2, 2))
    
    max_val, df = plot_dms_datashader(dm1, dm2, dm1_name, dm2_name, gene, 
                                    ax=temp_ax, fig=temp_fig, 
                                    x_range=x_range, y_range=y_range)
    
    # Save the subplot data
    subplot_data = {
        'max_val': max_val,
        'df': df,
        'gene': gene,
        'subplot_params': subplot_params
    }
    
    plt.close(temp_fig)
    return subplot_data

if __name__ == '__main__':
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    
    # Use multiprocessing Pool to process genes in parallel
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        all_results = pool.map(process_gene, gene_names)
    
    # Calculate global axis ranges in parallel
    def extract_ranges(result):
        dm1, dm2, dm1_name, dm2_name, gene = result
        # Process data to get x,y values for range calculation
        if dm1.columns[0] != 'accession':
            dm1_temp = dm1.set_index(dm1.columns[0])
        else:
            dm1_temp = dm1.set_index('accession')

        if dm2.columns[0] != 'accession':
            dm2_temp = dm2.set_index(dm2.columns[0])
        else:
            dm2_temp = dm2.set_index('accession')

        common_accessions = dm1_temp.index.intersection(dm2_temp.index)
        dm1_aligned = dm1_temp.loc[common_accessions, common_accessions]
        dm2_aligned = dm2_temp.loc[common_accessions, common_accessions]
        
        dm1_array = dm1_aligned.to_numpy()
        dm2_array = dm2_aligned.to_numpy()
        rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
        
        return dm1_array[rows, cols], dm2_array[rows, cols]
    
    with mp.Pool(processes=num_processes) as pool:
        range_results = pool.map(extract_ranges, all_results)
    
    # Combine all x,y values
    all_x_vals = []
    all_y_vals = []
    for x_vals, y_vals in range_results:
        all_x_vals.extend(x_vals)
        all_y_vals.extend(y_vals)
    
    # Calculate global ranges
    global_x_min = min(all_x_vals)
    global_x_max = max(all_x_vals)
    global_y_min = min(all_y_vals)
    global_y_max = np.percentile(all_y_vals, 99.999)  # Use percentile to avoid outliers
    
    x_range = (global_x_min, global_x_max)
    y_range = (global_y_min, global_y_max)
    
    # Calculate figure size for 3x3 grid with 200x200 pixel subplots
    # Each subplot needs space for main plot (200px) + histogram (50px) + margins
    subplot_size_inches = 200 / 100  # 200 pixels at 100 DPI = 2 inches
    hist_size_inches = 50 / 100      # 50 pixels for histograms
    margin_inches = 0.3              # Margin between subplots
    
    # Total size per subplot group (main + histograms + margins)
    total_subplot_width = subplot_size_inches + hist_size_inches + margin_inches
    total_subplot_height = subplot_size_inches + hist_size_inches + margin_inches
    
    # Figure dimensions
    fig_width = 3 * total_subplot_width + 1.5  # +1.5 for colorbar space
    fig_height = 3 * total_subplot_height + 0.5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Calculate subplot positions manually for precise alignment
    subplot_width = subplot_size_inches / fig_width
    subplot_height = subplot_size_inches / fig_height
    hist_width = hist_size_inches / fig_width
    hist_height = hist_size_inches / fig_height
    margin_w = margin_inches / fig_width
    margin_h = margin_inches / fig_height
    
    # Prepare arguments for parallel plotting
    plot_args = []
    for idx, (dm1, dm2, dm1_name, dm2_name, gene) in enumerate(all_results):
        row = idx // 3
        col = idx % 3
        
        # Calculate subplot positions
        left = col * (subplot_width + hist_width + margin_w) + margin_w/2
        bottom = (2-row) * (subplot_height + hist_height + margin_h) + margin_h/2
        
        subplot_params = {
            'idx': idx, 'row': row, 'col': col,
            'left': left, 'bottom': bottom,
            'subplot_width': subplot_width, 'subplot_height': subplot_height,
            'hist_width': hist_width, 'hist_height': hist_height
        }
        
        plot_args.append((dm1, dm2, dm1_name, dm2_name, gene, x_range, y_range, subplot_params))
    
    # Process plots in parallel
    with mp.Pool(processes=num_processes) as pool:
        subplot_results = pool.map(plot_single_gene, plot_args)
    
    # Plot all genes and track max density
    max_density = 0
    
    for subplot_data in subplot_results:
        max_val = subplot_data['max_val']
        df = subplot_data['df']
        gene = subplot_data['gene']
        params = subplot_data['subplot_params']
        
        row = params['row']
        col = params['col']
        left = params['left']
        bottom = params['bottom']
        subplot_width = params['subplot_width']
        subplot_height = params['subplot_height']
        hist_width = params['hist_width']
        hist_height = params['hist_height']
        
        # Main plot
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
        
        # Get the corresponding data
        dm1, dm2, dm1_name, dm2_name, _ = all_results[params['idx']]
        max_val, df = plot_dms_datashader(dm1, dm2, dm1_name, dm2_name, gene, ax=ax, fig=fig, 
                                        x_range=x_range, y_range=y_range)
        max_density = max(max_density, max_val)
        
        # Hide main plot spines adjacent to histograms for seamless look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Top histogram (x-axis) - directly above main plot
        ax_histx = fig.add_axes([left, bottom + subplot_height, subplot_width, hist_height])
        ax_histx.hist(df['x'], bins=50, alpha=0.7, color='skyblue', density=True)
        ax_histx.set_xlim(x_range)
        ax_histx.set_xticks([])
        ax_histx.set_yticks([])
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['bottom'].set_visible(False)
        
        # Right histogram (y-axis) - directly to the right of main plot
        ax_histy = fig.add_axes([left + subplot_width, bottom, hist_width, subplot_height])
        ax_histy.hist(df['y'], bins=50, alpha=0.7, color='lightcoral', orientation='horizontal', density=True)
        ax_histy.set_ylim(y_range)
        ax_histy.set_xticks([])
        ax_histy.set_yticks([])
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
    
    # Add single colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_density)
    fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fire', cc.fire)
    sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=norm)
    
    # Position colorbar on the right side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax, label='Density')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Density', size=16)
    
    # Save with high DPI to ensure 200x200 pixel subplots
    os.makedirs('esm_runs/plots/covars_datashader', exist_ok=True)
    plt.savefig('esm_runs/plots/covars_datashader/all_genes_comparison.png',
                dpi=100, bbox_inches='tight')  # 100 DPI gives us exactly 200x200 pixels for 2x2 inch subplots
    plt.close()
