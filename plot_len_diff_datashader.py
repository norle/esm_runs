import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datashader as ds
from datashader.colors import colormap_select
import colorcet as cc
import os
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO # Added for FASTA parsing
import concurrent.futures # Added for parallel processing
matplotlib.use('agg')

def calculate_length_diff_matrix(fasta_file):
    """Calculates a distance matrix based on sequence length differences."""
    sequences = {}
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = len(record.seq)
    except FileNotFoundError:
        print(f"Error: Fasta file not found at {fasta_file}")
        return None
    except Exception as e:
        print(f"Error parsing fasta file {fasta_file}: {e}")
        return None

    if not sequences:
        print(f"Error: No sequences found in {fasta_file}")
        return None

    ids = list(sequences.keys())
    lengths = np.array([sequences[id] for id in ids])

    # Calculate pairwise absolute length differences
    len_diff_matrix = np.abs(lengths[:, np.newaxis] - lengths[np.newaxis, :])

    len_diff_df = pd.DataFrame(len_diff_matrix, index=ids, columns=ids)
    return len_diff_df

def plot_dms_datashader(dm1, dm2, dm1_name='dm1', dm2_name='dm2', gene_name='gene', title_suffix='', ax=None, fig=None, output_dir=None, filename_base=None, max_points=None): # Added max_points parameter
    """Plots two distance matrices against each other using datashader and saves individual plot."""
    # Ensure 'accession' is a column initially
    if 'accession' not in dm1.columns:
        dm1 = dm1.reset_index().rename(columns={'index': 'accession'})
    if 'accession' not in dm2.columns:
        dm2 = dm2.reset_index().rename(columns={'index': 'accession'})

    # Set 'accession' as index for alignment (use drop=True, default)
    dm1 = dm1.set_index('accession')
    dm2 = dm2.set_index('accession')

    print('Aligning dataframes...') 
    # Find common accessions from the index
    common_accessions = dm1.index.intersection(dm2.index)

    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene_name}. Skipping plot.")
        ax.text(0.5, 0.5, 'Not enough common data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"{gene_name} {title_suffix}", fontsize=16)
        return 0 # Return 0 density

    # Reindex both dataframes (rows and columns) using the index
    # Ensure common_accessions is sorted for consistent alignment if needed, though intersection usually handles it.
    # common_accessions = sorted(list(common_accessions)) # Optional, usually not necessary
    dm1_aligned = dm1.loc[common_accessions, common_accessions]
    dm2_aligned = dm2.loc[common_accessions, common_accessions]

    # Get numerical arrays and flatten upper triangle
    # No need to drop 'accession' as it's the index now
    dm1_array = dm1_aligned.to_numpy()
    dm2_array = dm2_aligned.to_numpy()
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]

    df = pd.DataFrame({
        'x': dm1_flat,
        'y': dm2_flat
    })

    # --- Optional: Sample points for testing ---
    if max_points is not None and len(df) > max_points:
        print(f"Sampling {max_points} points out of {len(df)} for plotting...")
        df = df.sample(n=max_points, random_state=42) # Use a fixed random state for reproducibility if needed

    if df.empty:
        print(f"Warning: No data points after alignment for {gene_name}. Skipping plot.")
        ax.text(0.5, 0.5, 'No common data points', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"{gene_name} {title_suffix}", fontsize=16)
        return 0

    print('Plotting...')
    min_val_x = df['x'].min()
    max_val_x = df['x'].max()
    min_val_y = df['y'].min()
    max_val_y = df['y'].max() # Moved this line up

    # Avoid zero range
    if max_val_x == min_val_x: max_val_x += 1e-6
    if max_val_y == min_val_y: max_val_y += 1e-6

    # --- Use fixed plot dimensions for a reasonable canvas size ---
    plot_width = 500
    plot_height = 500 # Set height equal to width for a square canvas

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                      x_range=(min_val_x, max_val_x),
                      y_range=(min_val_y, max_val_y))
    print(len(df), 'points to plot...') # Added print statement
    # Create density plot
    agg = canvas.points(df, 'x', 'y')
    # Apply dynspread after shading to make points more visible at lower resolution
    img = ds.tf.shade(agg, cmap=cc.fire)
    img = ds.tf.dynspread(img, threshold=0.5, max_px=5) # Added dynspread
    img = ds.tf.set_background(img, 'white')
    print('Converting to matplotlib figure...')
    # Convert to matplotlib figure. aspect='auto' makes the image fill the axes area.
    # Explicitly set origin='lower' for consistency
    # The extent ensures the axes labels match the data ranges
    ax.imshow(img.to_pil(), extent=[min_val_x, max_val_x, min_val_y, max_val_y], aspect='auto', origin='lower')
    # Force the axes *box* itself to be square
    ax.set_aspect(1, adjustable='box')

    # Save the individual datashader image if output_dir and filename_base are provided
    # if output_dir and filename_base:
    #     os.makedirs(output_dir, exist_ok=True)
    #     img_filename = os.path.join(output_dir, f"{filename_base}.png")
    #     try:
    #         img.to_pil().save(img_filename)
    #         print(f"Saved individual plot: {img_filename}")
    #     except Exception as e:
    #         print(f"Error saving individual plot {img_filename}: {e}")

    # Increase font sizes
    ax.set_xlabel(dm1_name, fontsize=14)
    ax.set_ylabel(dm2_name, fontsize=14)
    ax.set_title(f"{gene_name} {title_suffix}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return agg.values.max() if agg is not None else 0

def process_gene(gene):
    """Process a single gene: load ESM/Phylo DMs and calculate Length Diff DM."""
    print(f"Processing gene: {gene}")

    # --- Load ESM embeddings distance matrix (dm1) ---
    embed_path = f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_embeddings.npy'
    ids_path = f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_ids.txt'
    dm1 = None
    try:
        embed = np.load(embed_path)
        if embed.ndim == 1:
            embed = embed.reshape(-1, 1)

        embed_accessions = []
        with open(ids_path, 'r') as f:
            embed_accessions = [line.strip() for line in f.readlines()[:embed.shape[0]]]

        if len(embed_accessions) != embed.shape[0]:
             raise ValueError(f"Number of IDs ({len(embed_accessions)}) does not match number of embeddings ({embed.shape[0]}) for {gene}")

        embed_df = pd.DataFrame(embed, index=embed_accessions)
        embed_dist = pd.DataFrame(
            squareform(pdist(embed_df.values, metric='cosine')),
            index=embed_accessions,
            columns=embed_accessions
        )
        dm1 = embed_dist.reset_index().rename(columns={'index': 'accession'})
    except FileNotFoundError:
        print(f"Error: ESM embedding or ID file not found for {gene}")
    except Exception as e:
        print(f"Error processing ESM data for {gene}: {e}")

    # --- Load phylogenetic distance matrix (dm2) ---
    phylo_path = f'/home/s233201/full_dist_mats/new/full_mat_{gene.upper()}.csv'
    dm2 = None
    try:
        phylo_raw = pd.read_csv(phylo_path, sep='\s+', header=None, skiprows=1)
        phylo_accessions = phylo_raw.iloc[:, 0].values
        phylo = pd.DataFrame(phylo_raw.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)
        dm2 = phylo.reset_index().rename(columns={'index': 'accession'})
    except FileNotFoundError:
        print(f"Error: Phylogenetic distance matrix file not found for {gene}")
    except Exception as e:
        print(f"Error processing Phylogenetic data for {gene}: {e}")


    # --- Calculate length difference distance matrix (dm_len) ---
    fasta_path = f'/home/s233201/esm_runs/inputs_new/{gene.upper()}.fasta'
    dm_len_df = calculate_length_diff_matrix(fasta_path)
    dm_len = None
    if dm_len_df is not None:
        dm_len = dm_len_df.reset_index().rename(columns={'index': 'accession'})
    else:
        print(f"Error: Could not calculate length difference matrix for {gene}")

    return dm1, dm2, dm_len, gene

if __name__ == '__main__':
    # --- Test Mode Configuration ---
    test_mode = False # Set to False to run on all genes with full/more points
    num_genes_to_test = 8 # Number of genes to process in test mode
    max_points_for_testing = 1000 # Max points to plot per subplot in test mode
    max_points_for_full_run = int(1e4) # Max points for the full run (or None for all)
    # -----------------------------

    all_gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]

    if test_mode:
        print(f"--- RUNNING IN TEST MODE (Processing {num_genes_to_test} gene(s), max {max_points_for_testing} points) ---")
        gene_names = all_gene_names[:num_genes_to_test]
        test_max_points = max_points_for_testing
    else:
        print("--- RUNNING IN FULL MODE ---")
        gene_names = all_gene_names
        test_max_points = max_points_for_full_run # Use the setting for full run

    num_genes = len(gene_names) # Update num_genes based on mode

    # --- Parallel Data Processing ---
    print("Starting parallel data processing...") # Added print statement
    all_results_dict = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_gene = {executor.submit(process_gene, gene): gene for gene in gene_names}
        for future in concurrent.futures.as_completed(future_to_gene):
            gene = future_to_gene[future]
            try:
                result = future.result()
                all_results_dict[gene] = result
            except Exception as exc:
                print(f'{gene} generated an exception: {exc}')
                all_results_dict[gene] = (None, None, None, gene) # Store placeholder
    print("Parallel data processing finished.") # Added print statement

    # Order results according to the (potentially reduced) gene_names list
    all_results = [all_results_dict[gene] for gene in gene_names]
    print("Data collected and ordered.") # Added print statement

    # --- Plotting Setup for Two Figures ---
    # Adjust figsize for potentially taller square subplots (4 rows, 2 cols)
    rows_grid, cols_grid = 4, 2
    # Try making figure taller to accommodate square plots + spacing
    fig_width = 12 # Reduced width slightly
    fig_height = 20 # Increased height significantly
    fig1, axes1 = plt.subplots(rows_grid, cols_grid, figsize=(fig_width, fig_height), squeeze=False)
    # Adjust top margin for suptitle and vertical/horizontal spacing (hspace/wspace)
    fig1.subplots_adjust(top=0.93, hspace=0.6, wspace=0.4) # Increased hspace
    fig1.suptitle('ESM Distance vs Length Difference', fontsize=18) # Removed y=

    fig2, axes2 = plt.subplots(rows_grid, cols_grid, figsize=(fig_width, fig_height), squeeze=False)
    fig2.subplots_adjust(top=0.93, hspace=0.6, wspace=0.4) # Increased hspace
    fig2.suptitle('Phylogenetic Distance vs Length Difference', fontsize=18) # Removed y=

    max_density1 = 0
    max_density2 = 0
    fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fire', cc.fire)

    # --- Plot all genes (potentially fewer in test mode) ---
    print("Starting plot generation...")
    output_dir = 'esm_runs/plots/len_diff_datashader' # Define output dir earlier
    os.makedirs(output_dir, exist_ok=True) # Create it once
    # test_max_points is now set based on test_mode above

    for idx, (dm1, dm2, dm_len, gene) in enumerate(all_results):
        print(f"Plotting results for gene: {gene} ({idx+1}/{num_genes})...")
        # Get axes for the current gene
        row = idx // cols_grid
        col = idx % cols_grid
        # Check if indices are within bounds (important if grid size changes)
        if row < rows_grid and col < cols_grid:
             ax1 = axes1[row, col]
             ax2 = axes2[row, col]
        else:
             print(f"Warning: Index ({row}, {col}) out of bounds for grid ({rows_grid}x{cols_grid}). Skipping plot for {gene}.")
             continue # Skip if grid is smaller than needed (shouldn't happen with fixed 4x2)


        # Plot 1: ESM vs Length Difference (on fig1)
        max_val1 = 0
        if dm1 is not None and dm_len is not None:
            # plot_dms_datashader now sets aspect ratio correctly
            max_val1 = plot_dms_datashader(dm1, dm_len,
                                        dm1_name='ESM Distance (Cosine)',
                                        dm2_name='Length Difference',
                                        gene_name=gene,
                                        title_suffix='',
                                        ax=ax1, fig=fig1,
                                        output_dir=output_dir, # Pass output dir
                                        filename_base=f"{gene}_esm_vs_len", # Use f-string
                                        max_points=test_max_points) # Pass max_points
            ax1.set_title(f"{gene}", fontsize=16)
        else:
            ax1.text(0.5, 0.5, 'Missing Data', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            ax1.set_title(f"{gene}", fontsize=16)

        # Plot 2: Phylo vs Length Difference (on fig2)
        max_val2 = 0
        if dm2 is not None and dm_len is not None:
             max_val2 = plot_dms_datashader(dm2, dm_len,
                                        dm1_name='Phylogenetic Distance',
                                        dm2_name='Length Difference',
                                        gene_name=gene,
                                        title_suffix='',
                                        ax=ax2, fig=fig2,
                                        output_dir=output_dir, # Pass output dir
                                        filename_base=f"{gene}_phylo_vs_len", # Use f-string
                                        max_points=test_max_points) # Pass max_points
             ax2.set_title(f"{gene}", fontsize=16)
        else:
            ax2.text(0.5, 0.5, 'Missing Data', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title(f"{gene}", fontsize=16)

        max_density1 = max(max_density1, max_val1)
        max_density2 = max(max_density2, max_val2)

    # Hide empty subplots if any
    total_plots_possible = rows_grid * cols_grid
    for i in range(num_genes, total_plots_possible):  # Hide unused subplots
        row = i // cols_grid
        col = i % cols_grid
        if row < rows_grid and col < cols_grid:
            axes1[row, col].set_visible(False)
            axes2[row, col].set_visible(False)

    # --- Add Colorbars and Save Figures ---
    # output_dir is already defined and created above

    # Colorbar and saving for Figure 1 (ESM vs Length)
    print("Processing and saving Figure 1 (ESM vs Length)...")
    if max_density1 > 0:
        norm1 = matplotlib.colors.Normalize(vmin=0, vmax=max_density1)
        sm1 = plt.cm.ScalarMappable(cmap=fire_cmap, norm=norm1)
        # Adjust layout considering the square aspect ratio and colorbar
        # Need to leave space on the right for the colorbar
        fig1.subplots_adjust(right=0.88, top=0.93, hspace=0.6, wspace=0.4) # Re-apply adjustments
        cbar_ax1 = fig1.add_axes([0.9, 0.15, 0.02, 0.7]) # Adjust position/width if needed
        cbar1 = fig1.colorbar(sm1, cax=cbar_ax1, label='Density')
        cbar1.ax.tick_params(labelsize=12)
        cbar1.set_label('Density', size=14)
    else:
        print("Warning: Max density is 0 for ESM vs Length plot, cannot create colorbar.")
        # Use tight_layout carefully, might conflict with subplots_adjust
        fig1.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect to leave space for suptitle

    plt.figure(fig1.number) # Ensure fig1 is the current figure for saving
    plt.savefig(os.path.join(output_dir, 'esm_vs_len_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"ESM vs Length plot saved to {output_dir}/esm_vs_len_comparison.png")

    # Colorbar and saving for Figure 2 (Phylo vs Length)
    print("Processing and saving Figure 2 (Phylo vs Length)...") # Added print statement
    if max_density2 > 0:
        norm2 = matplotlib.colors.Normalize(vmin=0, vmax=max_density2)
        sm2 = plt.cm.ScalarMappable(cmap=fire_cmap, norm=norm2)
        # Adjust layout considering the square aspect ratio and colorbar
        fig2.subplots_adjust(right=0.88, top=0.93, hspace=0.6, wspace=0.4) # Re-apply adjustments
        cbar_ax2 = fig2.add_axes([0.9, 0.15, 0.02, 0.7]) # Adjust position/width if needed
        cbar2 = fig2.colorbar(sm2, cax=cbar_ax2, label='Density')
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label('Density', size=14)
    else:
        print("Warning: Max density is 0 for Phylo vs Length plot, cannot create colorbar.")
        fig2.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect to leave space for suptitle

    plt.figure(fig2.number) # Ensure fig2 is the current figure for saving
    plt.savefig(os.path.join(output_dir, 'phylo_vs_len_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Phylo vs Length plot saved to {output_dir}/phylo_vs_len_comparison.png")

    print("Processing complete.")
