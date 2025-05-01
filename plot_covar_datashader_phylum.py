import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datashader as ds
from datashader.colors import colormap_select
import colorcet as cc
import os
from functools import partial
from scipy.spatial.distance import pdist, squareform

matplotlib.use('agg')

def get_phylum_colors():
    """Return TOL 27 color palette for phylum pairs"""
    return [
        '#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499',
        '#882255', '#6699CC', '#661100', '#DD6677', '#AA4466', '#4477AA', '#228833',
        '#CCBB44', '#EE8866', '#BBCC33', '#AAAA00', '#EEDD88', '#FFAABB', '#77AADD',
        '#99DDFF', '#44BB99', '#DDDDDD', '#000000', '#F0E442', '#BBBBBB'
    ]


def plot_dms_datashader(dm1, dm2, dm1_name='dm1', dm2_name='dm2', gene_name='gene', ax=None, fig=None, taxa_df=None):
    """Create a datashader plot comparing two distance matrices with phylum-based coloring."""
    print(f"Starting plot for {gene_name}...")

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
    print(f"  - Found {len(common_accessions)} common accessions for {gene_name}")

    # Reindex both dataframes (rows and columns) to align them
    dm1_aligned = dm1.loc[common_accessions, common_accessions]
    dm2_aligned = dm2.loc[common_accessions, common_accessions]
    print(f"  - Aligned distance matrices for {gene_name}")

    # Get numerical arrays and flatten upper triangle
    dm1_array = dm1_aligned.to_numpy()
    dm2_array = dm2_aligned.to_numpy()
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]
    print(f"  - Extracted {len(dm1_flat)} pairwise distances for {gene_name}")

    # Create a DataFrame with the coordinates and organism IDs for coloring
    aligned_accessions = dm1_aligned.index.values # Use index from aligned matrix
    org_ids_rows = aligned_accessions[rows]
    org_ids_cols = aligned_accessions[cols]

    # Create DataFrame more efficiently
    print(f"  - Creating dataframe for {gene_name}...")
    df = pd.DataFrame({
        'x': dm1_flat,
        'y': dm2_flat,
        'org_id_row': org_ids_rows,
        'org_id_col': org_ids_cols
    })


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
        # Ensure taxa_df uses the correct column name ('Accession') and no truncation
        # Ensure keys are strings for reliable matching
        # --- MODIFICATION START ---
        # Strip version suffix (e.g., .1) from accession IDs in taxa_df before creating the dictionary
        taxa_accessions_stripped = taxa_df['Accession'].astype(str).str.split('.').str[0]
        acc_to_phylum = dict(zip(taxa_accessions_stripped, taxa_df['Phylum']))
        # --- MODIFICATION END ---
        print(f"  - Debug: Created acc_to_phylum dictionary with {len(acc_to_phylum)} entries.")
        if acc_to_phylum:
            print(f"  - Debug: First 5 keys in acc_to_phylum: {list(acc_to_phylum.keys())[:5]}")
        else:
            print("  - Debug: acc_to_phylum dictionary is empty!")

        # OPTIMIZATION: Use vectorized operations when possible
        print(f"  - Mapping phyla to {len(df)} data points...")
        # Ensure IDs being mapped are also strings
        org_ids_row_str = df['org_id_row'].astype(str)
        org_ids_col_str = df['org_id_col'].astype(str)

        # --- MODIFICATION START ---
        # Strip version suffix from org_ids before mapping
        org_ids_row_str_stripped = org_ids_row_str.str.split('.').str[0]
        org_ids_col_str_stripped = org_ids_col_str.str.split('.').str[0]

        print(f"  - Debug: First 5 stripped org_id_row values to map: {org_ids_row_str_stripped.unique()[:5]}")
        print(f"  - Debug: First 5 stripped org_id_col values to map: {org_ids_col_str_stripped.unique()[:5]}")

        # Perform the mapping using the stripped IDs
        mapped_rows = org_ids_row_str_stripped.map(acc_to_phylum)
        mapped_cols = org_ids_col_str_stripped.map(acc_to_phylum)
        # --- MODIFICATION END ---

        # Check how many mappings were successful before filling NaNs
        successful_row_maps = mapped_rows.notna().sum()
        successful_col_maps = mapped_cols.notna().sum()
        print(f"  - Debug: Number of successful row mappings (before fillna): {successful_row_maps} / {len(df)}")
        print(f"  - Debug: Number of successful col mappings (before fillna): {successful_col_maps} / {len(df)}")

        if successful_row_maps == 0 or successful_col_maps == 0:
            print("  - WARNING: Few or no successful phylum mappings. Check accession ID matching between distance matrices and taxa file.")

        # Use .map which is generally faster than pd.Series(df[...].map(...))
        df['phylum_row'] = mapped_rows.fillna('Unknown')
        df['phylum_col'] = mapped_cols.fillna('Unknown')

        # Print counts of mapped phyla vs Unknown
        print(f"  - Debug: Value counts for phylum_row after fillna:\n{df['phylum_row'].value_counts().head()}")
        print(f"  - Debug: Value counts for phylum_col after fillna:\n{df['phylum_col'].value_counts().head()}")

        # Truncate phylum names to the first 13 characters
        df['phylum_row_trunc'] = df['phylum_row'].astype(str).str[:13]
        df['phylum_col_trunc'] = df['phylum_col'].astype(str).str[:13]

        # OPTIMIZATION: Vectorized phylum pair creation using truncated names
        print(f"  - Creating phylum pairs using truncated names...")
        # Ensure consistent ordering using np.minimum/maximum
        phylum1 = df['phylum_row_trunc'] # Use truncated names
        phylum2 = df['phylum_col_trunc'] # Use truncated names
        df['phylum_pair'] = np.minimum(phylum1, phylum2) + '_' + np.maximum(phylum1, phylum2)

        # Convert to categorical and count unique pairs
        df['phylum_pair'] = df['phylum_pair'].astype('category')
        unique_pair_count = len(df['phylum_pair'].cat.categories)
        print(f"  - Found {unique_pair_count} unique truncated phylum pairs for {gene_name}")

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

    # Load ESM embeddings from the 'new' directory
    print(f"  - Loading embeddings for {gene}...")
    embed_path = f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_embeddings.npy'
    ids_path = f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_ids.txt'
    phylo_path = f'/home/s233201/full_dist_mats/new/full_mat_{gene.upper()}.csv'

    try:
        embed = np.load(embed_path)
        if embed.ndim == 1:
            embed = embed.reshape(-1, 1)

        # Load accessions used for embeddings
        embed_accessions = []
        with open(ids_path, 'r') as f:
            # Ensure we only take as many accessions as embeddings
            embed_accessions = [line.strip() for line in f.readlines()[:embed.shape[0]]]
        print(f"  - Loaded {embed.shape[0]} embeddings with {len(embed_accessions)} accessions for {gene}")

        # Load phylogenetic distance matrix from the 'new' directory
        print(f"  - Loading phylogenetic distances for {gene}...")
        phylo_raw = pd.read_csv(phylo_path, sep='\s+', header=None, skiprows=1)

        # Get accessions from phylo matrix and set as index/columns
        phylo_accessions = phylo_raw.iloc[:, 0].values
        phylo = pd.DataFrame(phylo_raw.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)
        print(f"  - Found {len(phylo_accessions)} sequences in phylo matrix for {gene}")

        # Create DataFrame with accessions and embeddings
        embed_df = pd.DataFrame(embed, index=embed_accessions)

        # Convert to distance matrix using cosine distance
        print(f"  - Computing ESM distance matrix for {gene}...")
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

        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Completed processing for {gene} in {duration:.2f} seconds")

        return dm1, dm2, dm1_name, dm2_name, gene

    except FileNotFoundError as e:
        print(f"  - ERROR: Could not find file for gene {gene}: {e}")
        return None
    except Exception as e:
        print(f"  - ERROR: An unexpected error occurred while processing gene {gene}: {e}")
        return None


if __name__ == '__main__':
    print("Starting phylum-colored distance matrix comparison script")
    start_time = pd.Timestamp.now()

    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    print(f"Analyzing {len(gene_names)} genes: {', '.join(gene_names)}")

    print("Loading taxonomy data...")
    taxa_df = pd.read_csv('/home/s233201/esm_runs/inputs/taxa.csv')
    # Ensure the correct column name is used if it's not 'Accession'
    if 'Accession' not in taxa_df.columns:
         # Attempt common alternatives or raise an error
         if 'accession' in taxa_df.columns:
             taxa_df = taxa_df.rename(columns={'accession': 'Accession'})
         # Add more checks if needed
         else:
             raise ValueError("Taxonomy file must contain an 'Accession' column.")
    print(f"Found {len(taxa_df)} taxonomy records")

    # Process all genes first
    print("\nPre-processing all genes...")
    all_results = []
    for i, gene in enumerate(gene_names):
        print(f"\nProcessing gene {i+1}/{len(gene_names)}: {gene}")
        result = process_gene(gene, taxa_df)
        if result: # Only append if processing was successful
            all_results.append(result)
        else:
            print(f"Skipping gene {gene} due to processing errors.")

    if not all_results:
        print("\nNo genes processed successfully. Exiting.")
        exit()

    # Plot all genes with phylum-based coloring
    print("\nGenerating plots for all genes...")
    # Adjust grid spec if the number of successful results is different
    num_plots = len(all_results)
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols # Calculate required rows
    fig = plt.figure(figsize=(20, 6 * nrows)) # Adjust height based on rows
    gs = plt.GridSpec(nrows, ncols, height_ratios=[1] * nrows)

    for idx, (dm1, dm2, dm1_name, dm2_name, gene) in enumerate(all_results):
        row = idx // ncols
        col = idx % ncols
        print(f"\nPlotting gene {idx+1}/{len(all_results)}: {gene} (position: row {row+1}, col {col+1})")
        ax = fig.add_subplot(gs[row, col])
        # Pass the original taxa_df, mapping happens inside plot_dms_datashader
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