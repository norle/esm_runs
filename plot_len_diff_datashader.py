import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datashader as ds
from datashader.colors import colormap_select
import colorcet as cc
import os
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO
import concurrent.futures
from PIL import Image
from scipy.stats import pearsonr

matplotlib.use('agg')

def read_outlier_accessions(filepath):
    """Reads outlier accessions from a file and returns a set."""
    try:
        with open(filepath, 'r') as f:
            outliers = {line.strip() for line in f}
        return outliers
    except FileNotFoundError:
        print(f"Error: Outlier file not found at {filepath}")
        return set()
    except Exception as e:
        print(f"Error reading outlier file: {e}")
        return set()

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

def plot_dms_datashader(dm1: pd.DataFrame, dm2: pd.DataFrame, gene_name: str, max_points: int = None) -> tuple[Image.Image | None, float]:
    """Generates a datashader image and max density for two distance matrices."""
    if 'accession' not in dm1.columns:
        dm1 = dm1.reset_index().rename(columns={'index': 'accession'})
    if 'accession' not in dm2.columns:
        dm2 = dm2.reset_index().rename(columns={'index': 'accession'})

    dm1 = dm1.set_index('accession')
    dm2 = dm2.set_index('accession')

    common_accessions = dm1.index.intersection(dm2.index)

    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene_name}. Skipping image generation.")
        return None, 0, None, None

    dm1_aligned = dm1.loc[common_accessions, common_accessions]
    dm2_aligned = dm2.loc[common_accessions, common_accessions]

    dm1_array = dm1_aligned.to_numpy()
    dm2_array = dm2_aligned.to_numpy()
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]

    df = pd.DataFrame({'x': dm1_flat, 'y': dm2_flat})

    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    if df.empty:
        print(f"Warning: No data points after alignment/sampling for {gene_name}. Skipping image generation.")
        return None, 0, None, None

    min_val_x, max_val_x = df['x'].min(), df['x'].max()
    min_val_y, max_val_y = df['y'].min(), df['y'].max()

    if max_val_x == min_val_x: max_val_x += 1e-6
    if max_val_y == min_val_y: max_val_y += 1e-6

    plot_width = 500
    plot_height = 500

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                      x_range=(min_val_x, max_val_x),
                      y_range=(min_val_y, max_val_y))
    agg = canvas.points(df, 'x', 'y')
    img = ds.tf.shade(agg, cmap=cc.fire)
    img = ds.tf.dynspread(img, threshold=0.5, max_px=5)
    img = ds.tf.set_background(img, 'white')

    pil_image = img.to_pil()
    max_density = agg.values.max() if agg is not None else 0

    extent = [min_val_x, max_val_x, min_val_y, max_val_y]
    pil_image.info['extent'] = extent

    # Calculate Pearson correlation and p-value
    r, p = pearsonr(df['x'], df['y'])

    return pil_image, max_density, r, p

def process_gene(gene):
    """Process a single gene: load ESM/Phylo DMs and calculate Length Diff DM."""
    print(f"Processing gene: {gene}")

    # Read outlier accessions
    outlier_file = '/home/s233201/outliers_set.txt'
    outlier_accessions = read_outlier_accessions(outlier_file)

    embed_path = f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_embeddings.npy'
    ids_path = f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_ids.txt'
    dm1 = None
    try:
        embed = np.load(embed_path)
        if embed.ndim == 1:
            embed = embed.reshape(-1, 1)

        # Read all accessions first
        with open(ids_path, 'r') as f:
            embed_accessions = [line.strip() for line in f.readlines()]

        # Create mask for non-outlier accessions
        valid_mask = np.array([acc not in outlier_accessions for acc in embed_accessions])
        
        # Filter both accessions and embeddings using the same mask
        filtered_accessions = np.array(embed_accessions)[valid_mask]
        filtered_embed = embed[valid_mask]

        # Create distance matrix from filtered data
        embed_df = pd.DataFrame(filtered_embed, index=filtered_accessions)
        embed_dist = pd.DataFrame(
            squareform(pdist(embed_df.values, metric='cosine')),
            index=filtered_accessions,
            columns=filtered_accessions
        )
        dm1 = embed_dist.reset_index().rename(columns={'index': 'accession'})
    except FileNotFoundError:
        print(f"Error: ESM embedding or ID file not found for {gene}")
    except Exception as e:
        print(f"Error processing ESM data for {gene}: {e}")

    phylo_path = f'/home/s233201/full_dist_mats/clean/full_mat_{gene.upper()}.csv'
    dm2 = None
    try:
        phylo_raw = pd.read_csv(phylo_path, sep='\s+', header=None, skiprows=1)
        phylo_accessions = phylo_raw.iloc[:, 0].values

        # Filter out outlier accessions for Phylo
        phylo_accessions = [acc for acc in phylo_accessions if acc not in outlier_accessions]
        phylo_raw = phylo_raw[phylo_raw.iloc[:, 0].isin(phylo_accessions)]
        phylo = pd.DataFrame(phylo_raw.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)

        dm2 = phylo.reset_index().rename(columns={'index': 'accession'})
    except FileNotFoundError:
        print(f"Error: Phylogenetic distance matrix file not found for {gene}")
    except Exception as e:
        print(f"Error processing Phylogenetic data for {gene}: {e}")

    fasta_path = f'/home/s233201/esm_runs/inputs_new/{gene.upper()}.fasta'
    dm_len_df = calculate_length_diff_matrix(fasta_path)
    dm_len = None
    if dm_len_df is not None:
        dm_len = dm_len_df.reset_index().rename(columns={'index': 'accession'})
    else:
        print(f"Error: Could not calculate length difference matrix for {gene}")

    return dm1, dm2, dm_len, gene

def generate_plot_images(dm1, dm2, dm_len, gene, max_points):
    """Generates PIL images and max densities for ESM vs Len and Phylo vs Len."""
    print(f"Generating plot images for gene: {gene}...")
    img1, dens1, r1, p1 = (None, 0, None, None)
    if dm1 is not None and dm_len is not None:
        try:
            img1, dens1, r1, p1 = plot_dms_datashader(dm1.copy(), dm_len.copy(),
                                            gene_name=f"{gene}_esm_vs_len",
                                            max_points=max_points)
        except Exception as e:
            print(f"Error generating ESM vs Len image for {gene}: {e}")

    img2, dens2, r2, p2 = (None, 0, None, None)
    if dm2 is not None and dm_len is not None:
        try:
            img2, dens2, r2, p2 = plot_dms_datashader(dm2.copy(), dm_len.copy(),
                                            gene_name=f"{gene}_phylo_vs_len",
                                            max_points=max_points)
        except Exception as e:
            print(f"Error generating Phylo vs Len image for {gene}: {e}")

    print(f"Finished image generation for gene: {gene}.")
    return gene, img1, dens1, r1, p1, img2, dens2, r2, p2

if __name__ == '__main__':
    test_mode = False
    num_genes_to_test = 8
    max_points_for_testing = 1000
    max_points_for_full_run = None

    all_gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]

    if test_mode:
        print(f"--- RUNNING IN TEST MODE (Processing {num_genes_to_test} gene(s), max {max_points_for_testing} points) ---")
        gene_names = all_gene_names[:num_genes_to_test]
        test_max_points = max_points_for_testing
    else:
        print("--- RUNNING IN FULL MODE ---")
        gene_names = all_gene_names
        test_max_points = max_points_for_full_run

    num_genes = len(gene_names)

    print("Starting parallel data processing...")
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
                all_results_dict[gene] = (None, None, None, gene)
    print("Parallel data processing finished.")

    all_results = [all_results_dict[gene] for gene in gene_names]
    print("Data collected and ordered.")

    print("Starting parallel plot image generation...")
    plot_results_dict = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_gene = {}
        for dm1, dm2, dm_len, gene in all_results:
            future = executor.submit(generate_plot_images, dm1, dm2, dm_len, gene, test_max_points)
            future_to_gene[future] = gene

        for future in concurrent.futures.as_completed(future_to_gene):
            gene = future_to_gene[future]
            try:
                plot_results_dict[gene] = future.result()
            except Exception as exc:
                print(f'{gene} generated an exception during image generation: {exc}')
                plot_results_dict[gene] = (gene, None, 0, None, None, None, 0, None, None)
    print("Parallel plot image generation finished.")

    print("Starting final plot assembly...")
    output_dir = 'esm_runs/plots/len_diff_datashader'
    os.makedirs(output_dir, exist_ok=True)

    dm1_label = 'ESM Distance (Cosine)'
    dm2_label_len = 'Length Difference'
    dm1_label_phylo = 'Evolutionary Distance'

    rows_grid, cols_grid = 3, 3
    fig_width = 15  # Increased for 3x3 layout
    fig_height = 15  # Square layout

    # Create separate figures for ESM vs Len and Phylo vs Len
    fig1, axes1 = plt.subplots(rows_grid, cols_grid, figsize=(fig_width, fig_height), squeeze=False)
    fig1.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)  # Tighter spacing
    fig1.suptitle('ESM Distance vs Length Difference', fontsize=18, y=0.98)

    fig2, axes2 = plt.subplots(rows_grid, cols_grid, figsize=(fig_width, fig_height), squeeze=False)
    fig2.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)  # Tighter spacing
    fig2.suptitle('Phylogenetic Distance vs Length Difference', fontsize=18, y=0.98)

    max_density1 = 0
    max_density2 = 0
    fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fire', cc.fire)

    # Populate the ESM vs Len subplots
    for idx, gene in enumerate(gene_names):
        print(f"Plotting ESM vs Len for gene: {gene} ({idx+1}/{num_genes})...")
        row = idx // cols_grid
        col = idx % cols_grid

        if row >= rows_grid or col >= cols_grid:
            print(f"Warning: Index ({row}, {col}) out of bounds. Skipping plot assembly for {gene}.")
            continue

        ax1 = axes1[row, col]
        _gene, pil_img1, dens1, r1, p1, pil_img2, dens2, r2, p2 = plot_results_dict.get(gene, (gene, None, 0, None, None, None, 0, None, None))

        if pil_img1 is not None:
            extent1 = pil_img1.info.get('extent', None)
            if extent1:
                ax1.imshow(pil_img1, extent=extent1, aspect='auto', origin='upper')

                xmin, xmax, ymin, ymax = extent1

                # Fix ESM distance range to show only the actual data range
                # ESM distances are typically between 0 and 2 (cosine distance)
                if xmin < 0:
                    # Use actual data range instead of centering with expanded limits
                    ax1.set_xlim(0, xmax)
                else:
                    ax1.set_xlim(xmin, xmax)

                ax1.set_ylim(ymin, ymax)

                ax1.set_xlabel(dm1_label, fontsize=14)
                ax1.set_ylabel(dm2_label_len, fontsize=14)
                ax1.tick_params(axis='both', which='major', labelsize=12)
                max_density1 = max(max_density1, dens1)

                # Add Pearson correlation and p-value to the top right
                r_str = f"{r1:.2f}" if r1 is not None else "N/A"
                p_str = f"{p1:.2e}" if p1 is not None and p1 > 0.0 else "<1e-16" if p1 is not None else "N/A"
                ax1.text(0.95, 0.95, f"r={r_str}\np={p_str}", transform=ax1.transAxes,
                        fontsize=14, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.8, pad=0.5, edgecolor='none'))
            else:
                print(f"Warning: Extent missing for ESM vs Len image for {gene}")
                ax1.text(0.5, 0.5, 'Plotting Error', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'Missing Data / Error', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title(f"{gene.upper()}", fontsize=16)

    # Populate the Phylo vs Len subplots
    for idx, gene in enumerate(gene_names):
        print(f"Plotting Phylo vs Len for gene: {gene} ({idx+1}/{num_genes})...")
        row = idx // cols_grid
        col = idx % cols_grid

        if row >= rows_grid or col >= cols_grid:
            print(f"Warning: Index ({row}, {col}) out of bounds. Skipping plot assembly for {gene}.")
            continue

        ax2 = axes2[row, col]
        _gene, pil_img1, dens1, r1, p1, pil_img2, dens2, r2, p2 = plot_results_dict.get(gene, (gene, None, 0, None, None, None, 0, None, None))

        if pil_img2 is not None:
            extent2 = pil_img2.info.get('extent', None)
            if extent2:
                ax2.imshow(pil_img2, extent=extent2, aspect='auto', origin='upper')
                
                # Set evolutionary distance range to 0-2.5
                xmin, xmax, ymin, ymax = extent2
                ax2.set_xlim(0, 2.5)
                ax2.set_ylim(ymin, ymax)
                
                ax2.set_xlabel(dm1_label_phylo, fontsize=14)
                ax2.set_ylabel(dm2_label_len, fontsize=14)
                ax2.tick_params(axis='both', which='major', labelsize=12)
                max_density2 = max(max_density2, dens2)

                # Add Pearson correlation and p-value to the top right
                r_str = f"{r2:.2f}" if r2 is not None else "N/A"
                p_str = f"{p2:.2e}" if p2 is not None and  p2 > 0.0 else "<1e-16" if p2 is not None else "N/A"
                ax2.text(0.95, 0.95, f"r={r_str}\np={p_str}", transform=ax2.transAxes,
                        fontsize=14, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.8, pad=0.5, edgecolor='none'))
            else:
                print(f"Warning: Extent missing for Phylo vs Len image for {gene}")
                ax2.text(0.5, 0.5, 'Plotting Error', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'Missing Data / Error', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title(f"{gene.upper()}", fontsize=16)

    # Remove any unused subplots
    for i in range(num_genes, rows_grid * cols_grid):
        row = i // cols_grid
        col = i % cols_grid
        if row < rows_grid and col < cols_grid:
            axes1[row, col].set_visible(False)
            axes2[row, col].set_visible(False)

    # Add colorbars to Figure 1 (ESM vs Len)
    print("Processing and saving Figure 1 (ESM vs Length)...")
    if max_density1 > 0:
        norm1 = matplotlib.colors.Normalize(vmin=0, vmax=max_density1)
        sm1 = plt.cm.ScalarMappable(cmap=fire_cmap, norm=norm1)
        fig1.subplots_adjust(right=0.92, top=0.95, hspace=0.3, wspace=0.2)
        cbar_ax1 = fig1.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar1 = fig1.colorbar(sm1, cax=cbar_ax1, label='Density')
        cbar1.ax.tick_params(labelsize=12)
        cbar1.set_label('Density', size=14)
    else:
        print("Warning: Max density is 0 for ESM vs Length plot, cannot create colorbar.")
        fig1.tight_layout(rect=[0, 0, 1, 0.93])

    plt.figure(fig1.number)
    plt.savefig(os.path.join(output_dir, 'esm_vs_len_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"ESM vs Length plot saved to {output_dir}/esm_vs_len_comparison.png")

    # Add colorbars to Figure 2 (Phylo vs Len)
    print("Processing and saving Figure 2 (Phylo vs Length)...")
    if max_density2 > 0:
        norm2 = matplotlib.colors.Normalize(vmin=0, vmax=max_density2)
        sm2 = plt.cm.ScalarMappable(cmap=fire_cmap, norm=norm2)
        fig2.subplots_adjust(right=0.92, top=0.95, hspace=0.3, wspace=0.2)
        cbar_ax2 = fig2.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar2 = fig2.colorbar(sm2, cax=cbar_ax2, label='Density')
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label('Density', size=14)
    else:
        print("Warning: Max density is 0 for Phylo vs Length plot, cannot create colorbar.")
        fig2.tight_layout(rect=[0, 0, 1, 0.93])

    plt.figure(fig2.number)
    plt.savefig(os.path.join(output_dir, 'phylo_vs_len_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Phylo vs Length plot saved to {output_dir}/phylo_vs_len_comparison.png")

    print("Processing complete.")
