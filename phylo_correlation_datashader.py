import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import os
from scipy.stats import pearsonr
from PIL import Image
from phylo_correlation import load_phylo_matrices
import concurrent.futures

def plot_dms_datashader(dm1: pd.DataFrame, dm2: pd.DataFrame, gene_name: str, max_points: int = None, plot_width: int = 400, plot_height: int = 400) -> Image.Image | None:
    """Generates a datashader image for two distance matrices."""
    if 'accession' not in dm1.columns:
        dm1 = dm1.reset_index().rename(columns={'index': 'accession'})
    if 'accession' not in dm2.columns:
        dm2 = dm2.reset_index().rename(columns={'index': 'accession'})

    dm1 = dm1.set_index('accession')
    dm2 = dm2.set_index('accession')

    common_accessions = dm1.index.intersection(dm2.index)

    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene_name}. Skipping image generation.")
        return None

    dm1_aligned = dm1.loc[common_accessions, common_accessions]
    dm2_aligned = dm2.loc[common_accessions, common_accessions]

    # Sort indices to ensure alignment
    common_accessions_sorted = sorted(common_accessions)
    dm1_aligned = dm1_aligned.loc[common_accessions_sorted, common_accessions_sorted]
    dm2_aligned = dm2_aligned.loc[common_accessions_sorted, common_accessions_sorted]

    dm1_array = dm1_aligned.to_numpy()
    dm2_array = dm2_aligned.to_numpy()
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]

    df = pd.DataFrame({'x': dm1_flat, 'y': dm2_flat})

    # coerce to numeric, drop NaN or infinite values
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x','y'])
    df = df[np.isfinite(df['x']) & np.isfinite(df['y'])]

    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    if df.empty:
        print(f"Warning: No data points after alignment/sampling for {gene_name}. Skipping image generation.")
        return None

    min_val_x, max_val_x = df['x'].min(), df['x'].max()
    min_val_y, max_val_y = df['y'].min(), df['y'].max()

    if max_val_x == min_val_x: max_val_x += 1e-6
    if max_val_y == min_val_y: max_val_y += 1e-6
    # Use provided plot_width and plot_height
    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                       x_range=(min_val_x, max_val_x),
                       y_range=(min_val_y, max_val_y))
    agg = canvas.points(df, 'x', 'y')
    img = tf.shade(agg, cmap=cc.fire)
    img = tf.dynspread(img, threshold=0.4, max_px=10)
    img = tf.set_background(img, 'white')
    pil_image = img.to_pil()
    return pil_image

def generate_single_plot(args):
    """Helper function to generate a single datashader plot."""
    # Unpack plot dimensions from args
    i, j, gene1, gene2, mat1, mat2, max_points, plot_width, plot_height = args
    if i == j:
        return i, j, None
    try:
        img = plot_dms_datashader(mat1, mat2, gene_name=f"{gene1}_vs_{gene2}",
                                  max_points=max_points, plot_width=plot_width, plot_height=plot_height)
        return i, j, img
    except Exception as e:
        print(f"Error generating plot for {gene1} vs {gene2}: {e}")
        return i, j, None

def create_datashader_grid(phylo_matrices, output_path='/home/s233201/figures/phylo_correlation_datashader_grid.png', max_points=None):
    """Generates a grid of datashader images for all gene pairs using parallel processing."""
    gene_names = list(phylo_matrices.keys())
    num_genes = len(gene_names)
    # scale figure size per gene
    tile_size = 3
    fig, axes = plt.subplots(num_genes, num_genes,
                             figsize=(num_genes * tile_size,
                                      num_genes * tile_size))
    dpi = fig.get_dpi()
    # compute resolution per subplot (in pixels)
    subplot_px = int(tile_size * dpi)
    # Prepare tasks for parallel processing
    tasks = []
    for i in range(num_genes):
        for j in range(num_genes):
            gene1, gene2 = gene_names[i], gene_names[j]
            tasks.append((i, j, gene1, gene2, 
                          phylo_matrices[gene1].copy(), 
                          phylo_matrices[gene2].copy(),
                          max_points, subplot_px, subplot_px))
    # Process tasks in parallel
    print(f"Starting parallel processing of {len(tasks)} plots...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(generate_single_plot, tasks))

    # Plot results
    for i, j, img in results:
        ax = axes[i, j]
        # remove ticks but keep frame
        ax.set_xticks([])
        ax.set_yticks([])

        if i == j:
            ax.text(0.5, 0.5, gene_names[i],
                    ha='center', va='center', fontsize=40)
        elif img is not None:
            ax.imshow(img, aspect='auto')
        else:
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center', fontsize=40)

        # draw a thin black frame around each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')

        # label top row with gene names
        if i == 0 and j != i:
            ax.set_title(gene_names[j], fontsize=40, pad=4)
        # label left column with gene names
        if j == 0 and i != j:
            ax.set_ylabel(gene_names[i], fontsize=40, labelpad=4)

    plt.suptitle('Phylogenetic Distance Matrix Comparison (Datashader)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Datashader grid saved to {output_path}")

if __name__ == '__main__':
    # Create figures directory if it doesn't exist
    os.makedirs('/home/s233201/figures', exist_ok=True)

    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    phylo_matrices = load_phylo_matrices(gene_names)

    # Print the number of NaN values in each matrix
    for gene, matrix in phylo_matrices.items():
        nan_count = matrix.isna().sum().sum()
        print(f"Number of NaN values in {gene}: {nan_count}")

    if phylo_matrices:
        create_datashader_grid(phylo_matrices,
                               output_path='/home/s233201/figures/phylo_correlation_datashader_grid.png')
    else:
        print("Failed to load phylogenetic matrices.")
