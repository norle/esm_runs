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
from tqdm import tqdm
import gc
from joblib import Parallel, delayed

def plot_dms_datashader(dm_y: pd.DataFrame, dm_x: pd.DataFrame, gene_y_name: str, gene_x_name: str, max_points: int = None, plot_width: int = 200, plot_height: int = 200) -> Image.Image | None:
    """Generates a datashader image for two distance matrices."""
    # Convert matrix values to numeric type
    dm_y = dm_y.apply(pd.to_numeric, errors='coerce')
    dm_x = dm_x.apply(pd.to_numeric, errors='coerce')

    if 'accession' not in dm_y.columns:
        dm_y = dm_y.reset_index().rename(columns={'index': 'accession'})
    if 'accession' not in dm_x.columns:
        dm_x = dm_x.reset_index().rename(columns={'index': 'accession'})

    dm_y = dm_y.set_index('accession')
    dm_x = dm_x.set_index('accession')

    common_accessions = dm_y.index.intersection(dm_x.index)

    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene_y_name} vs {gene_x_name}. Skipping image generation.")
        return None

    dm_y_aligned = dm_y.loc[common_accessions, common_accessions]
    dm_x_aligned = dm_x.loc[common_accessions, common_accessions]

    # Sort indices to ensure alignment
    common_accessions_sorted = sorted(common_accessions)
    dm_y_aligned = dm_y_aligned.loc[common_accessions_sorted, common_accessions_sorted]
    dm_x_aligned = dm_x_aligned.loc[common_accessions_sorted, common_accessions_sorted]

    # Print 0.9999 quantiles for both matrices
    q_y = np.quantile(dm_y_aligned.values[np.triu_indices(len(dm_y_aligned), k=1)], 0.9999)
    q_x = np.quantile(dm_x_aligned.values[np.triu_indices(len(dm_x_aligned), k=1)], 0.9999)
    print(f"0.9999 quantiles - {gene_y_name} (Y-axis): {q_y:.4f}, {gene_x_name} (X-axis): {q_x:.4f}")

    dm_y_array = dm_y_aligned.to_numpy()
    dm_x_array = dm_x_aligned.to_numpy()
    rows, cols = np.triu_indices(dm_y_array.shape[0], k=1)
    dm_y_flat = dm_y_array[rows, cols]
    dm_x_flat = dm_x_array[rows, cols]

    df = pd.DataFrame({'x': dm_x_flat, 'y': dm_y_flat})

    # coerce to numeric, drop NaN or infinite values
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x','y'])
    df = df[np.isfinite(df['x']) & np.isfinite(df['y'])]

    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    if df.empty:
        print(f"Warning: No data points after alignment/sampling for {gene_y_name} vs {gene_x_name}. Skipping image generation.")
        return None

    # Use fixed range of 0 to 2 for both axes
    min_val_x, max_val_x = 0, 2
    min_val_y, max_val_y = 0, 2

    # Use provided plot_width and plot_height
    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                       x_range=(min_val_x, max_val_x),
                       y_range=(min_val_y, max_val_y))
    agg = canvas.points(df, 'x', 'y')
    img = tf.shade(agg, cmap=cc.fire)
    #img = tf.dynspread(img, threshold=0.4, max_px=10)
    img = tf.set_background(img, 'white')
    pil_image = img.to_pil()
    return pil_image

def worker_plot(i, j, gene1, gene2, plot_width, plot_height, max_points, matrix_paths):
    """Worker function that loads only the required matrices for a pair."""
    if i == j:
        return (i, j, None)
    try:
        mat1 = pd.read_pickle(matrix_paths[gene1])
        mat2 = pd.read_pickle(matrix_paths[gene2])
        # Note: gene1 is for Y-axis (row), gene2 is for X-axis (column)
        img = plot_dms_datashader(mat1, mat2, gene1, gene2, max_points, plot_width, plot_height)
        del mat1, mat2
        gc.collect()
        return (i, j, img)
    except Exception as e:
        print(f"Error plotting {gene1} vs {gene2}: {e}")
        return (i, j, None)

def create_datashader_grid(phylo_matrices, output_path='/home/s233201/figures/phylo_correlation_datashader_grid.png', max_points=None):
    """Generates a grid of datashader images for all gene pairs using parallel processing."""
    gene_names = list(phylo_matrices.keys())
    num_genes = len(gene_names)
    print(f"\nProcessing {num_genes} genes ({num_genes}x{num_genes} = {num_genes**2} total plots)")

    # Save matrices to disk and keep paths
    matrix_paths = {}
    for gene in gene_names:
        path = f"/tmp/{gene}_matrix.pkl"
        phylo_matrices[gene].to_pickle(path)
        matrix_paths[gene] = path

    tile_size = 3
    fig, axes = plt.subplots(num_genes, num_genes,
                            figsize=(num_genes * tile_size,
                                    num_genes * tile_size))
    dpi = fig.get_dpi()
    subplot_px = int(min(200, tile_size * dpi))

    tasks = []
    for i in range(num_genes):
        for j in range(num_genes):
            gene1, gene2 = gene_names[i], gene_names[j]
            tasks.append((i, j, gene1, gene2, subplot_px, subplot_px))

    num_workers = min(8, os.cpu_count())  # Reduce if still OOM
    print(f"Starting parallel processing with {num_workers} workers...")

    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(worker_plot)(i, j, gene1, gene2, plot_width, plot_height, max_points, matrix_paths)
        for (i, j, gene1, gene2, plot_width, plot_height) in tqdm(tasks, desc="Scheduling tasks", unit="cell")
    )

    print("\nRendering plot grid...")
    for i, j, img in tqdm(results, desc="Rendering grid", unit="cell"):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        if i == j:
            ax.text(0.5, 0.5, gene_names[i].upper(),
                    ha='center', va='center', fontsize=40)
        elif img is not None:
            ax.imshow(img, aspect='auto')
            img.close()
        else:
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center', fontsize=40)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')
        if i == 0 and j != i:
            ax.set_title(gene_names[j].upper(), fontsize=40, pad=4)
        if j == 0 and i != j:
            ax.set_ylabel(gene_names[i].upper(), fontsize=40, labelpad=4)

    print("\nSaving final figure...")
    plt.suptitle('Phylogenetic Distance Matrix Comparison (Datashader)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close('all')
    gc.collect()
    print(f"✓ Successfully saved datashader grid to {output_path}")

    # Clean up temp files
    for path in matrix_paths.values():
        try:
            os.remove(path)
        except Exception:
            pass

if __name__ == '__main__':
    print("Starting phylogenetic correlation analysis...")
    # Create figures directory if it doesn't exist
    os.makedirs('/home/s233201/figures', exist_ok=True)

    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    phylo_matrices = load_phylo_matrices(gene_names)

    # Print the number of NaN values in each matrix
    for gene, matrix in tqdm(phylo_matrices.items(), desc="Checking matrices", unit="gene"):
        nan_count = matrix.isna().sum().sum()
        print(f"Number of NaN values in {gene}: {nan_count}")
    
    if phylo_matrices:
        create_datashader_grid(phylo_matrices,
                               output_path='/home/s233201/figures/phylo_correlation_datashader_grid.png')
    else:
        print("Failed to load phylogenetic matrices.")
