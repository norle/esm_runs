import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from phylo_correlation import load_phylo_matrices
import os
import math
from numba import jit, prange, set_num_threads

set_num_threads(32)

@jit(nopython=True, parallel=True)
def _calculate_mean_distances_numba(matrix_values: np.ndarray) -> np.ndarray:
    """Optimized mean distance calculation using Numba with parallelization."""
    n = matrix_values.shape[0]
    means = np.zeros(n)
    for i in prange(n):
        total = 0.0
        count = 0
        for j in range(n):
            if i != j:
                total += matrix_values[i, j]
                count += 1
        means[i] = total / count
    return means

def calculate_mean_distances(matrix: pd.DataFrame) -> pd.Series:
    """Calculate mean distance for each organism, excluding self-comparisons."""
    # Ensure matrix is indexed properly
    if 'accession' not in matrix.columns:
        matrix = matrix.reset_index().rename(columns={'index': 'accession'})
    matrix = matrix.set_index('accession')
    
    # Use optimized numba function for calculation
    matrix_values = matrix.values.astype(np.float64)
    mean_distances = _calculate_mean_distances_numba(matrix_values)
    
    return pd.Series(mean_distances, index=matrix.index)

def plot_distance_histograms(phylo_matrices: dict, output_path: str):
    """Create a grid of histograms showing distance distributions for each gene."""
    print(f"Processing {len(phylo_matrices)} phylogenetic matrices...")
    n_genes = len(phylo_matrices)
    n_cols = min(3, n_genes)  # max 3 columns
    n_rows = math.ceil(n_genes / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_genes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    with open('/home/s233201/outliers.txt', 'w') as outfile:
        for idx, (gene_name, matrix) in enumerate(phylo_matrices.items()):
            print(f"Calculating distances for {gene_name} ({idx + 1}/{n_genes})")
            ax = axes[idx]
            mean_distances = calculate_mean_distances(matrix)
            
            cutoff = np.mean(mean_distances.values) + 4 * np.std(mean_distances.values)
            print(f"Cutoff for {gene_name}: {cutoff:.2f}")
            
            ax.hist(mean_distances, bins=30, edgecolor='black')
            ax.set_title(f'{gene_name.upper()}')
            ax.set_xlabel('Mean Distance')
            ax.set_ylabel('Frequency')
            ax.axvline(cutoff, color='red', linestyle='dashed', linewidth=1, label=f'Cutoff: {cutoff:.2f}')
            ax.legend()
            
            outliers = mean_distances[mean_distances > cutoff].index.tolist()
            outfile.write(f"Outliers for {gene_name}: {outliers}\n")
            print(f"Outliers for {gene_name}: {outliers}")
            
    # Hide empty subplots if any
    for idx in range(len(phylo_matrices), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution of Mean Phylogenetic Distances by Gene', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    print("Starting phylogenetic distance analysis...")
    os.makedirs('/home/s233201/figures', exist_ok=True)
    
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    print(f"Loading phylogenetic matrices for {len(gene_names)} genes...")
    phylo_matrices = load_phylo_matrices(gene_names,phylo_path='/home/s233201/full_dist_mats/fast/')
    
    if phylo_matrices:
        print("Creating distance histograms...")
        plot_distance_histograms(
            phylo_matrices,
            output_path='/home/s233201/figures/phylo_distance_histograms.png'
        )
        print("✓ Histograms saved to /home/s233201/figures/phylo_distance_histograms.png")
    else:
        print("❌ Failed to load phylogenetic matrices.")
