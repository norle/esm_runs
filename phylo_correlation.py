import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import concurrent.futures

def load_phylo_matrices(gene_names, phylo_path='/home/s233201/full_dist_mats/fast/'):
    """Loads phylogenetic distance matrices for specified genes."""
    phylo_matrices = {}
    for gene in gene_names:
        print(f"Loading phylogenetic distance matrix for {gene}...")
        phylo_path_gene = os.path.join(phylo_path, f'full_mat_{gene.upper()}.csv')
        try:
            phylo_raw = pd.read_csv(phylo_path_gene, header=None, skiprows=1, engine='pyarrow')
            # Split the single column by whitespace
            phylo_split = phylo_raw[0].str.split(expand=True)
            phylo_accessions = phylo_split.iloc[:, 0].values
            phylo = pd.DataFrame(phylo_split.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)
            phylo_matrices[gene] = phylo
        except FileNotFoundError:
            print(f"Error: Phylogenetic distance matrix file not found for {gene}")
            return None
        except Exception as e:
            print(f"Error processing Phylogenetic data for {gene}: {e}")
            return None
    return phylo_matrices

def calculate_single_correlation(args):
    """Calculate correlation for a single pair of matrices."""
    i, j, mat1, mat2, gene1, gene2, method = args
    
    common_accessions = mat1.index.intersection(mat2.index)
    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene1} and {gene2}. Skipping correlation.")
        return i, j, (np.nan, np.nan)

    mat1_aligned = mat1.loc[common_accessions, common_accessions]
    mat2_aligned = mat2.loc[common_accessions, common_accessions]

    # Extract upper triangle (excluding diagonal)
    rows, cols = np.triu_indices(mat1_aligned.shape[0], k=1)
    vec1 = mat1_aligned.values[rows, cols]
    vec2 = mat2_aligned.values[rows, cols]

    # Calculate Pearson or Spearman correlation and p-value
    if method == 'spearman':
        from scipy.stats import spearmanr
        r, p = spearmanr(vec1, vec2)
    else:  # Default to pearson
        r, p = pearsonr(vec1, vec2)
    return i, j, (r, p)

def calculate_correlation_matrix(phylo_matrices, method='pearson'):
    """Calculates the correlation matrix between flattened distance matrices."""
    gene_names = list(phylo_matrices.keys())
    num_genes = len(gene_names)
    correlation_matrix = np.zeros((num_genes, num_genes))
    p_value_matrix = np.zeros((num_genes, num_genes))

    # Prepare all tasks
    tasks = []
    for i in range(num_genes):
        for j in range(i, num_genes):
            gene1, gene2 = gene_names[i], gene_names[j]
            tasks.append((i, j, phylo_matrices[gene1], phylo_matrices[gene2], gene1, gene2, method))

    # Process tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        results = executor.map(calculate_single_correlation, tasks)

        # Process results
        for i, j, (r, p) in results:
            correlation_matrix[i, j] = r
            correlation_matrix[j, i] = r
            p_value_matrix[i, j] = p
            p_value_matrix[j, i] = p

    correlation_df = pd.DataFrame(correlation_matrix, index=gene_names, columns=gene_names)
    p_value_df = pd.DataFrame(p_value_matrix, index=gene_names, columns=gene_names)
    return correlation_df, p_value_df

def plot_correlation_heatmap(correlation_matrix, output_path='/home/s233201/figures/phylo_correlation_heatmap.png', title='Pearson Correlation Heatmap'):
    """Generates a heatmap of the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def plot_spearman_heatmap(correlation_matrix, output_path='/home/s233201/figures/phylo_spearman_heatmap.png'):
    """Generates a heatmap of the Spearman correlation matrix."""
    plot_correlation_heatmap(correlation_matrix, output_path, title='Spearman Correlation Heatmap')

if __name__ == '__main__':
    # Create figures directory if it doesn't exist
    os.makedirs('/home/s233201/figures', exist_ok=True)
    
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    phylo_matrices = load_phylo_matrices(gene_names)

    if phylo_matrices:
        correlation_df, p_value_df = calculate_correlation_matrix(phylo_matrices)

        print("Correlation Matrix:")
        print(correlation_df)

        print("\nP-value Matrix:")
        print(p_value_df)

        plot_correlation_heatmap(correlation_df, output_path='/home/s233201/figures/phylo_correlation_heatmap.png')

        # Calculate and plot Spearman correlation
        spearman_corr_df, spearman_p_value_df = calculate_correlation_matrix(phylo_matrices, method='spearman')
        print("\nSpearman Correlation Matrix:")
        print(spearman_corr_df)

        print("\nSpearman P-value Matrix:")
        print(spearman_p_value_df)

        plot_spearman_heatmap(spearman_corr_df, output_path='/home/s233201/figures/phylo_spearman_heatmap.png')
    else:
        print("Failed to load phylogenetic matrices.")
