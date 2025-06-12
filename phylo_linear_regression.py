import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import concurrent.futures
import gc

def clean_distance_matrix(matrix, percentile=99.99):
    """Clean distance matrix by capping values at specified percentile."""
    # Get upper triangle values (excluding diagonal)
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    upper_triangle = matrix.values[rows, cols]
    
    # Calculate percentile threshold
    threshold = np.nanpercentile(upper_triangle, percentile)
    
    # Create a copy and cap values
    cleaned = matrix.copy()
    cleaned.values[cleaned.values > threshold] = threshold
    
    return cleaned

def load_single_matrix(gene, phylo_path='/home/s233201/full_dist_mats/clean/'):
    """Load a single phylogenetic distance matrix."""
    phylo_path_gene = os.path.join(phylo_path, f'full_mat_{gene.upper()}.csv')
    try:
        phylo_raw = pd.read_csv(phylo_path_gene, header=None, skiprows=1, engine='pyarrow')
        phylo_split = phylo_raw[0].str.split(expand=True)
        phylo_accessions = phylo_split.iloc[:, 0].values
        # Convert string values to float, excluding the first column (accessions)
        phylo_values = phylo_split.iloc[:, 1:].astype(float)
        phylo = pd.DataFrame(phylo_values.values, index=phylo_accessions, columns=phylo_accessions)
        return clean_distance_matrix(phylo)
    except Exception as e:
        print(f"Error loading matrix for {gene}: {e}")
        return None

def calculate_single_regression(args):
    """Calculate linear regression with intercept for a single pair of matrices."""
    i, j, gene1, gene2 = args
    
    # Load matrices within the process
    mat1 = load_single_matrix(gene1)
    mat2 = load_single_matrix(gene2)
    
    if mat1 is None or mat2 is None:
        return i, j, (np.nan, np.nan, np.nan)

    common_accessions = mat1.index.intersection(mat2.index)
    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene1} and {gene2}. Skipping regression.")
        return i, j, (np.nan, np.nan, np.nan)

    mat1_aligned = mat1.loc[common_accessions, common_accessions]
    mat2_aligned = mat2.loc[common_accessions, common_accessions]

    mat1_aligned = mat1_aligned.astype(np.float64)
    mat2_aligned = mat2_aligned.astype(np.float64)

    rows, cols = np.triu_indices(mat1_aligned.shape[0], k=1)
    X = mat1_aligned.values[rows, cols].reshape(-1, 1)
    y = mat2_aligned.values[rows, cols]
    
    mask = np.isfinite(X.flatten()) & np.isfinite(y)
    X = X[mask].reshape(-1, 1)
    y = y[mask]
    
    if len(X) < 2:
        return i, j, (np.nan, np.nan, np.nan)
    
    # Fit regression model with intercept
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    
    # Calculate R-squared
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    
    return i, j, (reg.coef_[0], reg.intercept_, r2)

def calculate_regression_matrices(gene_names):
    """Calculates slope, intercept, and R-squared matrices from linear regression with intercept."""
    num_genes = len(gene_names)
    slope_matrix = np.zeros((num_genes, num_genes))
    intercept_matrix = np.zeros((num_genes, num_genes))
    r2_matrix = np.zeros((num_genes, num_genes))

    tasks = []
    for i in range(num_genes):
        for j in range(num_genes):
            if i != j:
                tasks.append((i, j, gene_names[i], gene_names[j]))
            else:
                slope_matrix[i, j] = 1.0  # Perfect slope for self-comparison
                intercept_matrix[i, j] = 0.0  # Intercept is 0 for self-comparison
                r2_matrix[i, j] = 1.0     # Perfect R-squared for self-comparison

    # Split tasks into 8 chunks
    num_workers = 8
    chunk_size = (len(tasks) + num_workers - 1) // num_workers
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    print(f"Processing {len(tasks)} regression tasks in {len(task_chunks)} chunks using {num_workers} workers")
    
    # Process chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk in task_chunks:
            chunk_results = list(executor.map(calculate_single_regression, chunk))
            
            # Process chunk results
            for i, j, (slope, intercept, r2) in chunk_results:
                slope_matrix[i, j] = slope
                intercept_matrix[i, j] = intercept
                r2_matrix[i, j] = r2

            # Force garbage collection after each chunk
            gc.collect()

    return (pd.DataFrame(slope_matrix, index=gene_names, columns=gene_names),
            pd.DataFrame(intercept_matrix, index=gene_names, columns=gene_names),
            pd.DataFrame(r2_matrix, index=gene_names, columns=gene_names))

def plot_heatmap(slope_matrix, intercept_matrix, r2_matrix, output_path='/home/s233201/figures/phylo_regression_heatmap.png'):
    """Creates a heatmap with regression equations and R-squared values."""
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    sns.heatmap(slope_matrix, 
                annot=True, 
                cmap='rocket_r',
                linewidths=0,
                center=None,
                vmin=0,
                vmax=3.5,
                fmt='.3f',
                annot_kws={'size': 10.5, 'weight': 'bold'},
                cbar_kws={'label': 'Slope Value'})

    # Get the colorbar and modify its label properties
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Slope Value', size=7, weight='bold')

    # Customize annotations to include regression equation and R-squared values
    for i in range(len(slope_matrix.index)):
        for j in range(len(slope_matrix.columns)):
            text = plt.gca().texts[i * len(slope_matrix.columns) + j]
            slope = slope_matrix.iloc[i, j]
            intercept = intercept_matrix.iloc[i, j]
            r2 = r2_matrix.iloc[i, j]
            # Format: y = ax + b\nR²=...
            text.set_text(f"$y={slope:.2f}x{intercept:+.2f}$\nR²={r2:.2f}")

    plt.title('Linear Regression Analysis', fontsize=12, weight='bold')
    plt.xlabel('Predictor Gene', fontsize=10, weight='bold')
    plt.ylabel('Response Gene', fontsize=10, weight='bold')
    plt.xticks(rotation=45, fontsize=8, weight='bold')
    plt.yticks(fontsize=8, weight='bold')
    
    # Set tick labels to uppercase
    ax = plt.gca()
    ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
    ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regression heatmap saved to {output_path}")

if __name__ == '__main__':
    # Create figures directory if it doesn't exist
    os.makedirs('/home/s233201/figures', exist_ok=True)
    
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    
    print("\nCalculating linear regression with intercept for all gene pairs...")
    slope_df, intercept_df, r2_df = calculate_regression_matrices(gene_names)

    print("\nSlope Matrix:")
    print(slope_df)

    print("\nIntercept Matrix:")
    print(intercept_df)

    print("\nR-squared Matrix:")
    print(r2_df)

    # Save matrices to CSV
    slope_df.to_csv('/home/s233201/figures/phylo_slopes.csv')
    intercept_df.to_csv('/home/s233201/figures/phylo_intercepts.csv')
    r2_df.to_csv('/home/s233201/figures/phylo_r_squared.csv')

    # Create visualization
    plot_heatmap(slope_df, intercept_df, r2_df)
    
    print("\nRegression analysis complete!")
