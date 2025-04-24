import numpy as np
from scipy.spatial.distance import pdist, squareform
from numba import njit, prange
import numba
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

numba.set_num_threads(64)


@njit(parallel=True)
def mantel_permutation_test_nb(X_residuals, Y_residuals_matrix, perms):
    m = Y_residuals_matrix.shape[0]
    n = X_residuals.shape[0]
    correlations = np.zeros(perms + 1)
    
    # Calculate initial correlation
    order = np.arange(m)
    Y_residuals_permuted = np.zeros(n)
    
    # Store permutation results
    for i in prange(1, perms + 1):
        np.random.shuffle(order)
        Y_matrix_permuted = Y_residuals_matrix[order][:, order]
        # Convert to condensed form
        k = 0
        for a in range(m-1):
            for b in range(a+1, m):
                Y_residuals_permuted[k] = Y_matrix_permuted[a, b]
                k += 1
        
        # Calculate correlation
        covariance = (X_residuals * Y_residuals_permuted).sum()
        denominator = np.sqrt(np.sum(X_residuals**2) * np.sum(Y_residuals_permuted**2))
        correlations[i] = covariance / denominator if denominator != 0 else 0.0
    
    return correlations

@njit(parallel=True)
def cosine_pdist(X):
    n = X.shape[0]
    m = (n * (n-1)) // 2
    out = np.zeros(m)
    k = 0
    for i in prange(n-1):
        for j in range(i+1, n):
            dot = 0.0
            norm_i = 0.0
            norm_j = 0.0
            for d in range(X.shape[1]):
                dot += X[i, d] * X[j, d]
                norm_i += X[i, d] * X[i, d]
                norm_j += X[j, d] * X[j, d]
            if norm_i == 0.0 or norm_j == 0.0:
                out[k] = np.pi / 2  # 90 degrees for orthogonal vectors
            else:
                # Calculate true cosine distance using arccos
                cos_sim = dot / (np.sqrt(norm_i) * np.sqrt(norm_j))
                # Clip to handle numerical instabilities
                cos_sim = min(1.0, max(-1.0, cos_sim))
                out[k] = np.arccos(cos_sim)
            k += 1
    return out

def mantel_test(dist1, dist2, perms=1000, tail="two-tail"):
    # Convert to residuals
    triu_idx = np.triu_indices_from(dist1, k=1)
    X = dist1[triu_idx]
    Y = dist2[triu_idx]
    
    # Calculate residuals
    X_residuals = X - np.mean(X)
    Y_residuals = Y - np.mean(Y)
    
    # Convert Y residuals to matrix form for permutation
    Y_residuals_matrix = squareform(Y_residuals)
    
    # Compute correlations including the original
    correlations = mantel_permutation_test_nb(X_residuals, Y_residuals_matrix, perms)
    
    # Calculate actual correlation for the first position
    covariance = (X_residuals * Y_residuals).sum()
    denominator = np.sqrt(np.sum(X_residuals**2) * np.sum(Y_residuals**2))
    correlations[0] = covariance / denominator if denominator != 0 else 0.0
    
    # Calculate p-value based on tail
    r = correlations[0]
    if tail == "upper":
        p_value = np.sum(correlations >= r) / (perms + 1)
    elif tail == "lower":
        p_value = np.sum(correlations <= r) / (perms + 1)
    else:  # two-tail
        p_value = np.sum(np.abs(correlations) >= np.abs(r)) / (perms + 1)
    
    return r, p_value

def enzyme_pairwise():

    gene_names = ['lys1','lys2','lys4','lys9','lys12','lys20','aro8','aco2']
    
    # Load embeddings for each gene
    embeddings_dict = {}
    for gene in gene_names:
        path = f'/home/s233201/esm_runs/embeddings/{gene}.npy'
        embeddings = np.load(path)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        embeddings_dict[gene] = embeddings
    
    # Create matrices to store results
    n_genes = len(gene_names)
    r_matrix = np.zeros((n_genes, n_genes))
    p_matrix = np.zeros((n_genes, n_genes))
    
    # Iterate over pairs of genes and perform Mantel tests
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            gene1, gene2 = gene_names[i], gene_names[j]
            emb1 = embeddings_dict[gene1]
            emb2 = embeddings_dict[gene2]
            
            # Use numba-accelerated pdist
            dist1 = squareform(cosine_pdist(emb1))
            dist2 = squareform(cosine_pdist(emb2))
            
            r, p = mantel_test(dist1, dist2, perms=1000)
            print(f"Mantel test between {gene1} and {gene2}: r = {r:.4f}, p = {p:.4f}")
            
            # Store results in matrices
            r_matrix[i, j] = r
            r_matrix[j, i] = r
            p_matrix[i, j] = p
            p_matrix[j, i] = p
    
    # Create DataFrames for the results
    r_df = pd.DataFrame(r_matrix, index=gene_names, columns=gene_names)
    p_df = pd.DataFrame(p_matrix, index=gene_names, columns=gene_names)
    
    # Save results to CSV
    r_df.to_csv('/home/s233201/esm_runs/data/mantel_correlations.csv')
    p_df.to_csv('/home/s233201/esm_runs/data/mantel_pvalues.csv')
    
    # Create heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_df, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('Mantel Test Correlations')
    plt.tight_layout()
    plt.savefig('/home/s233201/esm_runs/mantel_correlations_heatmap.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Mantel Test P-values')
    plt.tight_layout()
    plt.savefig('/home/s233201/esm_runs/mantel_pvalues_heatmap.png')
    plt.close()

def enzyme_phylo_direct_pairs():
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    r_values = []
    p_values = []

    for gene in gene_names:
        # load embeddings
        embed = np.load(f'/home/s233201/esm_runs/embeddings/{gene}.npy')
        if embed.ndim == 1:
            embed = embed.reshape(-1, 1)
        
        # calculate cosine distance
        dist = squareform(cosine_pdist(embed))

        # load phylo 
        phylo = pd.read_csv(f'/home/s233201/full_dist_mats/full_mat_{gene.upper()}.csv', sep='\s+', header=None, skiprows=1)
        accessions = pd.read_csv('/home/s233201/esm_runs/clusters/phylo_clusters/lys20_dm_clusters.csv')['accession'].values
        
        # Set first column as index and first row as columns
        phylo.set_index(0, inplace=True)
        phylo.columns = phylo.index
        
        # Reorder both rows and columns using the accessions
        phylo = phylo.reindex(index=accessions, columns=accessions)
        
        dm = phylo.values
        del phylo

        print(dist.shape, dm.shape)
        # Perform Mantel test
        r, p = mantel_test(dist, dm, perms=1000)
        r_values.append(r)
        p_values.append(p)
        print(f"Mantel test between {gene} and phylo: r = {r:.4f}, p = {p:.4f}")

    # Create DataFrame for the results
    results_df = pd.DataFrame({
        'Gene': gene_names,
        'Correlation': r_values,
        'P-value': p_values
    }).set_index('Gene')

    # Create heatmap for correlations and p-values
    plt.figure(figsize=(10, 4))
    
    # Plot correlation values
    plt.subplot(1, 2, 1)
    sns.heatmap(results_df[['Correlation']], annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('Mantel Test Correlations\nwith Phylogenetic Distance')
    
    # Plot p-values
    plt.subplot(1, 2, 2)
    sns.heatmap(results_df[['P-value']], annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Mantel Test P-values\nwith Phylogenetic Distance')
    
    plt.tight_layout()
    plt.savefig('/home/s233201/esm_runs/mantel_phylo_comparison.png')
    plt.close()

    # Save results to CSV
    results_df.to_csv('/home/s233201/esm_runs/data/mantel_phylo_results.csv')

def plot_covariance_phylo():
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    
    # Create subplot figure
    fig = make_subplots(rows=4, cols=2, 
                        subplot_titles=gene_names,
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1)

    for idx, gene in enumerate(gene_names):
        print(f"Processing {gene}...")
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # load embeddings and calculate distances as before
        embed = np.load(f'/home/s233201/esm_runs/embeddings/{gene}.npy')
        if embed.ndim == 1:
            embed = embed.reshape(-1, 1)
        dist = squareform(cosine_pdist(embed))

        # load and process phylo data as before
        phylo = pd.read_csv(f'/home/s233201/full_dist_mats/full_mat_{gene.upper()}.csv', 
                           sep='\s+', header=None, skiprows=1)
        accessions = pd.read_csv('/home/s233201/esm_runs/clusters/phylo_clusters/lys20_dm_clusters.csv')['accession'].values
        phylo.set_index(0, inplace=True)
        phylo.columns = phylo.index
        phylo = phylo.reindex(index=accessions, columns=accessions)
        dm = phylo.values

        # Get indices for hover text
        triu_indices = np.triu_indices_from(dist, k=1)
        pair_indices = [f"{accessions[i]}-{accessions[j]}" 
                       for i, j in zip(triu_indices[0], triu_indices[1])]
        
        # Create scatter trace
        fig.add_trace(
            go.Scatter(
                x=dist[triu_indices],
                y=dm[triu_indices],
                mode='markers',
                marker=dict(size=2),
                opacity=0.5,
                hovertext=pair_indices,
                hoverinfo='text',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Cosine Distance", row=row, col=col)
        fig.update_yaxes(title_text="Phylogenetic Distance", row=row, col=col)

    # Update layout
    fig.update_layout(
        height=1600,
        width=1000,
        title_text="Covariance Plots",
        showlegend=False
    )

    # Save as interactive HTML
    fig.write_html('/home/s233201/esm_runs/plots/covariance_plots_interactive.html')
    
    # Also save static version
    fig.write_image('/home/s233201/esm_runs/plots/covariance_plots.png', 
                    width=1000, height=1600, scale=2)

if __name__ == '__main__':

    #enzyme_phylo_direct_pairs()
    plot_covariance_phylo()

