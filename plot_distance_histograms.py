import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import multiprocessing as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns

matplotlib.use('agg')

def process_gene_distances(gene):
    """Process a single gene and return distance data"""
    print(f"Processing gene: {gene}")

    # Load ESM embeddings
    embed = np.load(f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_embeddings.npy')
    if embed.ndim == 1:
        embed = embed.reshape(-1, 1)

    # Load evolutionary distance matrix
    phylo_raw = pd.read_csv(f'/home/s233201/full_dist_mats/clean/full_mat_{gene.upper()}.csv',
                       sep='\s+', header=None, skiprows=1)

    # Get accessions from phylo matrix and set as index/columns
    phylo_accessions = phylo_raw.iloc[:, 0].values
    phylo = pd.DataFrame(phylo_raw.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)

    # Load accessions for embeddings
    embed_accessions = []
    with open(f'/home/s233201/esm_runs/embeddings_new/{gene.upper()}_ids.txt', 'r') as f:
        embed_accessions = [line.strip() for line in f.readlines()[:embed.shape[0]]]

    # Create DataFrame with accessions and embeddings
    embed_df = pd.DataFrame(embed, index=embed_accessions)

    # Convert to distance matrix using cosine distance
    embed_dist = pd.DataFrame(
        squareform(pdist(embed_df.values, metric='cosine')),
        index=embed_accessions,
        columns=embed_accessions
    )

    # Find common accessions
    common_accessions = embed_dist.index.intersection(phylo.index)

    # Align both matrices
    embed_aligned = embed_dist.loc[common_accessions, common_accessions]
    phylo_aligned = phylo.loc[common_accessions, common_accessions]

    # Get upper triangle values (excluding diagonal)
    embed_array = embed_aligned.to_numpy()
    phylo_array = phylo_aligned.to_numpy()
    rows, cols = np.triu_indices(embed_array.shape[0], k=1)
    
    esm_distances = embed_array[rows, cols]
    evolutionary_distances = phylo_array[rows, cols]

    return {
        'gene': gene,
        'esm_distances': esm_distances,
        'evolutionary_distances': evolutionary_distances,
        'esm_mean': np.mean(esm_distances),
        'evolutionary_mean': np.mean(evolutionary_distances),
        'n_pairs': len(esm_distances)
    }

def plot_individual_histograms(gene_data):
    """Create figure with individual subplots for each gene"""
    n_genes = len(gene_data)
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, data in enumerate(gene_data):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Calculate initial histograms to identify significant bins
        esm_counts, esm_bins = np.histogram(data['esm_distances'], bins=50)
        evol_counts, evol_bins = np.histogram(data['evolutionary_distances'], bins=50)
        
        # Filter bins with at least 0.5% of counts
        esm_threshold = len(data['esm_distances']) * 0.005
        evol_threshold = len(data['evolutionary_distances']) * 0.005
        
        esm_mask = esm_counts >= esm_threshold
        evol_mask = evol_counts >= evol_threshold
        
        # Get data ranges for significant bins
        if np.any(esm_mask):
            esm_significant_bins = esm_bins[:-1][esm_mask]
            esm_min, esm_max = esm_significant_bins.min(), esm_significant_bins.max() + (esm_bins[1] - esm_bins[0])
            # Filter data to significant range and rebin
            esm_filtered_data = data['esm_distances'][(data['esm_distances'] >= esm_min) & (data['esm_distances'] <= esm_max)]
            ax.hist(esm_filtered_data, bins=30, alpha=0.6, label='ESM distance', 
                   color='skyblue', density=False, range=(esm_min, esm_max))
        
        if np.any(evol_mask):
            evol_significant_bins = evol_bins[:-1][evol_mask]
            evol_min, evol_max = evol_significant_bins.min(), evol_significant_bins.max() + (evol_bins[1] - evol_bins[0])
            # Filter data to significant range and rebin
            evol_filtered_data = data['evolutionary_distances'][(data['evolutionary_distances'] >= evol_min) & (data['evolutionary_distances'] <= evol_max)]
            ax.hist(evol_filtered_data, bins=30, alpha=0.6, label='Evolutionary distance', 
                   color='lightcoral', density=False, range=(evol_min, evol_max))
        
        # Add vertical lines for means
        ax.axvline(data['esm_mean'], color='blue', linestyle='--', 
                  label=f'ESM mean: {data["esm_mean"]:.3f}')
        ax.axvline(data['evolutionary_mean'], color='red', linestyle='--',
                  label=f'Evolutionary mean: {data["evolutionary_mean"]:.3f}')
        
        ax.set_title(f'{data["gene"].upper()}\n(n={data["n_pairs"]} pairs)', fontsize=14)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_genes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_overlaid_histograms(gene_data):
    """Create figure with all genes overlaid"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(gene_data)))
    
    # ESM distances
    for i, data in enumerate(gene_data):
        # Calculate initial histogram to identify significant bins
        counts, bins = np.histogram(data['esm_distances'], bins=50)
        
        # Filter bins with at least 0.5% of counts
        threshold = len(data['esm_distances']) * 0.005
        mask = counts >= threshold
        
        if np.any(mask):
            significant_bins = bins[:-1][mask]
            data_min, data_max = significant_bins.min(), significant_bins.max() + (bins[1] - bins[0])
            # Filter data to significant range and rebin
            filtered_data = data['esm_distances'][(data['esm_distances'] >= data_min) & (data['esm_distances'] <= data_max)]
            ax1.hist(filtered_data, bins=30, alpha=0.6, 
                   label=f'{data["gene"].upper()} (μ={data["esm_mean"]:.3f})',
                   color=colors[i], density=False, range=(data_min, data_max))
    
    ax1.set_title('ESM Distance Distributions', fontsize=16)
    ax1.set_xlabel('ESM Distance', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Evolutionary distances
    for i, data in enumerate(gene_data):
        # Calculate initial histogram to identify significant bins
        counts, bins = np.histogram(data['evolutionary_distances'], bins=50)
        
        # Filter bins with at least 0.5% of counts
        threshold = len(data['evolutionary_distances']) * 0.005
        mask = counts >= threshold
        
        if np.any(mask):
            significant_bins = bins[:-1][mask]
            data_min, data_max = significant_bins.min(), significant_bins.max() + (bins[1] - bins[0])
            # Filter data to significant range and rebin
            filtered_data = data['evolutionary_distances'][(data['evolutionary_distances'] >= data_min) & (data['evolutionary_distances'] <= data_max)]
            ax2.hist(filtered_data, bins=30, alpha=0.6,
                   label=f'{data["gene"].upper()} (μ={data["evolutionary_mean"]:.3f})',
                   color=colors[i], density=False, range=(data_min, data_max))
    
    ax2.set_title('Evolutionary Distance Distributions', fontsize=16)
    ax2.set_xlabel('Evolutionary Distance', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_interactive_histograms(gene_data):
    """Create interactive figure with toggleable genes"""
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ESM Distance Distributions', 'Evolutionary Distance Distributions'),
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1[:len(gene_data)]
    
    for i, data in enumerate(gene_data):
        gene_name = data['gene'].upper()
        color = colors[i % len(colors)]
        
        # ESM distances - filter and rebin
        esm_counts, esm_bins = np.histogram(data['esm_distances'], bins=50)
        esm_threshold = len(data['esm_distances']) * 0.005
        esm_mask = esm_counts >= esm_threshold
        
        if np.any(esm_mask):
            esm_significant_bins = esm_bins[:-1][esm_mask]
            esm_min, esm_max = esm_significant_bins.min(), esm_significant_bins.max() + (esm_bins[1] - esm_bins[0])
            esm_filtered_data = data['esm_distances'][(data['esm_distances'] >= esm_min) & (data['esm_distances'] <= esm_max)]
            
            # Create histogram for ESM
            esm_hist, esm_bin_edges = np.histogram(esm_filtered_data, bins=30, range=(esm_min, esm_max))
            esm_bin_centers = (esm_bin_edges[:-1] + esm_bin_edges[1:]) / 2
            
            fig.add_trace(
                go.Scatter(
                    x=esm_bin_centers,
                    y=esm_hist,
                    mode='lines',
                    fill='tonexty' if i == 0 else 'tozeroy',
                    name=f'{gene_name}',
                    line=dict(color=color),
                    fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.3)'),
                    legendgroup=gene_name,
                    showlegend=True,
                    hovertemplate=f'<b>{gene_name} ESM</b><br>' +
                                  'Distance: %{x:.3f}<br>' +
                                  'Count: %{y}<br>' +
                                  f'Mean: {data["esm_mean"]:.3f}<br>' +
                                  '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Evolutionary distances - filter and rebin
        evol_counts, evol_bins = np.histogram(data['evolutionary_distances'], bins=50)
        evol_threshold = len(data['evolutionary_distances']) * 0.005
        evol_mask = evol_counts >= evol_threshold
        
        if np.any(evol_mask):
            evol_significant_bins = evol_bins[:-1][evol_mask]
            evol_min, evol_max = evol_significant_bins.min(), evol_significant_bins.max() + (evol_bins[1] - evol_bins[0])
            evol_filtered_data = data['evolutionary_distances'][(data['evolutionary_distances'] >= evol_min) & (data['evolutionary_distances'] <= evol_max)]
            
            # Create histogram for evolutionary
            evol_hist, evol_bin_edges = np.histogram(evol_filtered_data, bins=30, range=(evol_min, evol_max))
            evol_bin_centers = (evol_bin_edges[:-1] + evol_bin_edges[1:]) / 2
            
            fig.add_trace(
                go.Scatter(
                    x=evol_bin_centers,
                    y=evol_hist,
                    mode='lines',
                    fill='tonexty' if i == 0 else 'tozeroy',
                    name=f'{gene_name}',
                    line=dict(color=color),
                    fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.3)'),
                    legendgroup=gene_name,
                    showlegend=False,  # Don't duplicate in legend
                    hovertemplate=f'<b>{gene_name} Evolutionary</b><br>' +
                                  'Distance: %{x:.3f}<br>' +
                                  'Count: %{y}<br>' +
                                  f'Mean: {data["evolutionary_mean"]:.3f}<br>' +
                                  '<extra></extra>'
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_layout(
        title="Interactive Distance Distributions - Click legend to toggle genes",
        height=600,
        width=1200,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="ESM Distance", row=1, col=1)
    fig.update_xaxes(title_text="Evolutionary Distance", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    return fig

def perform_statistical_tests(gene_data):
    """Perform paired t-tests between all gene distributions"""
    n_genes = len(gene_data)
    gene_names = [data['gene'].upper() for data in gene_data]
    
    # Initialize matrices for p-values
    esm_pvalues = np.ones((n_genes, n_genes))
    evol_pvalues = np.ones((n_genes, n_genes))
    
    # Perform pairwise paired t-tests for ESM distances
    print("\n" + "="*80)
    print("PAIRWISE PAIRED T-TESTS FOR ESM DISTANCES")
    print("="*80)
    
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            # Get minimum sample size for fair comparison
            min_size = min(len(gene_data[i]['esm_distances']), len(gene_data[j]['esm_distances']))
            
            # Randomly sample equal number of distances
            esm_i = np.random.choice(gene_data[i]['esm_distances'], size=min_size, replace=False)
            esm_j = np.random.choice(gene_data[j]['esm_distances'], size=min_size, replace=False)
            
            # Perform paired t-test
            t_stat, p_val = stats.ttest_rel(esm_i, esm_j)
            esm_pvalues[i, j] = esm_pvalues[j, i] = p_val
            
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{gene_names[i]} vs {gene_names[j]}: t={t_stat:.3f}, p={p_val:.6f} {significance}")
    
    # Perform pairwise paired t-tests for evolutionary distances
    print("\n" + "="*80)
    print("PAIRWISE PAIRED T-TESTS FOR EVOLUTIONARY DISTANCES")
    print("="*80)
    
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            # Get minimum sample size for fair comparison
            min_size = min(len(gene_data[i]['evolutionary_distances']), len(gene_data[j]['evolutionary_distances']))
            
            # Randomly sample equal number of distances
            evol_i = np.random.choice(gene_data[i]['evolutionary_distances'], size=min_size, replace=False)
            evol_j = np.random.choice(gene_data[j]['evolutionary_distances'], size=min_size, replace=False)
            
            # Perform paired t-test
            t_stat, p_val = stats.ttest_rel(evol_i, evol_j)
            evol_pvalues[i, j] = evol_pvalues[j, i] = p_val
            
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{gene_names[i]} vs {gene_names[j]}: t={t_stat:.3f}, p={p_val:.6f} {significance}")
    
    # ESM vs Evolutionary distances within each gene
    print("\n" + "="*80)
    print("ESM vs EVOLUTIONARY DISTANCES WITHIN EACH GENE")
    print("="*80)
    
    within_gene_results = []
    for data in gene_data:
        gene_name = data['gene'].upper()
        
        # Get minimum sample size for fair comparison
        min_size = min(len(data['esm_distances']), len(data['evolutionary_distances']))
        
        # Randomly sample equal number of distances
        esm_sample = np.random.choice(data['esm_distances'], size=min_size, replace=False)
        evol_sample = np.random.choice(data['evolutionary_distances'], size=min_size, replace=False)
        
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(esm_sample, evol_sample)
        
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{gene_name}: ESM vs Evolutionary: t={t_stat:.3f}, p={p_val:.6f} {significance}")
        
        within_gene_results.append({
            'gene': gene_name,
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    return {
        'esm_pvalues': esm_pvalues,
        'evol_pvalues': evol_pvalues,
        'gene_names': gene_names,
        'within_gene_results': within_gene_results
    }

def plot_pvalue_heatmaps(stats_results, output_dir):
    """Create heatmaps showing p-values from statistical tests"""
    gene_names = stats_results['gene_names']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ESM p-values heatmap
    mask = np.triu(np.ones_like(stats_results['esm_pvalues']), k=1)
    sns.heatmap(stats_results['esm_pvalues'], 
                xticklabels=gene_names, yticklabels=gene_names,
                annot=True, fmt='.3f', cmap='viridis_r', 
                mask=mask, ax=ax1, cbar_kws={'label': 'p-value'})
    ax1.set_title('ESM Distance Pairwise Paired T-test P-values', fontsize=14)
    
    # Evolutionary p-values heatmap
    sns.heatmap(stats_results['evol_pvalues'], 
                xticklabels=gene_names, yticklabels=gene_names,
                annot=True, fmt='.3f', cmap='viridis_r', 
                mask=mask, ax=ax2, cbar_kws={'label': 'p-value'})
    ax2.set_title('Evolutionary Distance Pairwise Paired T-test P-values', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistical_test_pvalues.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar plot for within-gene comparisons
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    genes = [result['gene'] for result in stats_results['within_gene_results']]
    p_values = [result['p_value'] for result in stats_results['within_gene_results']]
    colors = ['red' if p < 0.05 else 'blue' for p in p_values]
    
    bars = ax.bar(genes, p_values, color=colors, alpha=0.7)
    ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold')
    ax.set_ylabel('P-value', fontsize=12)
    ax.set_xlabel('Gene', fontsize=12)
    ax.set_title('ESM vs Evolutionary Distances Within Each Gene (Paired T-test P-values)', fontsize=14)
    ax.legend()
    
    # Add significance annotations
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        if p_val < 0.001:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   '***', ha='center', va='bottom', fontweight='bold')
        elif p_val < 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   '**', ha='center', va='bottom', fontweight='bold')
        elif p_val < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   '*', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/within_gene_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_statistics(gene_data):
    """Print summary statistics for all genes"""
    print("\n" + "="*80)
    print("DISTANCE SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Gene':<8} {'ESM Mean':<12} {'Evol Mean':<12} {'ESM Std':<12} {'Evol Std':<12} {'N Pairs':<10}")
    print("-"*80)
    
    for data in gene_data:
        esm_std = np.std(data['esm_distances'])
        evol_std = np.std(data['evolutionary_distances'])
        print(f"{data['gene'].upper():<8} {data['esm_mean']:<12.4f} {data['evolutionary_mean']:<12.4f} "
              f"{esm_std:<12.4f} {evol_std:<12.4f} {data['n_pairs']:<10}")
    
    # Overall statistics
    all_esm = np.concatenate([data['esm_distances'] for data in gene_data])
    all_evol = np.concatenate([data['evolutionary_distances'] for data in gene_data])
    
    print("-"*80)
    print(f"{'OVERALL':<8} {np.mean(all_esm):<12.4f} {np.mean(all_evol):<12.4f} "
          f"{np.std(all_esm):<12.4f} {np.std(all_evol):<12.4f} {len(all_esm):<10}")
    print("="*80)

if __name__ == '__main__':
    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    
    # Process all genes in parallel
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        gene_data = pool.map(process_gene_distances, gene_names)
    
    # Create output directory
    output_dir = 'esm_runs/plots/distance_histograms'
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary statistics
    print_summary_statistics(gene_data)
    
    # Perform statistical tests
    stats_results = perform_statistical_tests(gene_data)
    
    # Create statistical test visualizations
    plot_pvalue_heatmaps(stats_results, output_dir)
    
    # Create and save individual histograms
    fig1 = plot_individual_histograms(gene_data)
    fig1.savefig(f'{output_dir}/individual_gene_histograms.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Create and save overlaid histograms
    fig2 = plot_overlaid_histograms(gene_data)
    fig2.savefig(f'{output_dir}/overlaid_histograms.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Create and save interactive histograms
    fig3 = plot_interactive_histograms(gene_data)
    fig3.write_html(f'{output_dir}/interactive_histograms.html')
    
    print(f"\nPlots saved to {output_dir}/")
    print("- individual_gene_histograms.png: Separate subplots for each gene")
    print("- overlaid_histograms.png: All genes overlaid in two panels")
    print("- interactive_histograms.html: Interactive plot with toggleable genes")
    print("- statistical_test_pvalues.png: Heatmaps of pairwise paired t-test p-values")
    print("- within_gene_comparisons.png: ESM vs evolutionary comparisons within each gene")
    print("\nStatistical significance: *** p<0.001, ** p<0.01, * p<0.05")
