import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from compare_clusters import variation_of_information, split_join_distance

# Define the desired gene order
GENE_ORDER = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]

def get_matching_cluster_files(esm_dir, phylo_dir):
    # Special cases for longer prefixes
    special_genes = {'lys20', 'lys12'}
    
    # Get files with dynamic prefix length
    esm_files = {}
    for f in os.listdir(esm_dir):
        if f.endswith('_clusters.csv'):
            prefix = f[:5] if any(f.startswith(gene) for gene in special_genes) else f[:4]
            esm_files[prefix] = f
            
    phylo_files = {}
    for f in os.listdir(phylo_dir):
        if f.endswith('_clusters.csv'):
            prefix = f[:5] if any(f.startswith(gene) for gene in special_genes) else f[:4]
            phylo_files[prefix] = f
    
    # Get common prefixes
    common_prefixes = set(esm_files.keys()) & set(phylo_files.keys())
    
    # Create pairs maintaining the specified order
    ordered_pairs = []
    for gene in GENE_ORDER:
        prefix = gene[:5] if gene in special_genes else gene[:4]
        if prefix in common_prefixes:
            ordered_pairs.append((
                os.path.join(esm_dir, esm_files[prefix]),
                os.path.join(phylo_dir, phylo_files[prefix]),
                gene
            ))
    
    return ordered_pairs

def compare_clusters(esm_file, phylo_file):
    # Read and sort both cluster files
    esm_df = pd.read_csv(esm_file).sort_values('accession')
    phylo_df = pd.read_csv(phylo_file).sort_values('accession')
    
    # Ensure we're comparing the same sequences
    common_idx = set(esm_df['accession']) & set(phylo_df['accession'])
    esm_df = esm_df[esm_df['accession'].isin(common_idx)]
    phylo_df = phylo_df[phylo_df['accession'].isin(common_idx)]
    
    # Calculate all metrics
    ari = adjusted_rand_score(esm_df['cluster'], phylo_df['cluster'])
    vi = variation_of_information(esm_df['cluster'].values, phylo_df['cluster'].values)
    sj = split_join_distance(esm_df['cluster'].values, phylo_df['cluster'].values)
    
    return ari, vi, sj

def create_comparison_plots(file_pairs):
    labels = [pair[2] for pair in file_pairs]  # Using prefixes as labels
    ari_scores = []
    vi_scores = []
    sj_scores = []
    
    for esm_file, phylo_file, _ in file_pairs:
        ari, vi, sj = compare_clusters(esm_file, phylo_file)
        ari_scores.append(ari)
        vi_scores.append(vi)
        sj_scores.append(sj)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot ARI scores with flipped colormap
    sns.heatmap([ari_scores], ax=ax1, cmap='viridis_r',  # flipped colormap
                xticklabels=labels, yticklabels=['ARI'],
                annot=True, fmt='.2f', cbar_kws={'orientation': 'horizontal'})
    ax1.set_title('Adjusted Rand Index (higher is closer)')
    
    # Plot VI scores with flipped colormap
    sns.heatmap([vi_scores], ax=ax2, cmap='viridis',  # changed to flipped colormap
                xticklabels=labels, yticklabels=['VI'],
                annot=True, fmt='.2f', cbar_kws={'orientation': 'horizontal'})
    ax2.set_title('Variation of Information (lower is closer)')
    
    # Plot SJ scores with flipped colormap
    sns.heatmap([sj_scores], ax=ax3, cmap='viridis',  # changed to flipped colormap
                xticklabels=labels, yticklabels=['SJ'],
                annot=True, fmt='.2f', cbar_kws={'orientation': 'horizontal'})
    ax3.set_title('Split-Join Distance (lower is closer)')
    
    plt.tight_layout()
    return fig

def main():
    esm_dir = '/home/s233201/esm_runs/clusters/5_20_05_leaf'
    phylo_dir = '/home/s233201/esm_runs/clusters/phylo_clusters'
    
    file_pairs = get_matching_cluster_files(esm_dir, phylo_dir)
    fig = create_comparison_plots(file_pairs)
    
    plt.savefig('phylo_cluster_comparison_metrics.png')
    plt.close()
    
    print("Comparison plots have been saved to 'phylo_cluster_comparison_metrics.png'")

if __name__ == "__main__":
    main()
