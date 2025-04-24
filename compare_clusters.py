import numpy as np
import pandas as pd
from collections import Counter
from math import log
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

def read_clusters(filepath):
    df = pd.read_csv(filepath)
    return df['cluster'].values

def variation_of_information(clustering1, clustering2):
    # Remove -1 (noise points) from both clusterings
    mask = (clustering1 != -1) & (clustering2 != -1)
    x = clustering1[mask]
    y = clustering2[mask]
    
    if len(x) == 0:
        return 0.0
    
    # Create contingency matrix
    n = len(x)
    x_counts = Counter(x)
    y_counts = Counter(y)
    contingency = Counter(zip(x, y))
    
    # Calculate entropies and mutual information
    h_x = -sum((cnt/n) * log(cnt/n, 2) for cnt in x_counts.values())
    h_y = -sum((cnt/n) * log(cnt/n, 2) for cnt in y_counts.values())
    
    mi = 0.0
    for (i, j), nij in contingency.items():
        mi += (nij/n) * log((n*nij)/(x_counts[i]*y_counts[j]), 2)
    
    # VI is the sum of the entropies minus twice the mutual information
    vi = h_x + h_y - 2*mi
    return vi

def rand_index(clustering1, clustering2):
    # Remove -1 (noise points) from both clusterings
    mask = (clustering1 != -1) & (clustering2 != -1)
    x = clustering1[mask]
    y = clustering2[mask]
    
    if len(x) == 0:
        return 0.0
    
    return adjusted_rand_score(x, y)

def split_join_distance(clustering1, clustering2):
    # Remove -1 (noise points) from both clusterings
    mask = (clustering1 != -1) & (clustering2 != -1)
    x = clustering1[mask]
    y = clustering2[mask]
    
    if len(x) == 0:
        return 0.0
    
    # Create contingency matrix
    contingency = pd.crosstab(x, y).values
    
    # Calculate splits and joins
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    
    splits = np.sum(np.sum(contingency * (contingency - 1) / 2))
    joins_x = np.sum(row_sums * (row_sums - 1) / 2)
    joins_y = np.sum(col_sums * (col_sums - 1) / 2)
    
    # Calculate split-join distance
    sj_distance = (joins_x - splits) + (joins_y - splits)
    return sj_distance

def get_cluster_files(directory):
    files = []
    for f in os.listdir(directory):
        if f.endswith('_clusters.csv'):
            files.append(os.path.join(directory, f))
    return sorted(files)

def create_comparison_matrices(files):
    n = len(files)
    vi_matrix = np.zeros((n, n))
    ri_matrix = np.zeros((n, n))
    sj_matrix = np.zeros((n, n))
    
    labels = [os.path.basename(f).replace('_clusters.csv', '') for f in files]
    
    for i, file1 in enumerate(files):
        clusters1 = read_clusters(file1)
        for j, file2 in enumerate(files):
            clusters2 = read_clusters(file2)
            vi_matrix[i, j] = variation_of_information(clusters1, clusters2)
            ri_matrix[i, j] = rand_index(clusters1, clusters2)
            sj_matrix[i, j] = split_join_distance(clusters1, clusters2)
    
    return vi_matrix, ri_matrix, sj_matrix, labels

def plot_heatmaps(vi_matrix, ri_matrix, sj_matrix, labels):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # RI heatmap
    sns.heatmap(ri_matrix, ax=ax1, cmap='viridis_r',
                xticklabels=labels, yticklabels=labels)
    ax1.set_title('Adjusted Rand Index')

    # VI heatmap
    sns.heatmap(vi_matrix, ax=ax2, cmap='viridis', 
                xticklabels=labels, yticklabels=labels)
    ax2.set_title('Variation of Information')
    
    
    
    # SJ heatmap
    sns.heatmap(sj_matrix, ax=ax3, cmap='viridis',
                xticklabels=labels, yticklabels=labels)
    ax3.set_title('Split-Join Distance')
    
    plt.tight_layout()
    plt.savefig('cluster_comparison_heatmaps.png')
    plt.close()

def main():
    cluster_dir = '/home/s233201/esm_runs/clusters/phylo_clusters'
    files = get_cluster_files(cluster_dir)
    target_order = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    files = sorted(
        files,
        key=lambda f: target_order.index(
            (os.path.basename(f).replace('_clusters.csv','').strip().lower()[:5]
             if os.path.basename(f).replace('_clusters.csv','').strip().lower().startswith("lys20")
             else os.path.basename(f).replace('_clusters.csv','').strip().lower()[:4])
        ) if (
            (os.path.basename(f).replace('_clusters.csv','').strip().lower()[:5]
             if os.path.basename(f).replace('_clusters.csv','').strip().lower().startswith("lys20")
             else os.path.basename(f).replace('_clusters.csv','').strip().lower()[:4])
            in target_order
        ) else len(target_order)
    )
    
    vi_matrix, ri_matrix, sj_matrix, labels = create_comparison_matrices(files)
    plot_heatmaps(vi_matrix, ri_matrix, sj_matrix, labels)
    
    print("Comparison heatmaps have been saved to 'cluster_comparison_heatmaps.png'")

if __name__ == "__main__":
    main()
