import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import plotly.express as px
import plotly.offline as offline

from phylo_correlation import load_phylo_matrices

def create_interactive_plot(dm1: pd.DataFrame, dm2: pd.DataFrame, gene_name: str, max_points: int = 1000):
    """Generates an interactive Plotly plot for two distance matrices."""
    if 'accession' not in dm1.columns:
        dm1 = dm1.reset_index().rename(columns={'index': 'accession'})
    if 'accession' not in dm2.columns:
        dm2 = dm2.reset_index().rename(columns={'index': 'accession'})

    dm1 = dm1.set_index('accession')
    dm2 = dm2.set_index('accession')

    common_accessions = dm1.index.intersection(dm2.index)

    if len(common_accessions) < 2:
        print(f"Warning: Less than 2 common accessions for {gene_name}. Skipping plot generation.")
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

    # Get the accessions for each point
    accession_pairs = [(common_accessions_sorted[i], common_accessions_sorted[j]) for i, j in zip(rows, cols)]

    df = pd.DataFrame({'x': dm1_flat, 'y': dm2_flat, 'accession_pairs': accession_pairs})

    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    if df.empty:
        print(f"Warning: No data points after alignment/sampling for {gene_name}. Skipping plot generation.")
        return None

    fig = px.scatter(df, x='x', y='y', hover_data=['accession_pairs'],
                     title=f"Phylogenetic Distance Comparison: {gene_name}",
                     labels={"x": "Distance Matrix 1", "y": "Distance Matrix 2"})

    # Customize the plot
    fig.update_traces(marker=dict(size=5, opacity=0.6))

    return fig

def main():
    output_dir = '/home/s233201/figures'
    os.makedirs(output_dir, exist_ok=True)

    gene_names = ["lys20", "aco2", "lys4", "lys12", "aro8", "lys2", "lys9", "lys1"]
    phylo_matrices = load_phylo_matrices(gene_names)

    if phylo_matrices:
        gene_names = list(phylo_matrices.keys())
        num_genes = len(gene_names)
        
        # Generate plots sequentially
        for i in tqdm(range(num_genes), desc="Generating Plots"):
            for j in range(i + 1, num_genes):
                gene1, gene2 = gene_names[i], gene_names[j]
                try:
                    plot = create_interactive_plot(
                        phylo_matrices[gene1].copy(),
                        phylo_matrices[gene2].copy(),
                        gene_name=f"{gene1}_vs_{gene2}",
                        max_points=1000
                    )
                    if plot:
                        # show(plot) # replaced with saving to a file
                        filename = os.path.join(output_dir, f"{gene1}_vs_{gene2}.html")
                        offline.plot(plot, filename=filename, auto_open=False)
                except Exception as e:
                    print(f"Error generating plot for {gene1} vs {gene2}: {e}")
    else:
        print("Failed to load phylogenetic matrices.")

if __name__ == '__main__':
    main()
