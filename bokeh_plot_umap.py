import numpy as np
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import gridplot, column
from bokeh.palettes import Set3
from bokeh.models import (ColumnDataSource, HoverTool, Div, 
                         Panel, Tabs, Legend, LegendItem,
                         CDSView, GroupFilter)  # Added missing imports
import colorcet as cc  # Better color palettes
from umap import UMAP
import os
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from Bio import SeqIO
import pandas as pd

# Reuse color scheme from original script
PHYLUM_COLORS = {
    'Ascomycota': '#377eb8',    
    'Basidiomycota': '#e41a1c',
    'Mucoromycota': '#4daf4a',  
    'Zoopagomycota': '#984ea3',
    'Chytridiomycota': '#ff7f00',
    'Blastocladiomycota': '#ffff33',
    'Cryptomycota': '#a65628'   
}

# Reuse the data loading functions from the original script
def load_embeddings(embedding_file):
    return np.load(embedding_file)

def load_taxa_info(taxa_file):
    df = pd.read_csv(taxa_file)
    return dict(zip(df.Accession, df.Phylum))

def get_fasta_accessions(fasta_file):
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        acc = record.id.split()[0]
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

def process_gene_data_bokeh(gene_name):
    """Process single gene data - reused from original with minimal changes"""
    embedding_file = f'/home/s233201/esm_runs/embeddings_new/{gene_name}_embeddings.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    umap_cache_file = f'/home/s233201/esm_runs/umap_cache/{gene_name}_umap.npz'
    
    # Try to load cached UMAP coordinates
    if os.path.exists(umap_cache_file):
        print(f"Loading cached UMAP coordinates for {gene_name}")
        cached_data = np.load(umap_cache_file)
        return gene_name, cached_data['umap_coords'], cached_data['phyla'], cached_data['accessions']
    
    # If no cache exists, process the data
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    filtered_accessions = [acc for acc in fasta_accessions if acc in taxa_dict]
    phyla = [taxa_dict[acc] for acc in filtered_accessions]
    
    embeddings_all = load_embeddings(embedding_file)
    accession_to_idx = {acc: i for i, acc in enumerate(fasta_accessions)}
    indices_to_keep = [accession_to_idx[acc] for acc in filtered_accessions if acc in accession_to_idx]
    
    if not indices_to_keep:
        return gene_name, np.array([]), [], []
    
    embeddings = embeddings_all[indices_to_keep, :]
    if embeddings.shape[0] < 2:
        return gene_name, np.array([]), [], []
        
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    n_neighbors_val = min(100, embeddings.shape[0] - 1)
    umap_model = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=n_neighbors_val 
    )
    
    umap_coords = umap_model.fit_transform(distance_matrix)
    
    # Cache the results
    np.savez(umap_cache_file, 
             umap_coords=umap_coords,
             phyla=phyla,
             accessions=filtered_accessions)
    
    return gene_name, umap_coords, phyla, filtered_accessions

def create_bokeh_plot(gene_name, umap_coords, phyla, accessions, width=400, height=400):
    """Create a single Bokeh plot"""
    
    # Create data source
    source_data = {
        'x': umap_coords[:, 0],
        'y': umap_coords[:, 1],
        'phylum': phyla,
        'accession': accessions
    }
    source = ColumnDataSource(source_data)
    
    # Create plot
    p = figure(width=width, height=height, title=gene_name)
    p.title.text_font_size = '16pt'
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ('Accession', '@accession'),
        ('Phylum', '@phylum')
    ])
    p.add_tools(hover)
    
    # Plot points by phylum
    legend_items = []
    for phylum in PHYLUM_COLORS:
        view = CDSView(filters=[GroupFilter(column_name='phylum', group=phylum)])
        circle = p.circle('x', 'y', size=3, color=PHYLUM_COLORS[phylum], 
                         alpha=0.6, source=source, view=view)
        legend_items.append(LegendItem(label=phylum, renderers=[circle]))
    
    # Add legend
    legend = Legend(items=legend_items)
    p.add_layout(legend, 'right')
    
    # Style the plot
    p.grid.visible = False
    p.axis.visible = False
    p.min_border = 0
    
    return p

def main_bokeh_plots(gene_names, output_file_path):
    """Create Bokeh plots with tabs for combined and individual views"""
    
    # Process all genes
    with multiprocessing.Pool() as pool:
        results = pool.map(process_gene_data_bokeh, gene_names)
    
    # Filter valid results
    valid_results = [res for res in results if res[1] is not None and res[1].shape[0] > 0]
    
    # Create combined view (small plots)
    plots = []
    for gene_name, coords, phyla, accessions in valid_results:
        p = create_bokeh_plot(gene_name, coords, phyla, accessions)
        plots.append(p)
    
    # Arrange in grid
    n_cols = 4
    grid = gridplot(plots, ncols=n_cols)
    combined_tab = Panel(child=grid, title="All Plots")
    
    # Create individual tabs (large plots)
    individual_tabs = []
    for gene_name, coords, phyla, accessions in valid_results:
        p = create_bokeh_plot(gene_name, coords, phyla, accessions, 
                            width=1000, height=800)
        tab = Panel(child=p, title=gene_name)
        individual_tabs.append(tab)
    
    # Combine all tabs
    tabs = Tabs(tabs=[combined_tab] + individual_tabs)
    
    # Save to HTML file
    output_file(output_file_path)
    save(tabs)

if __name__ == "__main__":
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    output_dir = '/home/s233201/esm_runs/plots_bokeh'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'bokeh_umap_plots.html')
    
    main_bokeh_plots(gene_names, output_file_path)
