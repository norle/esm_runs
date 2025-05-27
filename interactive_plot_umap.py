import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
from Bio import SeqIO
import os
import multiprocessing
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import gridplot, row, column
from bokeh.io import output_file
from bokeh.palettes import Spectral7
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem

# Define phylum colors (consistent with the original script)
PHYLUM_COLORS = {
    'Ascomycota': '#377eb8',    # Blue
    'Basidiomycota': '#e41a1c', # Red
    'Mucoromycota': '#4daf4a',  # Green
    'Zoopagomycota': '#984ea3', # Purple
    'Chytridiomycota': '#ff7f00',# Orange
    'Blastocladiomycota': '#ffff33',# Yellow
    'Cryptomycota': '#a65628'   # Brown
}

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    embeddings = np.load(embedding_file)
    # print(f"Loaded embedding array shape for {os.path.basename(embedding_file)}: {embeddings.shape}")
    return embeddings

def load_taxa_info(taxa_file):
    """Load taxa information from CSV file."""
    df = pd.read_csv(taxa_file)
    return dict(zip(df.Accession, df.Phylum))

def get_fasta_accessions(fasta_file):
    """Get accessions in order from FASTA file."""
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        acc = record.id.split()[0]
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

def process_gene_data_interactive(gene_name):
    """Process single gene and return UMAP coordinates, phyla, and accessions."""
    # Paths
    embedding_file = f'/home/s233201/esm_runs/embeddings_new/{gene_name}_embeddings.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    umap_cache_file = f'/home/s233201/esm_runs/umap_cache/{gene_name}_umap.npz'
    
    print(f"Processing {gene_name} for interactive plot...")
    
    # Create cache directory if it doesn't exist
    os.makedirs('/home/s233201/esm_runs/umap_cache', exist_ok=True)
    
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    
    filtered_accessions = [acc for acc in fasta_accessions if acc in taxa_dict]
    phyla = [taxa_dict[acc] for acc in filtered_accessions]
    
    # Try to load cached UMAP coordinates
    if os.path.exists(umap_cache_file):
        print(f"Loading cached UMAP coordinates for {gene_name}")
        cached_data = np.load(umap_cache_file)
        return gene_name, cached_data['umap_coords'], cached_data['phyla'], cached_data['accessions']
    
    # If no cache exists, proceed with normal processing
    embeddings_all = load_embeddings(embedding_file)

    # Create a mapping from fasta_accessions to their original indices
    accession_to_idx = {acc: i for i, acc in enumerate(fasta_accessions)}
    
    # Filter embeddings to match filtered_accessions
    # This assumes embeddings are ordered according to the original fasta_file
    indices_to_keep = [accession_to_idx[acc] for acc in filtered_accessions if acc in accession_to_idx]
    
    if not indices_to_keep:
        print(f"Warning: No matching accessions found in taxa file for {gene_name}. Skipping.")
        return gene_name, np.array([]), [], []

    embeddings = embeddings_all[indices_to_keep, :]

    if embeddings.shape[0] == 0:
        print(f"Warning: No embeddings to process for {gene_name} after filtering. Skipping.")
        return gene_name, np.array([]), [], []
    if embeddings.shape[0] < 2 : # UMAP needs at least 2 samples
        print(f"Warning: Not enough data points ({embeddings.shape[0]}) for UMAP for {gene_name}. Skipping.")
        return gene_name, np.array([]), [], []


    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    # Ensure n_neighbors is less than the number of samples
    n_neighbors_val = min(100, embeddings.shape[0] - 1)
    if n_neighbors_val < 2: # UMAP's n_neighbors must be at least 2
        print(f"Warning: n_neighbors too small ({n_neighbors_val}) for {gene_name} with {embeddings.shape[0]} samples. Skipping UMAP.")
        return gene_name, np.array([]), phyla, filtered_accessions


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


def main_interactive_bokeh(gene_names, output_html_file):
    """Generates and saves an interactive UMAP plot with subplots using Bokeh."""
    num_cores = multiprocessing.cpu_count()
    print(f"Running interactive plot generation on {num_cores} cores")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_gene_data_interactive, gene_names)
    
    # Filter out results where umap_coords might be empty
    valid_results = [res for res in results if res[1] is not None and res[1].shape[0] > 0]
    
    if not valid_results:
        print("No valid data to plot after processing all genes.")
        return

    plots = []
    tools = "pan,box_zoom,wheel_zoom,reset,save"
    
    # Create a figure for the shared legend
    legend_fig = figure(width=200, height=350, title="Phyla", frame_width=200, frame_height=350)
    legend_fig.axis.visible = False
    legend_fig.grid.visible = False
    legend_fig.toolbar.logo = None
    legend_fig.toolbar_location = None
    legend_fig.background_fill_alpha = 0
    legend_fig.border_fill_alpha = 0
    
    # Create legend items with larger markers
    legend_items = []
    for phylum in PHYLUM_COLORS.keys():
        r = legend_fig.scatter(x=[0], y=[0], size=15, 
                             color=PHYLUM_COLORS[phylum], 
                             fill_color=PHYLUM_COLORS[phylum],
                             line_color=None)  # Removed legend_label here
        legend_items.append((phylum, [r]))
    
    # Create and style the legend
    legend = Legend(items=legend_items, 
                   click_policy="hide",
                   background_fill_alpha=0.7,
                   label_text_font_size='12pt',
                   spacing=10,
                   padding=20)
    
    legend_fig.add_layout(legend, 'center')
    
    for gene_name, umap_coords, phyla_list, accessions_list in valid_results:
        p = figure(width=350, height=350, tools=tools, title=gene_name)
        p.title.text_font_size = '16pt'
        
        # Remove grid lines and axis ticks
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.minor_tick_line_color = None
        p.axis.major_label_text_color = None
        
        hover = HoverTool(tooltips=[("Accession", "@accession"), ("Phylum", "@phylum")])
        p.add_tools(hover)
        
        # Calculate plot bounds for better label positioning
        x_range = np.ptp(umap_coords[:, 0])
        y_range = np.ptp(umap_coords[:, 1])
        scale_factor = min(x_range, y_range) * 0.1  # 10% of the smaller range
        
        for phylum in PHYLUM_COLORS.keys():
            mask = [p == phylum for p in phyla_list]
            if any(mask):
                coords_subset = umap_coords[mask]
                accessions_subset = [acc for i, acc in enumerate(accessions_list) if mask[i]]
                
                source = ColumnDataSource(data={
                    'x': coords_subset[:, 0],
                    'y': coords_subset[:, 1],
                    'accession': accessions_subset,
                    'phylum': [phylum] * len(accessions_subset)
                })
                
                p.scatter('x', 'y', 
                        source=source,
                        color=PHYLUM_COLORS[phylum],
                        size=1,
                        marker='circle',
                        alpha=0.7)
        
        # Special organisms annotations with improved positioning
        special_organisms = {
            "GCF_000146045": "S. cerevisiae",
            "GCA_000230395": "A. niger",
            "GCF_000002655": "A. fumigatus",
            "GCF_000182895": "C. cinerea",
            "GCF_000149305": "R. delemar",
            "GCF_028827035": "P. chrysogenum"
        }
        
        accessions_list = list(accessions_list)
        for acc in special_organisms:
            if acc in accessions_list:
                idx = accessions_list.index(acc)
                x_pos = umap_coords[idx, 0]
                y_pos = umap_coords[idx, 1]
                
                # Add connecting line with fixed diagonal direction
                offset_x = scale_factor * 0.4
                offset_y = scale_factor * 0.4
                p.segment(x0=x_pos, y0=y_pos, 
                         x1=x_pos + offset_x, 
                         y1=y_pos + offset_y,
                         color='gray',
                         line_width=0.5)
                
                # Add text label in top-right position
                p.text(x_pos + offset_x, 
                      y_pos + offset_y,
                      text=[special_organisms[acc]],
                      text_font_size='8pt',
                      text_align='left',
                      text_baseline='bottom')
        
        plots.append(p)
    
    # Create grid of plots
    n_cols = 4
    grid = gridplot(plots, ncols=n_cols)
    
    # Combine grid with legend figure
    layout = row(grid, legend_fig)
    
    # Save the plot
    output_file(output_html_file)
    save(layout)
    print(f"Interactive plot saved to {output_html_file}")

if __name__ == "__main__":
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    output_dir = '/home/s233201/esm_runs/plots_interactive'
    os.makedirs(output_dir, exist_ok=True)
    output_html_file = os.path.join(output_dir, 'interactive_umap_subplots_bokeh.html')
    
    # Use the new Bokeh function instead of the Plotly one
    main_interactive_bokeh(gene_names, output_html_file)

