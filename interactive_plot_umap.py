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
    #'Cryptomycota': '#a65628'   # Brown
}

special_organisms = {
    "GCF_000146045": "S. cerevisiae",
    "GCA_000230395": "A. niger",
    "GCF_000002655": "A. fumigatus",
    "GCF_000182895": "C. cinerea",
    "GCF_000149305": "R. delemar",
    "GCF_028827035": "P. chrysogenum"
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

def load_outlier_accessions():
    """Load outlier accessions from the text file."""
    with open('/home/s233201/outliers_set.txt', 'r') as f:
        return {line.strip().split('.')[0] for line in f}

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
    
    # Load outliers
    outliers = load_outlier_accessions()
    
    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    
    # Filter out outliers and keep only those in taxa_dict
    filtered_accessions = [acc for acc in fasta_accessions 
                          if acc in taxa_dict and acc.split('.')[0] not in outliers]
    phyla = [taxa_dict[acc] for acc in filtered_accessions]
    
    # Try to load cached UMAP coordinates with error handling
    if os.path.exists(umap_cache_file):
        print(f"Loading cached UMAP coordinates for {gene_name}")
        try:
            cached_data = np.load(umap_cache_file)
            if all(key in cached_data for key in ['umap_coords', 'phyla', 'accessions']):
                # Convert numpy arrays back to lists for consistency
                cached_phyla = cached_data['phyla'].tolist()
                cached_accessions = cached_data['accessions'].tolist()
                return gene_name, cached_data['umap_coords'], cached_phyla, cached_accessions
            else:
                print(f"Cache file for {gene_name} is missing required data. Recomputing...")
                os.remove(umap_cache_file)  # Remove incomplete cache file
        except Exception as e:
            print(f"Error loading cache for {gene_name}: {e}. Recomputing...")
            if os.path.exists(umap_cache_file):
                os.remove(umap_cache_file)
    
    # If no cache exists, proceed with normal processing
    embeddings_all = load_embeddings(embedding_file)

    # Create a mapping from fasta_accessions to their original indices
    accession_to_idx = {acc: i for i, acc in enumerate(fasta_accessions)}
    
    # Filter embeddings to match filtered_accessions (after outlier removal)
    indices_to_keep = [accession_to_idx[acc] for acc in filtered_accessions if acc in accession_to_idx]
    
    if not indices_to_keep:
        print(f"Warning: No matching accessions found in taxa file for {gene_name} after outlier removal. Skipping.")
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
    
    # Cache the results with all required fields
    try:
        np.savez(umap_cache_file, 
                 umap_coords=umap_coords,
                 phyla=np.array(phyla, dtype=str),  # Convert lists to numpy arrays
                 accessions=np.array(filtered_accessions, dtype=str))
    except Exception as e:
        print(f"Warning: Failed to cache results for {gene_name}: {e}")
        if os.path.exists(umap_cache_file):
            os.remove(umap_cache_file)
    
    return gene_name, umap_coords, phyla, filtered_accessions


def adjust_label_positions(labels_data, plot_width, plot_height, min_distance=20):
    """Simple algorithm to adjust label positions to avoid overlaps."""
    adjusted_labels = []
    
    for i, (x, y, text) in enumerate(labels_data):
        adjusted_x, adjusted_y = x, y
        
        # Check against all previously placed labels
        for prev_x, prev_y, _ in adjusted_labels:
            dx = adjusted_x - prev_x
            dy = adjusted_y - prev_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                # Move label away from collision
                if dx == 0 and dy == 0:
                    # If exactly on top, move arbitrarily
                    adjusted_x += min_distance
                else:
                    # Move along the collision vector
                    scale = min_distance / distance
                    adjusted_x = prev_x + dx * scale
                    adjusted_y = prev_y + dy * scale
        
        adjusted_labels.append((adjusted_x, adjusted_y, text))
    
    return adjusted_labels

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
    
    # Create and style the legend figure as before
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
        
        # Calculate data limits for aspect ratio adjustment
        x_min, x_max = np.min(umap_coords[:, 0]), np.max(umap_coords[:, 0])
        y_min, y_max = np.min(umap_coords[:, 1]), np.max(umap_coords[:, 1])
        
        # Calculate ranges and centers
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        
        # Apply consistent aspect ratio (1.2 as in plot_umap.py)
        max_range = max(x_range, y_range) * 1.1  # 10% padding
        aspect_ratio = 1.2
        x_half_range = max_range * aspect_ratio / 2
        y_half_range = max_range / 2
        
        # Set plot ranges with appropriate aspect ratio
        p.x_range.start = x_center - x_half_range
        p.x_range.end = x_center + x_half_range
        p.y_range.start = y_center - y_half_range
        p.y_range.end = y_center + y_half_range
        
        # Calculate scale factor for labels based on the adjusted ranges
        scale_factor = min(x_half_range, y_half_range) * 0.1
        
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
                        size=2,  # Slightly larger point size
                        marker='circle',
                        alpha=0.7)
        
        # Special organisms annotations - simplified without manual arrows
        special_labels = []
        for acc in special_organisms:
            if acc in accessions_list:
                idx = accessions_list.index(acc)
                x_pos = umap_coords[idx, 0]
                y_pos = umap_coords[idx, 1]
                special_labels.append((x_pos, y_pos, special_organisms[acc]))
        
        # Adjust label positions to avoid overlaps
        if special_labels:
            adjusted_labels = adjust_label_positions(special_labels, 350, 350, min_distance=scale_factor)
            
            for (orig_x, orig_y, text), (adj_x, adj_y, _) in zip(special_labels, adjusted_labels):
                # Add line from original point to label if position was adjusted
                if abs(orig_x - adj_x) > 0.01 or abs(orig_y - adj_y) > 0.01:
                    p.line([orig_x, adj_x], [orig_y, adj_y], 
                          line_color='black', line_width=1, line_alpha=0.7)
                
                # Add the text at adjusted position
                p.text(adj_x, adj_y,
                      text=[text],
                      text_font_size='8pt',
                      text_align='center',
                      text_baseline='bottom',
                      text_color='black',
                      text_font_style='bold')
        
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

