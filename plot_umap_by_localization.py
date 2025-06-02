import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
from Bio import SeqIO
import os
import multiprocessing
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import gridplot, row, column
from bokeh.palettes import Category20
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Label, Arrow, NormalHead
import matplotlib.pyplot as plt

# Special organisms for labeling (add at top with other constants)
SPECIAL_ORGANISMS = {
    "GCF_000146045": "S. cerevisiae",
    "GCA_000230395": "A. niger",
    "GCF_000002655": "A. fumigatus",
    "GCF_000182895": "C. cinerea",
    "GCF_000149305": "R. delemar",
    "GCF_028827035": "P. chrysogenum"
}

# Define color palette for localizations
LOCALIZATION_COLORS = {
    'Nucleus': '#1f77b4',
    'Cytoplasm': '#ff7f0e',
    'Mitochondrion': '#2ca02c',
    'Plasma membrane': '#d62728',
    'Secreted': '#9467bd',
    'Endoplasmic reticulum': '#8c564b',
    'Golgi apparatus': '#e377c2',
    'Peroxisome': '#7f7f7f',
    'Vacuole': '#bcbd22',
    'Cell wall': '#17becf',
    'Other': '#cccccc'
}

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    embeddings = np.load(embedding_file)
    return embeddings

def get_fasta_accessions(fasta_file):
    """Get accessions in order from FASTA file."""
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        acc = record.id.split()[0]
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

def get_localizations(gene_name):
    """Get protein localizations from DeepLoc2 output."""
    deeploc_path = f'/home/s233201/deeploc2_package/outputs/accurate/{gene_name.lower()}'
    csv_files = [f for f in os.listdir(deeploc_path) if f.endswith('.csv')]
    if not csv_files:
        return None

    df = pd.read_csv(os.path.join(deeploc_path, csv_files[0]))
    # Process Protein_IDs the same way as in get_fasta_accessions
    df['Protein_ID'] = df['Protein_ID'].apply(lambda x: x.split()[0].split('.')[0])
    return dict(zip(df['Protein_ID'], df['Localizations']))

def process_gene_data(gene_name):
    """Process single gene and return UMAP coordinates with localizations."""
    embedding_file = f'/home/s233201/esm_runs/embeddings_new/{gene_name}_embeddings.npy'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    umap_cache_file = f'/home/s233201/esm_runs/umap_cache/{gene_name}_umap_loc.npz'
    
    os.makedirs('/home/s233201/esm_runs/umap_cache', exist_ok=True)
    print(f"Processing {gene_name}...")
    
    # Get localizations
    loc_dict = get_localizations(gene_name)
    if loc_dict is None:
        print(f"No localization data found for {gene_name}")
        return gene_name, None, None, None
    
    fasta_accessions = get_fasta_accessions(fasta_file)
    localizations = [loc_dict.get(acc, 'Other') for acc in fasta_accessions]
    
    # Try to load cached UMAP coordinates
    if os.path.exists(umap_cache_file):
        print(f"Loading cached UMAP coordinates for {gene_name}")
        cached_data = np.load(umap_cache_file)
        return gene_name, cached_data['umap_coords'], localizations, fasta_accessions
    
    # If no cache exists, proceed with processing
    embeddings = load_embeddings(embedding_file)
    
    # Check for minimum number of embeddings
    if embeddings.shape[0] < 2:
        print(f"Too few embeddings for {gene_name}")
        return gene_name, None, None, None
    
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    # Adjust n_neighbors based on number of samples
    n_neighbors_val = min(100, embeddings.shape[0] - 1)
    if n_neighbors_val < 2:
        print(f"Too few neighbors for {gene_name}")
        return gene_name, None, None, None
    
    umap_model = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=n_neighbors_val
    )
    
    umap_coords = umap_model.fit_transform(distance_matrix)
    
    # Cache the results
    np.savez(umap_cache_file, umap_coords=umap_coords)
    
    return gene_name, umap_coords, localizations, fasta_accessions

def add_organism_labels(ax, umap_coords, accessions, fontsize=6):
    """Add labels with lines pointing to special organisms."""
    # Get axis limits to ensure labels stay within bounds
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = x_max - x_min
    height = y_max - y_min
    
    # Reduced fontsize for labels
    fontsize = fontsize * 0.8
    
    for i, acc in enumerate(accessions):
        # Strip version number if present
        base_acc = acc.split('.')[0]
        if base_acc in SPECIAL_ORGANISMS:
            # Get the coordinates for this organism
            x, y = umap_coords[i, 0], umap_coords[i, 1]
            
            # Alternate between top and bottom placement
            idx = list(SPECIAL_ORGANISMS.keys()).index(base_acc)
            if idx % 2 == 0:
                offset_dist = height * 0.1
                offset_y = offset_dist
                va_setting = 'bottom'
            else:
                offset_dist = -height * 0.1
                offset_y = offset_dist
                va_setting = 'top'
            
            # Keep x-coordinate the same for straight vertical lines
            offset_x = 0
            
            # Make sure label stays within axis limits with margin
            margin = min(width, height) * 0.05
            label_x = x
            label_y = min(max(y + offset_y, y_min + margin), y_max - margin)
            
            # Add a straight line pointing to the organism
            ax.annotate(
                SPECIAL_ORGANISMS[base_acc],
                xy=(x, y),
                xytext=(label_x, label_y),
                fontsize=fontsize,
                color='black',
                ha='center',
                va=va_setting,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0",
                    color='black',
                    lw=0.5,
                    alpha=0.7
                ),
                bbox=None,
                zorder=5
            )

def main_localization_plot():
    """Generate UMAP plots colored by protein localization."""
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    output_dir = '/home/s233201/esm_runs/plots_localization'
    os.makedirs(output_dir, exist_ok=True)
    
    num_cores = multiprocessing.cpu_count()
    print(f"Running on {num_cores} cores")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_gene_data, gene_names)
    
    # Filter out failed results
    valid_results = [r for r in results if r[1] is not None]
    
    if not valid_results:
        print("No valid data to plot")
        return
    
    plots = []
    tools = "pan,box_zoom,wheel_zoom,reset,save"
    
    # Create legend figure with empty Legend layout
    legend_fig = figure(width=200, height=350, title="Localizations")
    legend_fig.axis.visible = False
    legend_fig.grid.visible = False
    legend_fig.toolbar.logo = None
    legend_fig.toolbar_location = None
    legend_fig.background_fill_alpha = 0
    legend_fig.border_fill_alpha = 0
    
    # Add empty legend that will be populated by legend_group
    legend_fig.add_layout(Legend(click_policy="hide",
                                background_fill_alpha=0.7,
                                label_text_font_size='12pt',
                                spacing=10,
                                padding=20), 'center')

    # Collect all unique localizations that appear in the data
    all_found_localizations = set()
    for _, _, gene_localizations, _ in valid_results:
        if gene_localizations: # Ensure gene_localizations is not None
            all_found_localizations.update(gene_localizations)

    bokeh_legend_items = []
    # Use sorted keys of LOCALIZATION_COLORS for a consistent order in the legend
    for loc_name in sorted(LOCALIZATION_COLORS.keys()):
        if loc_name in all_found_localizations or loc_name == 'Other': # Only add if present in the current dataset
            loc_color = LOCALIZATION_COLORS[loc_name]
            # Create a dummy renderer on legend_fig for the legend item.
            # This renderer is added to legend_fig but made invisible.
            # Its sole purpose is to be referenced by the LegendItem.
            dummy_renderer = legend_fig.scatter(
                x=[0], y=[0], # Position doesn't matter as it's invisible
                color=loc_color,
                size=10,       # Standard size for legend marker
                visible=False  # Make the glyph itself invisible on the plot
            )
            bokeh_legend_items.append(LegendItem(label=loc_name, renderers=[dummy_renderer]))
    
    # Assign the created items to the legend object in legend_fig
    # Assuming it's the first (and only) item in legend_fig.center, and it's a Legend instance
    print(f"Contents of legend_fig.center: {legend_fig.center}")  # Debugging line
    
    actual_legend_object = None
    for item in legend_fig.center:
        if isinstance(item, Legend):
            actual_legend_object = item
            break
    
    if actual_legend_object is not None:
        actual_legend_object.items = bokeh_legend_items
        actual_legend_object.location = "center"
        actual_legend_object.orientation = "vertical"
        actual_legend_object.spacing = 5
        actual_legend_object.glyph_width = 15
        actual_legend_object.glyph_height = 15
        actual_legend_object.label_standoff = 5
        actual_legend_object.margin = 5
        actual_legend_object.padding = 5
        actual_legend_object.background_fill_alpha = 0.7
        print(f"Number of legend items assigned: {len(bokeh_legend_items)}")  # Debugging line
    else:
        # This case should ideally not happen if legend_fig is set up as intended
        print("Warning: Could not find the manually added Legend object in legend_fig to populate.")

    # Create plots for each gene
    for gene_name, umap_coords, localizations, accessions in valid_results:
        p = figure(width=350, height=350, tools=tools, title=gene_name)
        p.title.text_font_size = '16pt'
        
        # Remove grid lines and axis ticks
        p.grid.grid_line_color = None
        p.axis.visible = False
        
        hover = HoverTool(tooltips=[
            ("Accession", "@accession"),
            ("Localization", "@localization"),
            ("Organism", "@organism")
        ])
        p.add_tools(hover)
        
        # Calculate plot bounds for better label positioning
        x_range = np.ptp(umap_coords[:, 0])
        y_range = np.ptp(umap_coords[:, 1])
        scale_factor = min(x_range, y_range) * 0.1
        
        # Create a single data source with all points
        all_x = umap_coords[:, 0]
        all_y = umap_coords[:, 1]
        all_acc = accessions
        all_loc = localizations
        all_org = [SPECIAL_ORGANISMS.get(acc.split('.')[0], "") for acc in accessions]
        all_colors = [LOCALIZATION_COLORS.get(loc, '#cccccc') for loc in localizations]
        
        source = ColumnDataSource(data={
            'x': all_x,
            'y': all_y,
            'accession': all_acc,
            'localization': all_loc,
            'organism': all_org,
            'color': all_colors
        })
        
        # Plot all points at once with legend_group
        p.scatter('x', 'y',
                 source=source,
                 color='color',
                 size=1.0,
                 alpha=0.7,
                 legend_group='localization')

        # Add organism labels (matching interactive script approach)
        for acc in accessions:
            base_acc = acc.split('.')[0]
            if base_acc in SPECIAL_ORGANISMS:
                idx = accessions.index(acc)
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
                      text=[SPECIAL_ORGANISMS[base_acc]],
                      text_font_size='8pt',
                      text_align='left',
                      text_baseline='bottom')
        
        # Hide individual plot legends
        p.legend.visible = False
        plots.append(p)
    
    # Arrange plots in grid with legend
    grid = gridplot(plots, ncols=4)
    layout = row(grid, legend_fig)
    
    # Save the plot
    output_file(os.path.join(output_dir, 'umap_by_localization.html'))
    save(layout)
    print(f"Plot saved to {output_dir}/umap_by_localization.html")

def plot_localization_static():
    """Generate static matplotlib plots colored by protein localization."""
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    output_dir = '/home/s233201/esm_runs/plots_localization'
    os.makedirs(output_dir, exist_ok=True)
    
    num_cores = multiprocessing.cpu_count()
    print(f"Running on {num_cores} cores")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_gene_data, gene_names)
    
    # Filter out failed results
    valid_results = [r for r in results if r[1] is not None]
    
    if not valid_results:
        print("No valid data to plot")
        return
    
    # Calculate the figure size for a 3x3 grid with wider aspect ratio (matching plot_umap.py)
    n_rows = 3
    n_cols = 3
    subplot_size = 5  # Increased from default
    subplot_aspect_ratio = 1.2  # Make plots wider (width = height * 1.2)
    fig_width = subplot_size * n_cols * subplot_aspect_ratio
    fig_height = subplot_size * n_rows
    
    # Create figure with wider subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Create plots for each gene (first 8 positions)
    for idx, (gene_name, umap_coords, localizations, accessions) in enumerate(valid_results):
        ax = axes[idx]
        
        # Plot points by localization
        for loc in set(localizations):
            mask = [l == loc for l in localizations]
            if any(mask):
                coords_subset = umap_coords[mask]
                color = LOCALIZATION_COLORS.get(loc, '#cccccc')
                ax.scatter(coords_subset[:, 0], coords_subset[:, 1],
                          c=color, label=loc, s=1.0, alpha=0.7)
        
        # Calculate the data limits for this subplot (matching plot_umap.py)
        x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
        y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
        
        # Calculate ranges and centers
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        
        # Apply the aspect ratio to the plot limits
        max_range = max(x_range, y_range) * 1.1  # 10% padding
        x_half_range = max_range * subplot_aspect_ratio / 2
        y_half_range = max_range / 2
        
        # Set limits with appropriate aspect ratio
        ax.set_xlim(x_center - x_half_range, x_center + x_half_range)
        ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
        
        # Add organism labels with larger fontsize (matching plot_umap.py)
        add_organism_labels(ax, umap_coords, accessions, fontsize=12)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a subtle grid for better readability
        ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
        
        # Set title with larger font (matching plot_umap.py)
        ax.set_title(f'{gene_name}', fontsize=18, pad=10, fontweight='bold')

    # Create an attractive legend in the last (9th) cell (matching plot_umap.py)
    legend_ax = axes[8]
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    
    # Add a title to the legend subplot
    legend_ax.text(0.5, 0.95, "Legend", fontsize=20, fontweight='bold', 
                  ha='center', va='top', transform=legend_ax.transAxes)
    
    # Create legend handles and labels
    handles, labels = [], []
    # Get all unique localizations that actually appear in the data
    all_localizations = set()
    for res in valid_results:
        all_localizations.update(res[2])
    
    # Add legend items for localizations that appear in the data with larger markers
    for loc in LOCALIZATION_COLORS.keys():
        if loc in all_localizations or loc == 'Other':  # Always include 'Other'
            color = LOCALIZATION_COLORS[loc]
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=14, 
                          label=loc))
            labels.append(loc)
    
    # Add the localization legend in the center
    legend1 = legend_ax.legend(
        handles=handles,
        labels=labels,
        loc='center',
        bbox_to_anchor=(0.5, 0.5),
        fontsize=13,
        title="Localization",
        title_fontsize=16,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Adjust layout with more space between subplots (matching plot_umap.py)
    plt.tight_layout(pad=2.5)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'umap_by_localization_static.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static plot saved to {output_dir}/umap_by_localization_static.png")

if __name__ == "__main__":
    main_localization_plot()
    plot_localization_static()
