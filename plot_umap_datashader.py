import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import os
from Bio import SeqIO
import multiprocessing
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import Sets1to3
import holoviews as hv
from holoviews.operation.datashader import datashade, spread
import colorcet as cc # Using colorcet for potentially more distinct colors if needed
# Ensure necessary backend for PNG export is installed (e.g., pip install bokeh selenium geckodriver)
# You might need to install geckodriver or chromedriver separately.
hv.extension('bokeh')

# Define phylum colors (can use ColorBrewer or other palettes)
# Using a slightly different palette for variety, ensure keys match your data
PHYLUM_COLORS = {
    'Ascomycota': '#377eb8',    # Blue
    'Basidiomycota': '#e41a1c', # Red
    'Mucoromycota': '#4daf4a', # Green
    'Zoopagomycota': '#984ea3', # Purple
    'Chytridiomycota': '#ff7f00', # Orange
    'Blastocladiomycota': '#ffff33',# Yellow (might be hard to see)
    'Cryptomycota': '#a65628', # Brown
    # Add more phyla and colors as needed
    'Unknown': '#bdbdbd' # Grey for Unknown
}

# --- Helper Functions (copied from plot_umap.py) ---

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    if not os.path.exists(embedding_file):
        print(f"Warning: Embedding file not found: {embedding_file}")
        return None, None
    embeddings = np.load(embedding_file)
    print(f"Loaded embedding array shape: {embeddings.shape} for {os.path.basename(embedding_file)}")
    protein_ids = [f"protein_{i+1}" for i in range(len(embeddings))]
    return embeddings, protein_ids

def load_taxa_info(taxa_file):
    """Load taxa information from CSV file."""
    df = pd.read_csv(taxa_file)
    # Ensure 'Accession' and 'Phylum' columns exist
    if 'Accession' not in df.columns or 'Phylum' not in df.columns:
        raise ValueError("Taxa file must contain 'Accession' and 'Phylum' columns.")
    # Handle potential missing phylum values if necessary
    df['Phylum'] = df['Phylum'].fillna('Unknown')
    return dict(zip(df.Accession, df.Phylum))

def get_fasta_accessions(fasta_file):
    """Get accessions in order from FASTA file."""
    if not os.path.exists(fasta_file):
        print(f"Warning: FASTA file not found: {fasta_file}")
        return []
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        acc = record.id.split()[0]
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

# --- Datashader Specific Functions ---

def process_gene_data_ds(gene_name):
    """Process single gene: load data, compute UMAP, return DataFrame."""
    print(f"Processing {gene_name}...")
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv'
    fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta'
    output_dir = '/home/s233201/esm_runs/plots' # Ensure output dir exists

    os.makedirs(output_dir, exist_ok=True)

    embeddings, _ = load_embeddings(embedding_file)
    if embeddings is None:
        return gene_name, None # Skip if embeddings are missing

    taxa_dict = load_taxa_info(taxa_file)
    fasta_accessions = get_fasta_accessions(fasta_file)
    if not fasta_accessions:
         return gene_name, None # Skip if fasta is missing or empty

    # Ensure all accessions are in taxa_dict, assign 'Unknown' if not
    phyla = [taxa_dict.get(acc, 'Unknown') for acc in fasta_accessions]

    # Check consistency
    if len(phyla) != embeddings.shape[0]:
         print(f"Warning: Mismatch between number of sequences in FASTA ({len(phyla)}) and embeddings ({embeddings.shape[0]}) for {gene_name}. Skipping.")
         return gene_name, None

    print(f"Calculating UMAP for {gene_name} with {embeddings.shape[0]} points...")
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0) # Ensure diagonal is zero
    distance_matrix = np.maximum(distance_matrix, 0) # Ensure no negative values due to precision

    umap_model = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=min(100, embeddings.shape[0] - 1) # Adjust n_neighbors if fewer points
    )

    try:
        umap_coords = umap_model.fit_transform(distance_matrix)
    except ValueError as e:
        print(f"Error during UMAP for {gene_name}: {e}. Skipping.")
        return gene_name, None


    df = pd.DataFrame({
        'x': umap_coords[:, 0],
        'y': umap_coords[:, 1],
        'phylum': pd.Categorical(phyla) # Use categorical for efficiency
    })
    print(f"Finished processing {gene_name}.")
    return gene_name, df

def create_datashader_plot(df, gene_name, phylum_colors):
    """Create a datashader plot colored by phylum using HoloViews."""
    if df is None or df.empty:
        # Return an empty plot or placeholder if data is missing
        # Increase size for consistency
        return hv.Text(0, 0, f"{gene_name}\n(No data)").opts(width=800, height=800)

    # Add 'Unknown' to the color key if present in data but not in PHYLUM_COLORS
    unique_phyla = df['phylum'].unique()
    plot_colors = {k: v for k, v in phylum_colors.items() if k in unique_phyla}
    if 'Unknown' in unique_phyla and 'Unknown' not in plot_colors:
        plot_colors['Unknown'] = '#bdbdbd' # Default grey for Unknown

    points = hv.Points(df, kdims=['x', 'y'], vdims=['phylum'])

    # Use datashade with the specified color key, add spreading
    shaded = datashade(points,
                       aggregator=ds.count_cat('phylum'),
                       color_key=plot_colors,
                       min_alpha=100 # Make points more visible
                       ).opts(width=800, height=800, title=gene_name) # Increased resolution

    # Apply point spreading after shading
    # spread_pixels controls how much points are enlarged
    spreaded = spread(shaded, px=1, shape='circle')

    return spreaded.opts(
        hv.opts.RGB(xaxis=None, yaxis=None, show_grid=False, title=gene_name)
    )

def process_and_save_single_gene(gene_name):
    """Helper function to process one gene and save its datashader plot (HTML and PNG)."""
    output_dir = '/home/s233201/esm_runs/plots'
    gene_name_processed, df = process_gene_data_ds(gene_name)

    if df is not None:
        plot = create_datashader_plot(df, gene_name_processed, PHYLUM_COLORS)
        # Save HTML
        output_path_html = os.path.join(output_dir, f'embeddings_umap_ds_{gene_name_processed.lower()}.html')
        print(f"Saving HTML plot for {gene_name_processed} to {output_path_html}")
        hv.save(plot, output_path_html, backend='bokeh')
        # Save PNG
        output_path_png = os.path.join(output_dir, f'embeddings_umap_ds_{gene_name_processed.lower()}.png')
        print(f"Saving PNG plot for {gene_name_processed} to {output_path_png}")
        try:
            hv.save(plot, output_path_png, backend='bokeh', fmt='png', dpi=150) # Added fmt='png' and dpi
        except Exception as e:
            print(f"Error saving PNG for {gene_name_processed}: {e}. Ensure selenium and a webdriver (geckodriver/chromedriver) are installed and in PATH.")
    else:
        print(f"Skipping plot generation for {gene_name} due to missing data or errors.")


if __name__ == "__main__":
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    use_subplots = True  # Toggle for subplot vs separate plots
    output_dir = '/home/s233201/esm_runs/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Add 'Unknown' to the main color dict if not already present
    if 'Unknown' not in PHYLUM_COLORS:
        PHYLUM_COLORS['Unknown'] = '#bdbdbd'

    num_cores = multiprocessing.cpu_count()
    print(f"Running on {num_cores} cores")

    if use_subplots:
        print("Processing genes for subplot layout...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Returns list of tuples: [(gene_name, dataframe), ...]
            results = pool.map(process_gene_data_ds, gene_names)

        plots = {}
        for gene_name, df in results:
             if df is not None:
                 print(f"Creating plot object for {gene_name}")
                 # Pass the potentially updated PHYLUM_COLORS
                 plots[gene_name] = create_datashader_plot(df, gene_name, PHYLUM_COLORS)
             else:
                 print(f"No data/plot for {gene_name}")
                 # Optionally create a placeholder (increase size)
                 plots[gene_name] = hv.Text(0, 0, f"{gene_name}\n(No data)").opts(width=800, height=800, xaxis=None, yaxis=None)


        # Ensure plots are ordered correctly according to gene_names for the layout
        ordered_plots = [plots[name] for name in gene_names if name in plots]

        if not ordered_plots:
            print("No plots generated. Exiting.")
        else:
            # Arrange plots in a grid (e.g., 2 rows, 4 columns)
            # HoloViews Layout arranges plots row by row
            n_cols = 4
            layout = hv.Layout(ordered_plots).cols(n_cols)

            # Save HTML Layout
            output_path_html = os.path.join(output_dir, 'embeddings_umap_ds_all.html')
            print(f"Saving combined subplot layout (HTML) to {output_path_html}")
            hv.save(layout, output_path_html, backend='bokeh')
            print("Combined HTML plot saved.")

            # Save PNG Layout
            output_path_png = os.path.join(output_dir, 'embeddings_umap_ds_all.png')
            print(f"Saving combined subplot layout (PNG) to {output_path_png}")
            try:
                # Increase dpi for better PNG resolution
                hv.save(layout, output_path_png, backend='bokeh', fmt='png', dpi=150)
                print("Combined PNG plot saved.")
            except Exception as e:
                print(f"Error saving combined PNG: {e}. Ensure selenium and a webdriver (geckodriver/chromedriver) are installed and in PATH.")


    else:
        print("Processing genes for individual plots (HTML and PNG)...")
        # Use pool.map with the helper function for individual saving
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(process_and_save_single_gene, gene_names)
        print("Individual plots saved.")
