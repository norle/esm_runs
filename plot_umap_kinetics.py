import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from Bio import SeqIO
import multiprocessing
import pickle  # Add at top with other imports
import matplotlib.colors as mcolors # Add for colorbar normalization
import warnings # Add import for warnings

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    embeddings = np.load(embedding_file)
    print(f"Loaded embedding array shape: {embeddings.shape}")
    
    # Generate sequential IDs since we don't have protein IDs anymore
    protein_ids = [f"protein_{i+1}" for i in range(len(embeddings))]
    
    return embeddings, protein_ids

def load_kinetic_data(kinetic_file):
    """Load kinetic data (Pre_label) from tab-separated file."""
    if not os.path.exists(kinetic_file):
        print(f"Error: Kinetic data file not found at {kinetic_file}")
        return {} # Return empty dict if file not found
    try:
        df = pd.read_csv(kinetic_file, sep='\t')
        if 'Accession' not in df.columns or 'Pre_label' not in df.columns:
            print(f"Error: Missing 'Accession' or 'Pre_label' column in {kinetic_file}")
            return {}
        # Ensure Pre_label is numeric, coercing errors to NaN
        df['Pre_label'] = pd.to_numeric(df['Pre_label'], errors='coerce')
        if df['Pre_label'].isnull().all():
            print(f"Warning: All 'Pre_label' values in {kinetic_file} are missing or non-numeric.")
        
        # Create dictionary, removing version numbers from accessions to match FASTA processing
        kinetic_map = {}
        for acc, label in zip(df.Accession, df.Pre_label):
            acc_base = acc.split('.')[0] # Remove version number
            kinetic_map[acc_base] = label
        return kinetic_map
    except Exception as e:
        print(f"Error loading or processing kinetic data file {kinetic_file}: {e}")
        return {}

def get_fasta_accessions(fasta_file):
    """Get accessions in order from FASTA file."""
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Assuming the accession is the first part of the header before any spaces
        acc = record.id.split()[0]
        # Remove version number if present
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions

def create_umap_plot(embeddings, protein_ids, output_dir, gene_name, kinetic_labels=None, ax=None, save_format='png', vmin=None, vmax=None):
    """Create and save UMAP plot of embeddings, colored by kinetic labels."""
    print(f"Input embedding matrix shape: {embeddings.shape}")
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    umap = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.2,
        n_neighbors=100
    )
    
    distance_matrix = 1 - similarity_matrix
    umap_coords = umap.fit_transform(distance_matrix)

    # If no axis provided, create new figure
    standalone_plot = ax is None
    if standalone_plot:
        fig, ax = plt.subplots(figsize=(10, 8)) # Use fig for colorbar later

    if kinetic_labels is not None:
        # Ensure kinetic_labels is a numpy array for easier handling
        kinetic_labels = np.array(kinetic_labels)
        valid_mask = ~np.isnan(kinetic_labels) # Mask to exclude NaN values

        # Use provided vmin/vmax or calculate from data if not provided
        if vmin is None:
            vmin = np.nanmin(kinetic_labels)
        if vmax is None:
            vmax = np.nanmax(kinetic_labels)

        scatter = ax.scatter(umap_coords[valid_mask, 0], umap_coords[valid_mask, 1],
                           alpha=0.6,
                           c=kinetic_labels[valid_mask],
                           cmap='viridis', # Use a continuous colormap
                           s=7,
                           vmin=vmin,
                           vmax=vmax)
        # Optionally plot NaN values in a different color/marker
        # nan_mask = np.isnan(kinetic_labels)
        # ax.scatter(umap_coords[nan_mask, 0], umap_coords[nan_mask, 1],
        #            alpha=0.6, color='grey', marker='x', s=7, label='NaN')

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add colorbar only if it's a standalone plot
        if standalone_plot:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Pre_label', rotation=270, labelpad=15)
            # Add legend for NaN values if plotted
            # if np.any(nan_mask):
            #     ax.legend()

    else:
        # Fallback if no kinetic labels are provided
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=10)
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_title(f'{gene_name}')

    # Only save if we're not using subplots (standalone plot)
    if standalone_plot:
        os.makedirs(output_dir, exist_ok=True)
        if save_format == 'png':
            plt.savefig(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}.png'),
                       dpi=300, bbox_inches='tight')
        elif save_format == 'fig':
            with open(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}.fig.pickle'), 'wb') as f:
                pickle.dump(fig, f) # Save the figure object
        plt.close(fig) # Close the figure

def main(gene_name, use_subplots=False, fig=None, ax=None, save_format='png'):
    # Paths
    embedding_file = f'/home/s233201/esm_runs/embeddings/filtered_embeddings/{gene_name.lower()}.npy'
    kinetic_file = '/home/s233201/Kinetic_parameters_predicted_label.txt' # ADDED
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    output_dir = '/home/s233201/esm_runs/plots'

    kinetic_dict = load_kinetic_data(kinetic_file) # ADDED
    fasta_accessions = get_fasta_accessions(fasta_file)
    # Map accessions to kinetic labels, handle missing keys with NaN
    # Use the full accession (version already removed by get_fasta_accessions)
    kinetic_labels = [kinetic_dict.get(acc, np.nan) for acc in fasta_accessions] # MODIFIED: Removed [:13]

    print("Loading embeddings...")
    embeddings, protein_ids = load_embeddings(embedding_file)
    print(f"Loaded {len(protein_ids)} sequences with embedding shape: {embeddings.shape}")

    print("Creating UMAP plot...")
    # Pass kinetic_labels instead of phyla
    create_umap_plot(embeddings, protein_ids, output_dir, gene_name,
                    kinetic_labels=kinetic_labels, ax=ax, save_format=save_format)

    if not use_subplots:
        ext = 'png' if save_format == 'png' else 'fig.pickle'
        print(f"Plot saved to {output_dir}/embeddings_umap_{gene_name.lower()}.{ext}")

def process_gene_data(gene_name):
    """Process single gene and return UMAP coordinates and kinetic labels"""
    embedding_file = f'/home/s233201/esm_runs/embeddings_new/{gene_name}_embeddings.npy'
    # taxa_file = '/home/s233201/esm_runs/inputs/taxa.csv' # REMOVED
    kinetic_file = '/home/s233201/Kinetic_parameters_predicted_label.txt' # ADDED
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'

    # taxa_dict = load_taxa_info(taxa_file) # REMOVED
    kinetic_dict = load_kinetic_data(kinetic_file) # ADDED
    if not kinetic_dict: # Check if kinetic data loading failed
         print(f"Warning: Could not load kinetic data for {gene_name}. Skipping kinetic coloring.")
         kinetic_labels = None # Set labels to None if data is missing
    else:
        fasta_accessions = get_fasta_accessions(fasta_file)
        # phyla = [taxa_dict[acc] for acc in fasta_accessions] # REMOVED
        # Map accessions to kinetic labels, handle missing keys with NaN
        # Use the full accession (version already removed by get_fasta_accessions)
        kinetic_labels = [kinetic_dict.get(acc, np.nan) for acc in fasta_accessions] # MODIFIED: Removed [:13]
        if all(np.isnan(k) for k in kinetic_labels):
             print(f"Warning: No matching kinetic labels found for any accessions in {gene_name}.fasta")
             # Optionally set kinetic_labels to None here if you want to disable coloring
             # kinetic_labels = None

    print(f"Processing {gene_name}...")
    embeddings, _ = load_embeddings(embedding_file)

    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    umap = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=100
    )
    
    umap_coords = umap.fit_transform(distance_matrix)
    # Return kinetic_labels instead of phyla
    return gene_name, umap_coords, kinetic_labels

if __name__ == "__main__":
    # gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    gene_names = ['LYS1']
    use_subplots = True  # Toggle for subplot vs separate plots
    save_format = 'png'  # Toggle between 'png' or 'fig' for saving format

    if use_subplots:
        # Process all genes in parallel
        num_cores = multiprocessing.cpu_count()
        print(f"Running on {num_cores} cores")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(process_gene_data, gene_names)

        # Filter out results where kinetic labels might be None or failed loading
        valid_results = [res for res in results if res[2] is not None]
        if not valid_results:
             print("Error: No valid kinetic data found for any processed genes. Cannot create plot.")
             exit() # Or handle differently

        # Determine global min/max for consistent color scaling across subplots
        # Use only non-NaN values from valid results
        all_kinetic_labels = np.concatenate([res[2] for res in valid_results])
        valid_kinetic_labels = all_kinetic_labels[~np.isnan(all_kinetic_labels)] # Filter out NaNs

        if valid_kinetic_labels.size == 0:
            print("Error: All kinetic labels across all genes are NaN. Cannot determine color scale.")
             # Handle this case: maybe assign default vmin/vmax or skip coloring
            global_vmin, global_vmax = 0, 1 # Assign default values
            norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
            print("Warning: Using default color scale [0, 1] due to lack of valid kinetic data.")
        else:
            global_vmin = np.nanmin(valid_kinetic_labels) # Use nanmin on filtered array
            global_vmax = np.nanmax(valid_kinetic_labels) # Use nanmax on filtered array
            # Add a small epsilon if vmin equals vmax to avoid Normalize issues
            if global_vmin == global_vmax:
                global_vmax += 1e-6
            norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)

        cmap = plt.get_cmap('viridis')

        # Calculate the figure size to ensure square subplots
        n_rows = 2
        n_cols = 4
        subplot_size = 4  # Size of each subplot in inches
        fig_width = subplot_size * n_cols
        fig_height = subplot_size * n_rows
        
        # Create figure with square subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = axes.ravel()

        # Plot results with individual square dimensions
        # Iterate through original results to maintain subplot order
        plotted_something = False # Flag to check if any points were plotted for colorbar
        for idx, result_tuple in enumerate(results):
            # Check if this result was valid (had kinetic labels)
            if result_tuple not in valid_results:
                 print(f"Skipping plot for {results[idx][0]} due to missing kinetic data.")
                 # Optionally clear the axis or add a placeholder text
                 ax = axes[idx]
                 ax.text(0.5, 0.5, 'No Kinetic Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 ax.set_title(f'{results[idx][0]}', fontsize=16, pad=5)
                 ax.set_xticks([])
                 ax.set_yticks([])
                 continue # Skip to the next gene

            gene_name, umap_coords, kinetic_labels = result_tuple
            ax = axes[idx]
            # Ensure kinetic_labels is a numpy array and handle NaNs
            kinetic_labels = np.array(kinetic_labels)
            valid_mask = ~np.isnan(kinetic_labels)

            if np.any(valid_mask): # Only plot if there are valid points
                # Use scatter plot with continuous coloring based on kinetic_labels
                scatter = ax.scatter(umap_coords[valid_mask, 0], umap_coords[valid_mask, 1],
                                   alpha=1,
                                   c=kinetic_labels[valid_mask],
                                   cmap=cmap, # Use the chosen colormap
                                   norm=norm, # Use the global normalization
                                   s=0.1,
                                   zorder=2)
                plotted_something = True # Mark that we plotted valid data
            else:
                 # Handle case where a gene had labels, but all were NaN
                 print(f"Warning: All kinetic labels for {gene_name} were NaN. Plotting points in grey.")
                 ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, color='grey', s=0.1, zorder=1)


            # Calculate the data limits for this subplot
            x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
            y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
            
            # Calculate ranges and centers
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            
            # Use the larger range to make the plot square
            max_range = max(x_range, y_range) * 1.1  # Add 10% padding
            
            # Set square limits centered on the data
            ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax.set_ylim(y_center - max_range/2, y_center + max_range/2)

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Set title
            ax.set_title(f'{gene_name}', fontsize=16, pad=5)

        # Adjust layout with enough spacing
        fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to make space for colorbar

        # Add a single colorbar for the entire figure only if valid data was plotted
        if plotted_something:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # Position: [left, bottom, width, height]
            # Create a ScalarMappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([]) # Dummy array needed
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Pre_label', rotation=270, labelpad=15)
        else:
            print("Skipping colorbar generation as no valid data points were plotted.")

        if save_format == 'png':
            plt.savefig('/home/s233201/esm_runs/plots/embeddings_umap_all_kinetic.png', # Updated filename
                       dpi=300, bbox_inches='tight')
        elif save_format == 'fig':
            with open('/home/s233201/esm_runs/plots/embeddings_umap_all_kinetic.fig.pickle', 'wb') as f: # Updated filename
                pickle.dump(fig, f)
        plt.close(fig) # Close the figure object
    else:
        # Process genes in parallel for separate plots
        num_cores = multiprocessing.cpu_count()
        print(f"Running on {num_cores} cores")
        with multiprocessing.Pool(processes=num_cores) as pool:
            from functools import partial
            main_with_format = partial(main, save_format=save_format)
            pool.map(main_with_format, gene_names)
