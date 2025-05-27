import torch # Not strictly needed here, but often included in ML projects
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity # Not used in this script
import os
import pandas as pd
from Bio import SeqIO
from pathlib import Path # For easier path manipulation

# Define phylum colors using ColorBrewer or similar
PHYLUM_COLORS = {
    'Ascomycota': '#377eb8',    # Blue
    'Basidiomycota': '#e41a1c', # Red
    'Mucoromycota': '#4daf4a', # Green
    'Zoopagomycota': '#984ea3', # Purple
    'Chytridiomycota': '#ff7f00', # Orange
    'Blastocladiomycota': '#ffff33',# Yellow (might be hard to see on white)
    'Cryptomycota': '#a65628', # Brown
    'Unknown': '#bdbdbd' # Grey for unknown or unassigned
}

def load_embeddings_and_ids(embedding_file_path_str):
    """Load embeddings from .npy file and corresponding protein IDs from .txt file."""
    embedding_file_path = Path(embedding_file_path_str)
    
    # Derive IDs file path from embedding file path
    # Assumes IDs file is in the same directory and has a similar name pattern
    # e.g., LYS20_layer12_embeddings.npy -> LYS20_layer12_ids.txt
    ids_file_name = embedding_file_path.name.replace('_embeddings.npy', '_ids.txt')
    ids_file_path = embedding_file_path.parent / ids_file_name

    if not embedding_file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file_path}")
    if not ids_file_path.exists():
        raise FileNotFoundError(f"IDs file not found: {ids_file_path}. Expected at: {ids_file_path}")

    embeddings = np.load(embedding_file_path)
    print(f"Loaded embedding array shape: {embeddings.shape} from {embedding_file_path.name}")
    
    # Remove extra dimension if present (e.g., shape (4990, 1, 1152) -> (4990, 1152))
    if len(embeddings.shape) == 3 and embeddings.shape[1] == 1:
        embeddings = np.squeeze(embeddings, axis=1)
        print(f"Reshaped embedding array to: {embeddings.shape}")
    
    with open(ids_file_path, 'r') as f:
        protein_ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(protein_ids)} protein IDs from {ids_file_path.name}")

    if embeddings.shape[0] != len(protein_ids):
        raise ValueError(f"Mismatch between number of embeddings ({embeddings.shape[0]}) and protein IDs ({len(protein_ids)}).")
        
    return embeddings, protein_ids

def load_taxa_info(taxa_file_path_str):
    """Load taxa information from CSV file."""
    taxa_file_path = Path(taxa_file_path_str)
    if not taxa_file_path.exists():
        raise FileNotFoundError(f"Taxa file not found: {taxa_file_path}")
        
    df = pd.read_csv(taxa_file_path)
    # Ensure 'Accession' and 'Phylum' columns exist
    if 'Accession' not in df.columns or 'Phylum' not in df.columns:
        raise ValueError("Taxa file must contain 'Accession' and 'Phylum' columns.")
    
    # Handle potential missing phylum values by assigning 'Unknown'
    df['Phylum'] = df['Phylum'].fillna('Unknown')
    
    # Clean accession IDs in taxa_dict (remove .version if present)
    taxa_dict = {}
    for _, row in df.iterrows():
        accession = str(row['Accession']).split('.')[0] # Take part before first dot
        taxa_dict[accession] = row['Phylum']
        
    return taxa_dict


def get_fasta_accessions_cleaned(fasta_file_path_str):
    """Get cleaned accessions (ID before first dot) in order from FASTA file."""
    fasta_file_path = Path(fasta_file_path_str)
    if not fasta_file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file_path}")

    accessions = []
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        # Clean accession: take part before first space, then part before first dot
        acc = record.id.split()[0] 
        acc = acc.split('.')[0]
        accessions.append(acc)
    return accessions


def create_pca_plot(embeddings, protein_ids_for_embeddings, taxa_dict, output_dir_str, gene_name, layer_name_str=""):
    """
    Create and save PCA plot of embeddings, colored by phylum.
    protein_ids_for_embeddings: List of IDs corresponding to the rows of embeddings.
    taxa_dict: Dictionary mapping cleaned accession IDs to phyla.
    """
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input embedding matrix shape for PCA: {embeddings.shape}")
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        print(f"No embeddings to plot for {gene_name}. Skipping PCA plot.")
        return

    # Get phyla for each embedding, using the loaded protein_ids
    # Clean the protein IDs from the embedding file just in case they have versions
    phyla_for_plot = []
    valid_embeddings_indices = []
    cleaned_protein_ids_for_plot = []

    for i, pid_original in enumerate(protein_ids_for_embeddings):
        pid_cleaned = pid_original.split('.')[0] # Clean ID to match taxa_dict keys
        phylum = taxa_dict.get(pid_cleaned, 'Unknown') # Default to 'Unknown' if not found
        phyla_for_plot.append(phylum)
        valid_embeddings_indices.append(i)
        cleaned_protein_ids_for_plot.append(pid_cleaned)
    
    if not valid_embeddings_indices:
        print(f"No valid embeddings with phylum info found for {gene_name}. Skipping PCA plot.")
        return
        
    embeddings_to_plot = embeddings[valid_embeddings_indices, :]

    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit(embeddings_to_plot).transform(embeddings_to_plot)
    
    explained_var = pca.explained_variance_ratio_ * 100
    
    plt.figure(figsize=(12, 10)) # Increased figure size for legend
    
    # Ensure PHYLUM_COLORS contains all phyla present in the data, including 'Unknown'
    all_phyla_in_data = set(phyla_for_plot)
    current_phylum_colors = PHYLUM_COLORS.copy()
    for p in all_phyla_in_data:
        if p not in current_phylum_colors:
            print(f"Warning: Phylum '{p}' found in data but not in PHYLUM_COLORS. Assigning grey.")
            current_phylum_colors[p] = '#bdbdbd' # Default to grey

    plotted_phyla = set()
    for phylum_to_plot in sorted(list(all_phyla_in_data)): # Sort for consistent legend order
        mask = [p == phylum_to_plot for p in phyla_for_plot]
        if np.any(mask): # Check if there are any points for this phylum
            plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                       alpha=0.7, label=phylum_to_plot, 
                       color=current_phylum_colors.get(phylum_to_plot, '#bdbdbd'), # Fallback color
                       s=15) # Increased point size
            plotted_phyla.add(phylum_to_plot)
    
    if plotted_phyla:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, markerscale=1.5, frameon=True, title="Phylum")
    else:
        # Fallback if no phyla info or all unknown (though previous checks should prevent this)
        plt.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.6, s=15, color='#bdbdbd', label='Data Points')
    
    title_str = f'PCA of {gene_name} Embeddings'
    if layer_name_str:
        title_str += f' (Layer {layer_name_str})'
    plt.title(title_str, fontsize=16)
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}% variance explained)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}% variance explained)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside

    plot_filename = f'embeddings_pca_{gene_name.lower()}'
    if layer_name_str:
        plot_filename += f'_layer{layer_name_str}'
    plot_filename += '.png'
    
    plt.savefig(output_dir / plot_filename, dpi=300) # bbox_inches='tight' removed, using tight_layout
    plt.close()
    print(f"PCA plot saved to {output_dir / plot_filename}")


def main():
    gene_name = 'LYS1'
    layer_identifier = '12' # As in "layer12" from your embedding script's output naming
    
    # Construct paths based on new naming convention from embedding script
    embedding_base_dir = Path(f'/home/s233201/esm_runs/embeddings_new_12th')
    embedding_file = embedding_base_dir / f'{gene_name.upper()}_embeddings.npy'
    
    taxa_file = '/home/s233201/esm_runs/data/taxa_clean_0424.csv' # Or your updated taxa file
    # FASTA file is only needed if you want to ensure the original order for some reason,
    # but it's better to rely on the IDs saved with embeddings.
    # fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta' 
    
    output_dir_plots = Path('/home/s233201/esm_runs/plots_pca_layer_embeddings') # New output dir for these plots
    
    print(f"--- Processing Gene: {gene_name}, Layer: {layer_identifier} ---")
    
    # Load taxa information (maps cleaned accession to phylum)
    print("Loading taxa information...")
    try:
        taxa_dict = load_taxa_info(taxa_file)
    except Exception as e:
        print(f"Error loading taxa info: {e}")
        return

    # Load embeddings and their corresponding protein IDs
    print("Loading embeddings and protein IDs...")
    try:
        embeddings, protein_ids_from_file = load_embeddings_and_ids(str(embedding_file))
    except FileNotFoundError as e:
        print(e)
        print(f"Skipping PCA for {gene_name} (layer {layer_identifier}) due to missing files.")
        return
    except ValueError as e:
        print(e)
        print(f"Skipping PCA for {gene_name} (layer {layer_identifier}) due to data mismatch.")
        return
        
    if embeddings.shape[0] == 0:
        print(f"No embeddings found in {embedding_file}. Skipping.")
        return
        
    print(f"Loaded {len(protein_ids_from_file)} sequences with embedding shape: {embeddings.shape}")
    
    print("Creating PCA plot...")
    try:
        create_pca_plot(embeddings, protein_ids_from_file, taxa_dict, str(output_dir_plots), gene_name, layer_name_str=layer_identifier)
    except Exception as e:
        print(f"Error creating PCA plot for {gene_name} (layer {layer_identifier}): {e}")
        import traceback
        traceback.print_exc()

    print(f"--- Finished Processing Gene: {gene_name}, Layer: {layer_identifier} ---")

if __name__ == "__main__":
    main()