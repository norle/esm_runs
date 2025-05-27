import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from Bio import SeqIO
import multiprocessing
from sklearn.cluster import HDBSCAN # Added scikit-learn HDBSCAN import
from scipy.cluster import hierarchy # Added import

# --- Helper Functions ---

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    if not os.path.exists(embedding_file):
        print(f"Warning: Embedding file not found: {embedding_file}")
        return None
    embeddings = np.load(embedding_file)
    
    # Check if embeddings are 3D and squeeze if one dimension is 1
    if embeddings.ndim == 3:
        if embeddings.shape[0] == 1:
            embeddings = embeddings.squeeze(0)
            print(f"Reshaped embeddings from 3D to 2D by squeezing the first dimension.")
        elif embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
            print(f"Reshaped embeddings from 3D to 2D by squeezing the second dimension.")
        elif embeddings.shape[2] == 1:
            embeddings = embeddings.squeeze(2)
            print(f"Reshaped embeddings from 3D to 2D by squeezing the third dimension.")
        else:
            print(f"Warning: Embeddings are 3D but no singleton dimension found to squeeze: {embeddings.shape}. Expecting 2D array.")
            # Depending on the expected structure, you might want to return None or raise an error
            # For now, let it proceed, cosine_similarity will likely fail if it's not (n_samples, n_features)

    print(f"Loaded embedding array shape: {embeddings.shape} for {os.path.basename(embedding_file)}")
    return embeddings

def get_fasta_accessions(fasta_file):
    """Get accessions in order from FASTA file. These will be used as tree leaf labels."""
    if not os.path.exists(fasta_file):
        print(f"Warning: FASTA file not found: {fasta_file}")
        return []
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):

        accessions.append(record.id)
    return accessions

# --- Newick Conversion Function (from Stack Overflow example) ---
def get_newick(node, parent_dist, leaf_names, newick='') -> str:
    """
    Convert sciply.cluster.hierarchy.to_tree()-output to Newick format.

    :param node: output of sciply.cluster.hierarchy.to_tree()
    :param parent_dist: output of sciply.cluster.hierarchy.to_tree().dist
    :param leaf_names: list of leaf names
    :param newick: leave empty, this variable is used in recursion.
    :returns: tree in Newick format
    """
    if node.is_leaf():
        # Ensure node.id is within bounds of leaf_names
        if node.id < len(leaf_names):
            return "%s:%.2f%s" % (leaf_names[node.id], parent_dist - node.dist, newick)
        else:
            # Fallback if node.id is out of bounds (should not happen with correct inputs)
            return "leaf_ID_error:%.2f%s" % (parent_dist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parent_dist - node.dist, newick)
        else:
            newick = ");"
        # Ensure get_left() and get_right() are valid before calling
        if node.get_left() is not None:
            newick = get_newick(node.get_left(), node.dist, leaf_names, newick=newick)
        if node.get_right() is not None:
            # Add comma only if left child processing added something
            if node.get_left() is not None :
                 newick = get_newick(node.get_right(), node.dist, leaf_names, newick=",%s" % (newick))
            else: # If no left child, don't prepend comma
                 newick = get_newick(node.get_right(), node.dist, leaf_names, newick=newick)
        newick = "(%s" % (newick)
        return newick

# --- Tree Generation Function ---

def generate_and_save_tree(gene_name):
    """
    Process single gene: load embeddings, get FASTA accessions,
    perform HDBSCAN clustering, and save the tree in Newick format.
    Returns a tuple (gene_name, success_status_boolean).
    """
    print(f"Processing {gene_name} for tree generation...")
    embedding_file = f'/home/s233201/esm_runs/embeddings_new_12th/{gene_name.upper()}_embeddings.npy'
    fasta_file = f'/home/s233201/esm_runs/inputs_new/{gene_name}.fasta'
    tree_output_dir = '/home/s233201/esm_runs/trees_12_hdb' # Consistent output directory

    os.makedirs(tree_output_dir, exist_ok=True)

    embeddings = load_embeddings(embedding_file)
    if embeddings is None:
        print(f"Skipping tree for {gene_name} due to missing embeddings.")
        return gene_name, False

    fasta_accessions = get_fasta_accessions(fasta_file)
    if not fasta_accessions:
         print(f"Warning: FASTA file for {gene_name} ('{fasta_file}') missing, empty, or unreadable. Skipping tree.")
         return gene_name, False

    if len(fasta_accessions) != embeddings.shape[0]:
         print(f"Warning: Mismatch between number of sequences in FASTA ({len(fasta_accessions)}) "
               f"and embeddings ({embeddings.shape[0]}) for {gene_name}. Skipping tree.")
         return gene_name, False

    if embeddings.shape[0] < 2: # Need at least 2 samples for linkage tree
        print(f"Warning: Not enough embeddings ({embeddings.shape[0]}) for clustering for {gene_name}. Need at least 2. Skipping tree.")
        return gene_name, False

    print(f"Calculating distance matrix for {gene_name} with {embeddings.shape[0]} points...")
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.maximum(distance_matrix, 0) # Ensure non-negative distances
    
    # Ensure distance_matrix is of type float64 for HDBSCAN
    distance_matrix = distance_matrix.astype(np.float64)

    print(f"Performing HDBSCAN clustering for {gene_name}...")
    tree_string = None # Initialize tree_string
    try:
        # Initialize sklearn.cluster.HDBSCAN
        clusterer = HDBSCAN(metric='precomputed', 
                            min_cluster_size=2,
                            allow_single_cluster=True,
                            store_single_linkage_tree=True) # Ensure the tree is stored

        # Fit HDBSCAN to the precomputed distance matrix
        clusterer.fit(distance_matrix)
        
        # Get the Newick tree string using single_linkage_tree_
        if hasattr(clusterer, 'single_linkage_tree_') and clusterer.single_linkage_tree_ is not None:
            linkage_matrix = clusterer.single_linkage_tree_
            
            # A valid linkage matrix for n samples has (n-1) rows.
            # HDBSCAN from sklearn might produce fewer rows if not all points are included in the main hierarchy
            # or if min_cluster_size affects the tree structure significantly.
            # We proceed if linkage_matrix has at least one row.
            if linkage_matrix.shape[0] > 0 : 
                try:
                    # rd=False as distances in single_linkage_tree_ are typically the ones to be used.
                    scipy_root_node = hierarchy.to_tree(linkage_matrix, rd=False)
                    if scipy_root_node is not None:
                        tree_string = get_newick(scipy_root_node, scipy_root_node.dist, fasta_accessions)
                    else:
                        # This case might not be hit if to_tree raises error instead of returning None
                        print(f"Warning: hierarchy.to_tree returned None for {gene_name}.")
                except Exception as e:
                    print(f"Error converting linkage matrix to tree for {gene_name}: {e}.")
            else:
                print(f"Warning: Invalid or empty linkage matrix from HDBSCAN for {gene_name}. Shape: {linkage_matrix.shape}.")
        else:
            print(f"Warning: HDBSCAN (sklearn) did not produce a single_linkage_tree_ for {gene_name}.")

        # Fallback if tree_string could not be generated from HDBSCAN
        if tree_string is None:
            print(f"Falling back to simple tree generation for {gene_name}.")
            if embeddings.shape[0] > 0 and len(fasta_accessions) == embeddings.shape[0]:
                if embeddings.shape[0] == 1: # Should be caught by earlier check
                     tree_string = f"({fasta_accessions[0]}:0.0);"
                else: # Multiple items, but HDBSCAN might not have formed a detailed tree
                     leaf_strings = [f"{name}:0.1" for name in fasta_accessions] # Arbitrary small branch length
                     tree_string = f"({','.join(leaf_strings)});"
                print(f"Warning: HDBSCAN did not produce a usable tree for {gene_name}. Generated a simple tree.")
            else:
                # This else corresponds to the inner if, if embeddings/fasta_accessions mismatch after trying to make simple tree
                print(f"Error: Fallback tree generation failed for {gene_name} due to data inconsistency.")
                return gene_name, False
        
        if tree_string is None: # Should not happen if fallback works, but as a safeguard
            print(f"Error: Tree string is still None after fallback for {gene_name}.")
            return gene_name, False

        tree_file_path = os.path.join(tree_output_dir, f'{gene_name.lower()}.tree')
        with open(tree_file_path, 'w') as f:
            f.write(tree_string) 
        print(f"Saved Newick tree for {gene_name} to {tree_file_path}")
        return gene_name, True

    except Exception as e:
        print(f"Error during HDBSCAN clustering or tree saving for {gene_name}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return gene_name, False

if __name__ == "__main__":
    gene_names = ["LYS20","LYS1"]
    num_cores = multiprocessing.cpu_count()
    print(f"Starting tree generation for {len(gene_names)} gene(s) using {num_cores} core(s).")

    # Define tree_output_dir here for the final message
    tree_output_dir = '/home/s233201/esm_runs/trees_12_hdb' 

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(generate_and_save_tree, gene_names)

    successful_genes = []
    failed_genes = []
    for gene, success_status in results:
        if success_status:
            successful_genes.append(gene)
        else:
            failed_genes.append(gene)

    print("\n--- Tree Generation Summary ---")
    if successful_genes:
        print(f"Successfully generated trees for ({len(successful_genes)} genes): {', '.join(successful_genes)}")
    if failed_genes:
        print(f"Failed or skipped tree generation for ({len(failed_genes)} genes): {', '.join(failed_genes)}")
    if not results:
        print("No genes were processed.")

    print(f"\nScript finished. Check the '{tree_output_dir}' directory for the .tree files.") # Use variable for output dir