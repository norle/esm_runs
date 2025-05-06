import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from Bio import SeqIO
import multiprocessing
# Import hierarchy for the to_tree function
from scipy.cluster import hierarchy # Changed this for clarity
from scipy.spatial.distance import squareform

# --- Custom Newick Function (from your Stack Overflow find) ---
def get_newick(node, parent_dist, leaf_names, newick='') -> str:
    """
    Convert sciply.cluster.hierarchy.to_tree()-output to Newick format.

    :param node: output of sciply.cluster.hierarchy.to_tree() (a ClusterNode object)
    :param parent_dist: distance of the parent of this node
    :param leaf_names: list of leaf names
    :param newick: leave empty, this variable is used in recursion.
    :returns: tree in Newick format
    """
    if node.is_leaf():
        # For leaves, the branch length is parent_dist - node.dist
        # However, SciPy's to_tree often sets node.dist to 0 for leaves when Z is from linkage.
        # The crucial part is the distance stored in Z itself, which linkage uses to build the tree.
        # The to_tree function when rd=False uses the distance values directly from Z.
        # The branch length for a leaf is the distance at which it merged with its sibling/parent.
        # If node.dist is 0 for leaves, then parent_dist is the correct branch length.
        return "%s:%.6f%s" % (leaf_names[node.id], parent_dist, newick) # Using parent_dist as branch length
    else:
        # For internal nodes, the branch length is parent_dist - node.dist
        branch_len = parent_dist - node.dist
        if len(newick) > 0:
            newick = "):%.6f%s" % (branch_len, newick)
        else:
            # This is the root of the tree, so it ends with a semicolon
            newick = ");" # Root doesn't have a branch length leading to it in this representation

        # Recursively build the Newick string for the left child
        newick = get_newick(node.get_left(), node.dist, leaf_names, newick=newick)
        # Recursively build the Newick string for the right child, prepending a comma
        newick = get_newick(node.get_right(), node.dist, leaf_names, newick=",%s" % (newick))
        # Enclose the children in parentheses
        newick = "(%s" % (newick)
        return newick

# --- Helper Functions ---

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    if not os.path.exists(embedding_file):
        print(f"Warning: Embedding file not found: {embedding_file}")
        return None
    embeddings = np.load(embedding_file)
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

# --- Tree Generation Function ---

def generate_and_save_tree(gene_name):
    """
    Process single gene: load embeddings, get FASTA accessions,
    perform hierarchical clustering, and save the tree in Newick format.
    Returns a tuple (gene_name, success_status_boolean).
    """
    print(f"Processing {gene_name} for tree generation...")
    embedding_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta'
    tree_output_dir = '/home/s233201/esm_runs/trees'

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

    if embeddings.shape[0] < 2:
        print(f"Warning: Not enough embeddings ({embeddings.shape[0]}) for clustering for {gene_name}. Need at least 2. Skipping tree.")
        return gene_name, False

    print(f"Calculating distance matrix for {gene_name} with {embeddings.shape[0]} points...")
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.maximum(distance_matrix, 0)

    print(f"Performing hierarchical clustering for {gene_name}...")
    try:
        condensed_dist_matrix = squareform(distance_matrix, checks=False)
        
        # Perform linkage (e.g., UPGMA/average linkage)
        Z = hierarchy.linkage(condensed_dist_matrix, method='average')
        
        # Convert linkage matrix to a ClusterNode tree structure
        # rd=False: The distances in Z are the heights of the merges.
        # If rd=True, it would try to "reduce" distances, which is not what we want here.
        tree_root_node = hierarchy.to_tree(Z, rd=False)
        
        # Use the custom get_newick function
        # The initial parent_dist for the root's "parent" can be considered its own distance,
        # or for a slightly more standard Newick, the root itself won't have an outgoing branch length.
        # The get_newick function handles the root case by just adding ');'.
        # The second argument to get_newick is the 'parent_dist' for the current node.
        # For the root, its 'parent_dist' passed to the function will be used to calculate its children's branch lengths.
        # tree_root_node.dist is the height of the root node itself.
        tree_string = get_newick(tree_root_node, tree_root_node.dist, fasta_accessions)
        
        tree_file_path = os.path.join(tree_output_dir, f'{gene_name.lower()}.tree')
        with open(tree_file_path, 'w') as f:
            f.write(tree_string) # The function should already add the trailing semicolon.
        print(f"Saved Newick tree for {gene_name} to {tree_file_path}")
        return gene_name, True

    except Exception as e:
        print(f"Error during hierarchical clustering or tree saving for {gene_name}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return gene_name, False

if __name__ == "__main__":
    gene_names = ["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"]
    num_cores = multiprocessing.cpu_count()
    print(f"Starting tree generation for {len(gene_names)} gene(s) using {num_cores} core(s).")

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

    print(f"\nScript finished. Check the '/home/s233201/esm_runs/trees' directory for the .tree files.")