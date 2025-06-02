import os
from ete3 import Tree
import pandas as pd

def get_all_outliers(file_path):
    """
    Reads an outliers file and returns a set of all outliers.
    """
    all_outliers = set()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Outliers for"):
                # Extract the list of outliers
                outliers_str = line.split(": ")[1].strip("[]\n").replace("'", "")
                outliers = outliers_str.split(", ")
                all_outliers.update(outliers)
    return all_outliers

def remove_outliers_from_matrix(matrix_file, outliers):
    """
    Removes specified outliers from a distance matrix file and saves the cleaned matrix to a new directory.
    """
    try:
        phylo_raw = pd.read_csv(matrix_file, header=None, skiprows=1, engine='pyarrow')
        # Split the single column by whitespace
        phylo_split = phylo_raw[0].str.split(expand=True)
        phylo_accessions = phylo_split.iloc[:, 0].values
        phylo = pd.DataFrame(phylo_split.iloc[:, 1:].values, index=phylo_accessions, columns=phylo_accessions)

        # Remove outlier rows and columns
        accessions_to_keep = [acc for acc in phylo_accessions if acc not in outliers]
        phylo = phylo.loc[accessions_to_keep, accessions_to_keep]

        # Save the cleaned matrix to a new file in the 'clean' directory
        clean_dir = os.path.join(os.path.dirname(matrix_file), "clean")
        os.makedirs(clean_dir, exist_ok=True)
        
        clean_matrix_file = os.path.join(clean_dir, os.path.basename(matrix_file))
        
        # Set column names to be the same as index names
        phylo.columns = phylo.index.values
        
        # Set the name of the index to be ''
        phylo.index.name = ''

        phylo.to_csv(clean_matrix_file, header=True, index=True, sep=" ")
        return True
    except Exception as e:
        print(f"Error processing {matrix_file}: {e}")
        return False

def remove_outliers_from_tree(tree_file, outliers):
    """
    Removes specified outliers from a phylogenetic tree using ete3 and saves the cleaned tree.
    """
    try:
        t = Tree(tree_file, format=1)  # Assuming Newick format
        
        # Find and remove outlier nodes
        for outlier in outliers:
            try:
                node_to_remove = t.search_nodes(name=outlier)[0]
                node_to_remove.delete()
            except IndexError:
                print(f"Outlier {outlier} not found in tree.")
        
        # Save the cleaned tree to a new file
        clean_tree_file = tree_file.replace(".treefile", "_clean.treefile")
        t.write(outfile=clean_tree_file, format=1)
        return True
    except Exception as e:
        print(f"Error processing {tree_file}: {e}")
        return False

if __name__ == "__main__":
    outliers_file = "/home/s233201/outliers.txt"
    enzyme_trees_dir = "/home/s233201/enzyme_trees"
    distance_matrices_dir = "/home/s233201/full_dist_mats/fast/"

    outliers = get_all_outliers(outliers_file)
    print(f"Total number of unique outliers: {len(outliers)}")

    # Save the outliers set to a file
    with open("outliers_set.txt", "w") as f:
        for outlier in outliers:
            f.write(outlier + "\n")
    print("Outliers saved to outliers_set.txt")

    # # Process each tree file in the directory
    # for filename in os.listdir(enzyme_trees_dir):
    #     if filename.endswith(".treefile"):  # Assuming tree files are in Newick format
    #         tree_file_path = os.path.join(enzyme_trees_dir, filename)
    #         print(f"Processing {filename}...")
    #         if remove_outliers_from_tree(tree_file_path, outliers):
    #             print(f"Outliers removed from {filename}, saved as *_clean.treefile")
    #         else:
    #             print(f"Failed to remove outliers from {filename}")
    
    # Process each distance matrix file in the directory
    for filename in os.listdir(distance_matrices_dir):
        if filename.startswith("full_mat_") and filename.endswith(".csv"):
            matrix_file_path = os.path.join(distance_matrices_dir, filename)
            print(f"Processing distance matrix {filename}...")
            if remove_outliers_from_matrix(matrix_file_path, outliers):
                print(f"Outliers removed from distance matrix {filename}, saved to full_dist_mats/fast/clean")
            else:
                print(f"Failed to remove outliers from distance matrix {filename}")

    print("Outlier removal complete.")
