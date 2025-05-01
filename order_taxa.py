import pandas as pd
from Bio import SeqIO
import os

def get_fasta_accessions(fasta_file):
    """Extracts the first 13 characters of accession IDs from a FASTA file."""
    accessions = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Assuming the accession is the first part of the ID before any space or pipe
        accession = record.id.split()[0].split('|')[0]
        accessions.append(accession[:13])
    return accessions

def order_taxa(taxa_df, fasta_file):
    """Orders the taxa DataFrame based on the accession order in the FASTA file."""
    # Get the ordered accessions (first 13 chars) from the FASTA file
    ordered_accessions = get_fasta_accessions(fasta_file)

    # Create a mapping from accession (13 chars) to its order
    accession_order_map = {acc: i for i, acc in enumerate(ordered_accessions)}

    # Create a temporary column with the first 13 chars of the 'Acession' column
    taxa_df['temp_accession_13'] = taxa_df['Accession'].astype(str).str[:13]

    # Map the order from the FASTA file to the DataFrame
    # Use .get() with a default value (e.g., infinity) for accessions not in the FASTA
    taxa_df['fasta_order'] = taxa_df['temp_accession_13'].map(accession_order_map)
    taxa_df['fasta_order'] = taxa_df['fasta_order'].fillna(float('inf')) # Handle missing accessions

    # Sort the DataFrame based on the FASTA order
    ordered_df = taxa_df.sort_values('fasta_order').reset_index(drop=True)

    # Remove temporary columns
    ordered_df = ordered_df.drop(columns=['temp_accession_13', 'fasta_order'])

    return ordered_df


if __name__ == "__main__":
    # ...existing code...

    # Define the path to the taxa CSV file and the FASTA file
    # !!! IMPORTANT: Update this path to your actual taxa CSV file !!!
    taxa_file = "/home/s233201/esm_runs/inputs/taxa.csv"
    fasta_file = "/home/s233201/esm_runs/inputs/ACO2.fasta"

    # Load the taxa information
    try:
        taxa_df = pd.read_csv(taxa_file)
        # Verify the 'Acession' column exists
        if 'Accession' not in taxa_df.columns:
            print(f"Error: Column 'Accession' not found in {taxa_file}. Found columns: {taxa_df.columns.tolist()}")
            exit()
    except FileNotFoundError:
        print(f"Error: Taxa file not found at {taxa_file}")
        exit()
    except Exception as e:
        print(f"Error loading taxa file: {e}")
        exit()

    # Get the accessions in order from the FASTA file (optional, already done in order_taxa)
    # accessions = get_fasta_accessions(fasta_file) # This line is not strictly needed anymore

    # Create a DataFrame with the ordered accessions and their corresponding taxa
    try:
        ordered_taxa_df = order_taxa(taxa_df.copy(), fasta_file) # Use a copy to avoid modifying original df
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_file}")
        exit()
    except Exception as e:
        print(f"Error ordering taxa: {e}")
        exit()


    # Save the ordered taxa DataFrame to a new CSV file
    output_dir = os.path.dirname(fasta_file) # Save in the same directory as the fasta file
    output_file = os.path.join(output_dir, "ordered_taxa.csv")
    try:
        ordered_taxa_df.to_csv(output_file, index=False)
        print(f"Ordered taxa saved to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")