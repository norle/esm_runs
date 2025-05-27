import os
import torch
import argparse
import numpy as np
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from pathlib import Path
from tqdm import tqdm


def load_sequences_from_fasta(fasta_file_path):
    """Load protein sequences from a FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def get_mean_embedding(client, sequence, device):
    """Get mean embedding for a protein sequence."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, 
        LogitsConfig(return_hidden_states=True, ith_hidden_layer=12)
    )
    # Get embeddings and compute mean across sequence length
    embeddings = logits_output.hidden_states  # Shape: (1, seq_len, embed_dim)
    embeddings = embeddings.squeeze(0)  # Remove the 0th dimension
    #print(f"Embeddings shape: {embeddings.shape}")
    mean_embedding = embeddings.mean(dim=1)  # Average across sequence length
    #print(f"Mean embedding shape: {mean_embedding.shape}")
    return mean_embedding


def main():

    model_name = 'esmc_600m'
    fasta_path = '/home/s233201/esm_runs/inputs_new/LYS1.fasta'
    # Define separate output paths for embeddings and IDs
    output_dir = Path('/home/s233201/esm_runs/embeddings_new_12th')
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    base_name = Path(fasta_path).stem # e.g., 'ARO8'
    embeddings_output_path = output_dir / f"{base_name}_embeddings.npy"
    ids_output_path = output_dir / f"{base_name}_ids.txt"


    # Check if CUDA is available and set to GPU 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA is not available, using CPU instead")
    else:
        torch.cuda.set_device(1)  # Use the second GPU
        print(f"Using CUDA device 1: {torch.cuda.get_device_name(1)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(1).total_memory / 1024**2:.0f}MB")

    # Load the ESM model
    print(f"Loading model {model_name}...")
    client = ESMC.from_pretrained(model_name).to(device)
    
    # Load sequences from FASTA file
    print(f"Loading sequences from {fasta_path}...")
    sequences = load_sequences_from_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Process each sequence
    embeddings_list = []
    ids_list = [] # List to store protein IDs
    for i, (protein_id, sequence) in tqdm(enumerate(sequences.items())):
        #print(f"Processing protein {i+1}/{len(sequences)}: {protein_id}")

        mean_embedding = get_mean_embedding(client, sequence, device)
        embeddings_list.append(mean_embedding.float().cpu().detach().numpy())
        ids_list.append(protein_id) # Store the ID
        #print(f"Successfully processed {protein_id}, embedding shape: {mean_embedding.shape}")


    # Convert list to numpy array and save embeddings
    embeddings_array = np.array(embeddings_list)
    print(f"Saving embeddings array with shape {embeddings_array.shape} to {embeddings_output_path}...")
    np.save(embeddings_output_path, embeddings_array)

    # Save IDs to a text file
    print(f"Saving {len(ids_list)} protein IDs to {ids_output_path}...")
    with open(ids_output_path, 'w') as f:
        for protein_id in ids_list:
            f.write(f"{protein_id}\n")

    print("Done!")


if __name__ == "__main__":
    main()
