import pandas as pd
import torch
import os
import numpy as np
from pathlib import Path
import glob


def load_valid_accessions(csv_path):
    """Load valid accessions from the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'Accession' not in df.columns:
            raise ValueError(f"'Accession' column not found in {csv_path}")
        # Only keep the first 13 characters of each accession
        return set(acc[:13] for acc in df['Accession'].values)
    except Exception as e:
        print(f"Error loading accessions from CSV: {e}")
        return set()


def load_embeddings_accessions(accessions_csv_path):
    """Load accessions that correspond to the embeddings."""
    try:
        df = pd.read_csv(accessions_csv_path)
        if 'Accession' not in df.columns:
            raise ValueError(f"'Accession' column not found in {accessions_csv_path}")
        return df['Accession'].tolist()
    except Exception as e:
        print(f"Error loading embeddings accessions from CSV: {e}")
        return []


def filter_embeddings(embeddings_path, valid_accessions_csv, embeddings_accessions_csv, output_path=None):
    """
    Filter embeddings to keep only those with accessions in valid_accessions.
    
    Args:
        embeddings_path: Path to the embeddings file or directory with .npy files
        valid_accessions_csv: Path to CSV containing valid accessions
        embeddings_accessions_csv: Path to CSV containing accessions corresponding to the embeddings
        output_path: Path to save filtered embeddings (default: original path with '_filtered' suffix)
    
    Returns:
        Tuple of (filtered_embeddings, filtered_accessions, stats)
    """
    # Check if embeddings_path is a directory containing .npy files
    if os.path.isdir(embeddings_path):
        npy_files = glob.glob(os.path.join(embeddings_path, "*.npy"))
        if npy_files:
            print(f"Found {len(npy_files)} .npy files in {embeddings_path}")
            return filter_multiple_npy_files(npy_files, valid_accessions_csv, embeddings_accessions_csv)
    
    # Create a subdirectory for filtered embeddings
    if output_path is None:
        base_dir = os.path.dirname(embeddings_path)
        filtered_dir = os.path.join(base_dir, "filtered_embeddings")
        os.makedirs(filtered_dir, exist_ok=True)
        
        # Use the same filename but in the new directory
        output_path = os.path.join(filtered_dir, os.path.basename(embeddings_path))
    
    # Load valid accessions
    valid_accessions = load_valid_accessions(valid_accessions_csv)
    print(f"Loaded {len(valid_accessions)} valid accessions from {valid_accessions_csv}")
    
    # Load embeddings accessions
    accessions = load_embeddings_accessions(embeddings_accessions_csv)
    print(f"Loaded {len(accessions)} embeddings accessions from {embeddings_accessions_csv}")
    
    # Load embeddings
    try:
        embeddings = torch.load(embeddings_path)
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("Embeddings must be a torch.Tensor")
        if len(embeddings) != len(accessions):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of accessions ({len(accessions)})")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None, {"error": str(e)}
    
    # Filter embeddings using only first 13 chars of accessions
    filtered_indices = [i for i, acc in enumerate(accessions) if acc[:13] in valid_accessions]
    filtered_accessions = [accessions[i] for i in filtered_indices]
    filtered_embeddings = embeddings[filtered_indices] if isinstance(embeddings, torch.Tensor) else [embeddings[i] for i in filtered_indices]
    
    # Save filtered embeddings
    torch.save({
        'accessions': filtered_accessions,
        'embeddings': filtered_embeddings
    }, output_path)
    
    # Stats
    stats = {
        "original_count": len(accessions),
        "filtered_count": len(filtered_accessions),
        "removed_count": len(accessions) - len(filtered_accessions),
        "output_path": output_path
    }
    
    return filtered_embeddings, filtered_accessions, stats


def filter_multiple_npy_files(npy_files, valid_accessions_csv, embeddings_accessions_csv):
    """
    Filter multiple .npy embedding files based on valid accessions.
    
    Args:
        npy_files: List of paths to .npy embedding files
        valid_accessions_csv: Path to CSV containing valid accessions
        embeddings_accessions_csv: Path to CSV containing accessions corresponding to the embeddings
        
    Returns:
        Tuple of (None, None, stats) with stats containing summarized information
    """
    # Load valid accessions
    valid_accessions = load_valid_accessions(valid_accessions_csv)
    print(f"Loaded {len(valid_accessions)} valid accessions from {valid_accessions_csv}")
    
    # Load embeddings accessions
    accessions = load_embeddings_accessions(embeddings_accessions_csv)
    print(f"Loaded {len(accessions)} embeddings accessions from {embeddings_accessions_csv}")
    
    # Filter indices using only first 13 chars of accessions
    filtered_indices = [i for i, acc in enumerate(accessions) if acc[:13] in valid_accessions]
    filtered_accessions = [accessions[i] for i in filtered_indices]
    
    # Create a subdirectory for filtered embeddings
    if npy_files:
        base_dir = os.path.dirname(npy_files[0])
        filtered_dir = os.path.join(base_dir, "filtered_embeddings")
        os.makedirs(filtered_dir, exist_ok=True)
    
    total_processed = 0
    processed_files = []
    
    # Process each .npy file
    for npy_file in npy_files:
        try:
            # Load .npy file
            embeddings = np.load(npy_file)
            
            # Check if array size matches number of indices
            array_size = embeddings.shape[0]
            
            # Adjust filtered indices to only include indices that are within the array bounds
            valid_indices = [i for i in filtered_indices if i < array_size]
            
            if len(valid_indices) != len(filtered_indices):
                print(f"Warning: Some indices were out of bounds for {npy_file}. "
                      f"Using {len(valid_indices)} indices instead of {len(filtered_indices)}.")
            
            # Filter embeddings with valid indices
            filtered_embeddings = embeddings[valid_indices]
            
            # Save to the same filename but in the filtered_embeddings subfolder
            output_path = os.path.join(filtered_dir, os.path.basename(npy_file))
            np.save(output_path, filtered_embeddings)
            
            total_processed += 1
            processed_files.append({
                "input": npy_file,
                "output": output_path,
                "original_shape": embeddings.shape,
                "filtered_shape": filtered_embeddings.shape,
                "valid_indices_count": len(valid_indices)
            })
            
            print(f"Processed {npy_file}: {embeddings.shape} → {filtered_embeddings.shape}")
            
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
    
    # Summarize stats
    stats = {
        "original_count": len(accessions),
        "filtered_count": len(filtered_accessions),
        "removed_count": len(accessions) - len(filtered_accessions),
        "processed_files": processed_files,
        "total_processed": total_processed,
        "output_dir": filtered_dir
    }
    
    return None, None, stats


if __name__ == "__main__":
    # Example usage with direct string inputs
    embeddings_path = "/home/s233201/esm_runs/embeddings"  # Directory with multiple .npy files
    valid_accessions_csv = "/home/s233201/esm_runs/data/taxa_clean_0424.csv"
    embeddings_accessions_csv = "/home/s233201/esm_runs/inputs/taxa.csv"
    
    # Filter embeddings
    _, _, stats = filter_embeddings(embeddings_path, valid_accessions_csv, embeddings_accessions_csv)
    
    # Print stats
    print(f"Original embeddings count: {stats['original_count']}")
    print(f"Filtered embeddings count: {stats['filtered_count']}")
    print(f"Removed {stats['removed_count']} embeddings")
    print(f"Processed {stats['total_processed']} .npy files")
