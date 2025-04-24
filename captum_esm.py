import os
import random
import numpy as np
import torch
from Bio import SeqIO
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NeuronConductance
import matplotlib.pyplot as plt
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ESMProteinTensor, LogitsConfig
from esm.utils.encoding import tokenize_sequence
from esm.tokenization import EsmSequenceTokenizer
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cosine


def load_random_sequences(fasta_file, num_sequences=2):
    """Load random sequences from a FASTA file."""
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    random_sequences = random.sample(sequences, num_sequences)
    return [(seq.id, str(seq.seq)) for seq in random_sequences]


def load_different_sequences(fasta_file, embeddings_file, num_sequences=10):
    """Load the most different sequences based on embedding cosine distances."""
    # Load embeddings
    print(f"Loading embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings with shape {embeddings.shape}")
    
    # Load all sequences
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Filter sequences by length
    seq_lengths = np.array([len(seq.seq) for seq in sequences])
    q25, q75 = np.percentile(seq_lengths, [25, 75])
    print(f"Sequence length statistics:")
    print(f"25th percentile: {q25}, 75th percentile: {q75}")
    print(f"Original sequence count: {len(sequences)}")
    
    # Keep only sequences within 25-75 percentile length range
    sequences = [seq for seq in sequences if q25 <= len(seq.seq) <= q75]
    print(f"Filtered sequence count: {len(sequences)}")
    
    if len(sequences) < num_sequences:
        print(f"Warning: Only {len(sequences)} sequences available after length filtering")
        num_sequences = len(sequences)
    
    # Update embeddings to match filtered sequences
    valid_indices = [i for i, seq in enumerate(sequences) if q25 <= len(seq.seq) <= q75]
    embeddings = embeddings[valid_indices]
    
    # Calculate cosine distances between all pairs of embeddings
    print("Calculating cosine distances...")
    distances = squareform(pdist(embeddings, metric='cosine'))
    
    if len(sequences) != embeddings.shape[0]:
        print(f"Warning: Number of sequences ({len(sequences)}) doesn't match embeddings shape ({embeddings.shape[0]})")
        sequences = sequences[:embeddings.shape[0]]  # Truncate if needed
    
    # Select the most different sequences from the first one
    reference_idx = 0
    distances_from_ref = distances[reference_idx]
    
    # Get indices of the most different sequences (highest cosine distances)
    most_different_indices = np.argsort(distances_from_ref)[-num_sequences-1:]
    
    # Make sure the reference sequence is included
    if reference_idx not in most_different_indices:
        most_different_indices = np.append([reference_idx], most_different_indices[:-1])
    
    # Get the corresponding sequences
    selected_sequences = [(sequences[i].id, str(sequences[i].seq)) for i in most_different_indices]
    
    # Print selected sequence ids and their distances
    print("Selected sequences (id, distance from reference):")
    for idx in most_different_indices:
        print(f"  {sequences[idx].id}: {distances_from_ref[idx]:.4f}")
    
    return selected_sequences


def compute_attributions(client, sequence1, sequence2, device):
    """Compute attributions using NeuronConductance on the last embedding layer."""
    tokenizer = EsmSequenceTokenizer()
    
    # Tokenize sequences using the ESM API
    protein1 = ESMProtein(sequence=sequence1)
    protein1_tensor = client.encode(protein1)
    protein2 = ESMProtein(sequence=sequence2)
    protein2_tensor = client.encode(protein2)
    
    # Extract the raw input tensors that we can use with Captum
    print(f"Protein1 tensor type: {type(protein1_tensor)}")
    print(f"Protein1 tensor attributes: {dir(protein1_tensor)}")
    
    # Extract tokens with proper shape checking
    if hasattr(protein1_tensor, 'sequence') and protein1_tensor.sequence is not None:
        tokens1 = protein1_tensor.sequence
        if isinstance(tokens1, torch.Tensor):
            print(f"Token shape from tensor: {tokens1.shape}")
        else:
            print(f"Sequence is not a tensor, converting...")
            # Convert to tensor if needed
            tokens1 = torch.tensor(protein1_tensor.sequence, 
                                  device=device, 
                                  dtype=torch.long).unsqueeze(0)
    else:
        print("Tokenizing sequence manually...")
        tokens1 = torch.tensor([tokenizer.encode(sequence1)], 
                              device=device, 
                              dtype=torch.long)
    
    print(f"Final tokens1 shape: {tokens1.shape}")
    
    # Create a wrapper model to expose the embeddings and compute cosine similarity
    class EmbeddingAnalysisModel(torch.nn.Module):
        def __init__(self, client_model):
            super().__init__()
            self.client = client_model
            
        def forward(self, tokens):
            # Create attention mask
            attention_mask = (tokens != self.client.tokenizer.pad_token_id)
            
            # Forward pass through transformer model
            output = self.client.forward(sequence_tokens=tokens, sequence_id=attention_mask)
            
            # Return the embeddings directly (these are the last hidden states)
            return output.embeddings
    
    # Initialize our wrapper model
    wrapper_model = EmbeddingAnalysisModel(client).to(device)
    
    # Pre-compute the target embedding from sequence 2
    with torch.no_grad():
        # Get embeddings for sequence 2
        outputs2 = client.forward(sequence_tokens=protein2_tensor.sequence, 
                                sequence_id=(protein2_tensor.sequence != client.tokenizer.pad_token_id))
        embeddings2 = outputs2.embeddings
        mean_embedding2 = embeddings2.mean(dim=1)  # [1, D]
    
    # Function to determine cosine similarity with target embedding
    def cosine_sim_func(embeddings):
        # Mean pooling across sequence length
        seq_mean = embeddings.mean(dim=1)
        # Cosine similarity with target embedding
        return torch.nn.functional.cosine_similarity(seq_mean, mean_embedding2)
    
    try:
        # Initialize NeuronConductance with our model
        neuron_cond = NeuronConductance(wrapper_model)
        
        # Baseline is zeros (common choice for token-based models)
        baseline = torch.zeros_like(tokens1)
        
        # Compute conductance for each position in the sequence
        print(f"Computing neuron conductance on input of shape {tokens1.shape}")
        attributions = neuron_cond.attribute(
            tokens1,
            baselines=baseline,
            target=cosine_sim_func,
            n_steps=50
        )
        
        # Extract per-position importance (sum across embedding dimension)
        position_importance = attributions.sum(dim=2).squeeze(0)
        
        # Convert to numpy array and take absolute values
        attributions_np = torch.abs(position_importance).cpu().detach().numpy()
    
    except Exception as e:
        print(f"Error during attribution calculation: {str(e)}")
        print("Falling back to simple embedding-based importance calculation...")
        
        # Fallback: compute a simpler attribution using embedding similarities
        with torch.no_grad():
            outputs1 = client.forward(sequence_tokens=tokens1, 
                                    sequence_id=(tokens1 != client.tokenizer.pad_token_id))
            embeddings1 = outputs1.embeddings.squeeze(0)  # [L, D]
            
            # Calculate similarity of each position with mean target embedding
            similarities = []
            for i in range(embeddings1.shape[0]):
                pos_embed = embeddings1[i:i+1]  # [1, D]
                sim = torch.nn.functional.cosine_similarity(pos_embed, mean_embedding2)
                similarities.append(sim.item())
            
            # Convert to numpy array
            attributions_np = np.array(similarities)
            # Take absolute values to focus on magnitude of impact
            attributions_np = np.abs(attributions_np)
    
    # Normalize attributions to [0,1] range for easier interpretation
    if attributions_np.size > 0:  # Check that we have non-empty attributions
        attributions_norm = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min() + 1e-9)
    else:
        # Fallback if something went wrong
        print("Warning: Empty attributions array. Using dummy values.")
        attributions_norm = np.zeros(len(sequence1))
    
    print(f"Attribution shape: {attributions_norm.shape}")
    
    return attributions_norm


def compute_multiple_attributions(client, sequences, device, num_comparisons=10):
    """Compute attributions for multiple sequence comparisons."""
    attributions_list = []
    sequence_pairs = []
    
    # Get a reference sequence (first one)
    ref_seq_id, ref_seq = sequences[0]
    print(f"Using reference sequence: {ref_seq_id}")
    
    # Compare with other sequences
    for i, (seq_id, seq) in enumerate(sequences[1:num_comparisons+1]):
        print(f"Comparing with sequence {i+1}/{min(num_comparisons, len(sequences)-1)}: {seq_id}")
        attributions = compute_attributions(client, ref_seq, seq, device)
        attributions_list.append(attributions)
        sequence_pairs.append((ref_seq_id, seq_id))
        
    return attributions_list, sequence_pairs, ref_seq


def plot_multiple_attributions(ref_sequence, attributions_list, sequence_pairs):
    """Plot attributions for multiple sequence comparisons."""
    # Create a figure for mean attributions across all sequences
    plt.figure(figsize=(15, 10))
    
    # Make sure attributions match sequence length
    seq_len = min(len(ref_sequence), min([attr.shape[0] for attr in attributions_list]))
    ref_sequence_plot = ref_sequence[:seq_len]
    
    # Create a grid of subplots
    num_plots = len(attributions_list)
    rows = (num_plots + 1) // 2  # Ceiling division
    cols = min(2, num_plots)
    
    # Plot each comparison
    for i, (attributions, (ref_id, comp_id)) in enumerate(zip(attributions_list, sequence_pairs)):
        # Get attributions for this comparison
        attr_plot = attributions[:seq_len]
        
        # Create subplot
        ax = plt.subplot(rows, cols, i+1)
        ax.bar(range(len(ref_sequence_plot)), attr_plot, alpha=0.7)
        ax.set_xticks(range(len(ref_sequence_plot)))
        ax.set_xticklabels(ref_sequence_plot, fontsize=6, rotation=90)
        ax.set_title(f"{ref_id} vs {comp_id}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Attribution")
    
    plt.tight_layout()
    plt.savefig("esm_runs/plots/multiple_attributions.png", dpi=300)
    plt.show()
    
    # Create a heatmap with all comparisons
    plt.figure(figsize=(20, 8))
    
    # Stack all attributions into a 2D array
    stacked_attr = np.vstack([attr[:seq_len] for attr in attributions_list])
    
    im = plt.imshow(stacked_attr, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Attribution Magnitude')
    
    # Set labels
    plt.xlabel("Sequence Position")
    plt.ylabel("Comparison #")
    
    # Set x and y axis labels
    plt.xticks(range(len(ref_sequence_plot)), ref_sequence_plot, fontsize=8, rotation=90)
    plt.yticks(range(len(sequence_pairs)), [f"{ref_id[:10]}...vs {comp_id[:10]}..." for ref_id, comp_id in sequence_pairs])
    
    plt.title("Attribution Heatmap Across Multiple Comparisons")
    plt.tight_layout()
    plt.savefig("esm_runs/plots/attribution_heatmap.png", dpi=300)
    plt.show()
    
    # Compute and plot the mean attribution across all comparisons
    mean_attr = np.mean(stacked_attr, axis=0)
    plt.figure(figsize=(15, 4))
    plt.bar(range(len(ref_sequence_plot)), mean_attr, alpha=0.7)
    plt.xticks(range(len(ref_sequence_plot)), ref_sequence_plot, fontsize=8, rotation=90)
    plt.xlabel("Position")
    plt.ylabel("Mean Attribution")
    plt.title("Mean Attribution Across All Comparisons")
    plt.tight_layout()
    plt.savefig("esm_runs/plots/mean_attribution_all.png", dpi=300)
    plt.show()


def main():
    gene_name = 'ACO2'
    fasta_file = f'/home/s233201/esm_runs/inputs/{gene_name}.fasta'
    embeddings_file = f'/home/s233201/esm_runs/embeddings/{gene_name.lower()}.npy'
    model_name = 'esmc_600m'
    num_comparisons = 10  # Number of sequence comparisons to make

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA is not available, using CPU instead")
    else:
        torch.cuda.set_device(1)
        print(f"Using CUDA device 1: {torch.cuda.get_device_name(1)}")

    # Load the ESM model
    print(f"Loading model {model_name}...")
    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()  # Set to evaluation mode
    client_raw = client.raw_model
    print(f"Model type: {type(client_raw)}")

    # Create directory for plots if it doesn't exist
    os.makedirs("esm_runs/plots", exist_ok=True)

    # Load the most different sequences based on embeddings
    print(f"Loading sequences from {fasta_file} based on embeddings...")
    sequences = load_different_sequences(fasta_file, embeddings_file, num_sequences=num_comparisons)
    print(f"Loaded {len(sequences)} sequences for comparison")

    # Compute attributions for multiple comparisons
    print("Computing attributions for multiple comparisons...")
    attributions_list, sequence_pairs, ref_seq = compute_multiple_attributions(
        client, sequences, device, num_comparisons
    )

    # Plot attributions
    print("Plotting attributions...")
    plot_multiple_attributions(ref_seq, attributions_list, sequence_pairs)


if __name__ == "__main__":
    main()