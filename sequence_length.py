import os
import matplotlib.pyplot as plt
from Bio import SeqIO
from pathlib import Path

# Create output directory
output_dir = Path("/home/s233201/esm_runs/plots/lengths")
output_dir.mkdir(parents=True, exist_ok=True)

# Get all FASTA files
input_dir = Path("/home/s233201/esm_runs/inputs_new")
fasta_files = list(input_dir.glob("*.fasta"))

# Create a 4x2 grid for subplots
num_files = len(fasta_files)
rows, cols = 4, 2
fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
axes = axes.flatten()

for i, fasta_file in enumerate(fasta_files):
    # Get sequence lengths
    lengths = [len(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    
    # Create histogram in the corresponding subplot
    ax = axes[i]
    ax.hist(lengths, bins=50, edgecolor='black')
    #ax.set_yscale('log')  # Set y-axis to log scale
    ax.set_title(f"{fasta_file.stem}")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Frequency (Log Scale)")

# Hide unused subplots if there are fewer files than subplots
for j in range(len(fasta_files), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and save the combined plot
plt.tight_layout()
output_path = output_dir / "combined_length_distributions.png"
plt.savefig(output_path)
plt.close()
