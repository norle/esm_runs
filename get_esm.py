import os
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("CUDA is not available, using CPU instead")
else:
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

protein = ESMProtein(sequence="AAAAA")
# First load the model normally
client = ESMC.from_pretrained("esmc_600m").to(device)
# Then disable flash attention

protein_tensor = client.encode(protein)
logits_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
print(logits_output.logits, logits_output.embeddings)