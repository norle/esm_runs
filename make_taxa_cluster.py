import pandas as pd

# Load taxa.csv
taxa_df = pd.read_csv('/home/s233201/esm_runs/inputs/taxa.csv')

# Load ACO2.fasta and extract accessions in order
accessions = []
with open('/home/s233201/esm_runs/inputs/ACO2.fasta') as fasta_file:
    for line in fasta_file:
        if line.startswith('>'):
            accession = line[1:14]  # Extract first 13 characters after '>'
            accessions.append(accession)

# Filter taxa_df to keep only rows with accessions in the fasta file and create a copy
filtered_taxa_df = taxa_df[taxa_df['Accession'].str[:13].isin(accessions)].copy()

# Create a mapping series for ordering
order_mapping = pd.Series(range(len(accessions)), index=accessions)

# Sort the dataframe based on the FASTA file order
filtered_taxa_df['sort_key'] = filtered_taxa_df['Accession'].str[:13].map(order_mapping)
filtered_taxa_df = filtered_taxa_df.sort_values('sort_key').drop('sort_key', axis=1)

# Convert Phyla to numeric values
phyla_mapping = {phylum: i for i, phylum in enumerate(filtered_taxa_df['Phylum'].unique())}
filtered_taxa_df['Phylum'] = filtered_taxa_df['Phylum'].map(phyla_mapping)

# Use accession numbers as sequence_idx
filtered_taxa_df['sequence_idx'] = filtered_taxa_df['Accession'].str[:13]

# Keep only the sequence_idx and Phylum (cluster) column
cluster_df = filtered_taxa_df[['sequence_idx', 'Phylum']].rename(columns={'Phylum': 'cluster'})

# Save the cluster assignments to a new CSV file
cluster_df.to_csv('/home/s233201/esm_runs/clusters/taxa_clusters.csv', index=False)