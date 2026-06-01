import pandas as pd

df = pd.read_csv("data/processed/ants_interactions.csv", low_memory=False)

print(f"Total before filtering: {len(df)}")

# Search terms
ants = "Formicidae"
microorganisms = ["Fungi", "Bacteria"]

# Function to check if taxon_path contains the term
def contains_term(series, terms):
    if isinstance(terms, str):
        terms = [terms]
    return series.str.contains("|".join(terms), na=False, case=False)

# Filter interactions between ants and microorganisms
filter_condition = (
    # Ant as source and microorganism as target
    (contains_term(df["source_taxon_path"], ants) & 
     contains_term(df["target_taxon_path"], microorganisms)) |
    
    # Microorganism as source and ant as target
    (contains_term(df["source_taxon_path"], microorganisms) & 
     contains_term(df["target_taxon_path"], ants))
)

filtered_df = df[filter_condition]

print(f"Total after filtering: {len(filtered_df)}")

print("\nInteraction types found:")
print(filtered_df["interaction_type"].value_counts())

print("\nMost frequent microorganisms as target:")
print(filtered_df["target_taxon_name"].value_counts().head(10))

print("\nMost frequent microorganisms as source:")
print(filtered_df["source_taxon_name"].value_counts().head(10))

filtered_df.to_csv("data/processed/ants_microorganisms.csv", index=False)

print("\nFile saved in data/processed/ants_microorganisms.csv")