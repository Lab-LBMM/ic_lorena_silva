import pandas as pd

df = pd.read_csv("data/processed/valid_ants.csv", low_memory=False)

df["pair_interaction"] = (
    df["source_taxon_name"] + " | " +
    df["interaction_type"] + " | " +
    df["target_taxon_name"]
)

pairs_unique = df["pair_interaction"].drop_duplicates().reset_index(drop=True)

pairs_unique.to_csv("data/processed/pairs_unique.csv", index=False)
print(f"Total unique pairs: {len(pairs_unique)}")
print(pairs_unique.head(10))