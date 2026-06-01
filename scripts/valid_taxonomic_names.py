import pandas as pd

df = pd.read_csv("data/processed/clean_ants.csv", low_memory=False)

print(f"Total before: {len(df)}")

valid_df = df.dropna(subset=["source_taxon_name", "target_taxon_name"])

invalid_terms = ["animalia", "plantae", "fungi", "unknown", "no name"]

valid_df = valid_df[
    ~valid_df["source_taxon_name"].str.lower().isin(invalid_terms) &
    ~valid_df["target_taxon_name"].str.lower().isin(invalid_terms)
]

print(f"Total after: {len(valid_df)}")

valid_df.to_csv("data/processed/valid_ants.csv", index=False)