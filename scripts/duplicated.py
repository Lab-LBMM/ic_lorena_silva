import pandas as pd

df = pd.read_csv("data/processed/ants_interactions.csv", low_memory=False)

print(f"Total before: {len(df)}")

df_no_duplicates = df.drop_duplicates(
    subset=["source_taxon_name", "interaction_type", "target_taxon_name"]
)

print(f"Total after: {len(df_no_duplicates)}")

df_no_duplicates.to_csv("data/processed/ants_no_duplicates.csv", index=False)
print("File saved!")