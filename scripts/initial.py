import pandas as pd

df = pd.read_csv("data/raw/interactions.csv", low_memory=False)

print(f"Total rows: {len(df)}")

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nInteraction types found:")
print(df["interaction_type"].value_counts())

df.to_csv("data/processed/ants_interactions.csv", index=False)

print("\nFile saved in data/processed/ants_interactions.csv")