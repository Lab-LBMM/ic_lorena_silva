import pandas as pd

df = pd.read_csv("data/processed/ants_no_duplicates.csv", low_memory=False)

relevant_columns = [
    "source_taxon_name",
    "source_taxon_path",
    "interaction_type",
    "target_taxon_name",
    "target_taxon_path",
    "latitude",
    "longitude",
    "event_date",
    "study_citation",
    "study_url"
]

clean_df = df[relevant_columns]

clean_df.to_csv("data/processed/clean_ants.csv", index=False)

print(clean_df.head())