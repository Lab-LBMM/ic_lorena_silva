import pandas as pd

df = pd.read_csv("data/raw/interactions.csv", low_memory=False)
print(f"Total: {len(df)}")

has_doi = df["study_url"].str.contains(r"doi\.org|10\.", na=False, case=False)
print(f"Com DOI: {has_doi.sum()}")
print(f"Sem DOI: {(~has_doi).sum()}")

df_doi = df[has_doi]

def contains_term(series, terms):
    if isinstance(terms, str):
        terms = [terms]
    return series.str.contains("|".join(terms), na=False, case=False)

condition = (
    (contains_term(df_doi["source_taxon_path"], "Formicidae") &
     contains_term(df_doi["target_taxon_path"], ["Fungi", "Bacteria"])) |
    (contains_term(df_doi["source_taxon_path"], ["Fungi", "Bacteria"]) &
     contains_term(df_doi["target_taxon_path"], "Formicidae"))
)

df_micro = df_doi[condition]
print(f"Com DOI e formigas x microorganismos: {len(df_micro)}")
