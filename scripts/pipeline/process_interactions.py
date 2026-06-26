import argparse
import os
import pandas as pd

__version__ = "1.0.0"

DEFAULT_COLUMNS = [
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

DEFAULT_INVALID_TERMS = ["animalia", "plantae", "fungi", "unknown", ""]

ANTS = "Formicidae"
MICROORGANISMS = ["Fungi", "Bacteria"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: from raw GloBI interactions to unique ant-microorganism interaction pairs."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/interactions.csv",
        help="Path to the raw input CSV file (default: data/raw/interactions.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/pairs_unique.csv",
        help="Path to the output CSV file with unique pairs (default: data/processed/pairs_unique.csv)"
    )
    parser.add_argument(
        "--columns", "-c",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="List of columns to keep (default: predefined relevant columns)"
    )
    parser.add_argument(
        "--invalid-terms", "-t",
        nargs="+",
        default=DEFAULT_INVALID_TERMS,
        help="List of taxon names to exclude (default: animalia plantae fungi unknown 'no name')"
    )
    parser.add_argument(
        "--ants", "-a",
        default=ANTS,
        help="Taxon name to filter as ants (default: Formicidae)"
    )
    parser.add_argument(
        "--microorganisms", "-m",
        nargs="+",
        default=MICROORGANISMS,
        help="List of taxon names to filter as microorganisms (default: Fungi Bacteria)"
    )
    parser.add_argument(
        "--separator", "-s",
        default=" | ",
        help="Separator used to join the pair fields (default: ' | ')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress and row counts at each step"
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    return parser.parse_args()


def contains_term(series, terms):
    if isinstance(terms, str):
        terms = [terms]
    return series.str.contains("|".join(terms), na=False, case=False)


def preprocess(df, verbose):
    if verbose:
        print(f"[preprocess] Total rows: {len(df)}")
    return df


def select_columns(df, columns, verbose):
    df = df[columns]
    if verbose:
        print(f"[select_columns] Columns kept: {list(df.columns)}")
    return df


def validate_taxa(df, invalid_terms, verbose):
    if verbose:
        print(f"[validate_taxa] Total before: {len(df)}")
    df = df.dropna(subset=["source_taxon_name", "target_taxon_name"])
    df = df[
        ~df["source_taxon_name"].str.lower().isin(invalid_terms) &
        ~df["target_taxon_name"].str.lower().isin(invalid_terms)
    ]
    if verbose:
        print(f"[validate_taxa] Total after: {len(df)}")
    return df


def deduplicate(df, verbose):
    if verbose:
        print(f"[deduplicate] Total before: {len(df)}")
    df = df.drop_duplicates(
        subset=["source_taxon_name", "interaction_type", "target_taxon_name"]
    )
    if verbose:
        print(f"[deduplicate] Total after: {len(df)}")
    return df


def filter_interactions(df, ants, microorganisms, verbose):
    if verbose:
        print(f"[filter_interactions] Total before: {len(df)}")
    condition = (
        (contains_term(df["source_taxon_path"], ants) &
         contains_term(df["target_taxon_path"], microorganisms)) |
        (contains_term(df["source_taxon_path"], microorganisms) &
         contains_term(df["target_taxon_path"], ants))
    )
    df = df[condition]
    if verbose:
        print(f"[filter_interactions] Total after: {len(df)}")
    return df


def extract_pairs(df, separator, verbose):
    pairs = (
        df["source_taxon_name"] + separator +
        df["interaction_type"] + separator +
        df["target_taxon_name"]
    ).drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"[extract_pairs] Total unique pairs: {len(pairs)}")
        print(pairs.head(10))
    return pairs


def main():
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    df = preprocess(df, args.verbose)
    df = select_columns(df, args.columns, args.verbose)
    df = validate_taxa(df, args.invalid_terms, args.verbose)
    df = deduplicate(df, args.verbose)
    df = filter_interactions(df, args.ants, args.microorganisms, args.verbose)
    pairs = extract_pairs(df, args.separator, args.verbose)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(f"\nFile saved in {args.output}")


if __name__ == "__main__":
    main()