#!/usr/bin/env python3

import argparse
import pandas as pd

__version__ = "1.1.0"

DEFAULT_COLUMNS = [
    "source_taxon_name",
    "source_taxon_path",
    "interaction_type",
    "target_taxon_name",
    "target_taxon_path",
    "study_citation",
    "study_url",
    "study_source_archive_uri"
]

ORIGIN_COLUMNS = [
    "source_taxon_name",
    "interaction_type",
    "target_taxon_name",
    "study_citation",
    "study_url",
    "study_source_archive_uri"
]

DEFAULT_INVALID_TERMS = ["animalia", "plantae", "fungi", "unknown", "no name"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for filtering, cleaning, and extracting unique Subject-Relation-Object pairs from interaction datasets."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the raw input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output CSV file with unique pairs"
    )
    parser.add_argument(
        "--columns", "-c",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="Columns to keep from the raw dataset (default: 10 predefined columns)"
    )
    parser.add_argument(
        "--invalid-terms", "-t",
        nargs="+",
        default=DEFAULT_INVALID_TERMS,
        help="Taxon names to exclude (default: animalia plantae fungi unknown 'no name')"
    )
    parser.add_argument(
        "--focal-taxon", "-a",
        default="Formicidae",
        help="Focal taxon to filter (default: Formicidae)"
    )
    parser.add_argument(
        "--interacting-taxa", "-m",
        nargs="+",
        default=["Fungi", "Bacteria"],
        help="Taxa interacting with the focal taxon (default: Fungi Bacteria)"
    )
    parser.add_argument(
        "--separator", "-s",
        default=" , ",
        help="Separator used to join the SRO fields (default: ' , ')"
    )
    parser.add_argument(
    "--output-origin",
    default=None,
    help="Optional path to save the dataset with the origin of data before pair extraction"
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


def select_columns(df, columns, verbose):
    df = df[columns]
    if verbose:
        print(f"[select_columns] Columns kept: {list(df.columns)}")
    return df


def validate_taxon(df, invalid_terms, verbose):
    if verbose:
        print(f"[validate_taxon] Total before: {len(df)}")
    df = df.dropna(subset=["source_taxon_name", "target_taxon_name"])
    df = df[
        ~df["source_taxon_name"].str.lower().isin(invalid_terms) &
        ~df["target_taxon_name"].str.lower().isin(invalid_terms)
    ]
    if verbose:
        print(f"[validate_taxon] Total after: {len(df)}")
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


def filter_interactions(df, focal_taxon, interacting_taxa, verbose):
    if verbose:
        print(f"[filter_interactions] Total before: {len(df)}")
    condition = (
        (contains_term(df["source_taxon_path"], focal_taxon) &
         contains_term(df["target_taxon_path"], interacting_taxa)) |
        (contains_term(df["source_taxon_path"], interacting_taxa) &
         contains_term(df["target_taxon_path"], focal_taxon))
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
    return pairs


def main():
 args = parse_args()
 df = pd.read_csv(args.input, low_memory=False)
 
 if args.verbose:
    print(f"[read] Total rows: {len(df)}")
    df = validate_taxon(df, args.invalid_terms, args.verbose)
    df = deduplicate(df, args.verbose)
    df = filter_interactions(df, args.focal_taxon, args.interacting_taxa, args.verbose)
 if args.output_origin:
    df[ORIGIN_COLUMNS].to_csv(args.output_origin, index=False)
    pairs = extract_pairs(df, args.separator, args.verbose)
    pairs.to_csv(args.output, index=False)
    print(f"\nFile saved in {args.output}")


if __name__ == "__main__":
    main()