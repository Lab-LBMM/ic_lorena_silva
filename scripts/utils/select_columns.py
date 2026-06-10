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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select relevant columns from a processed GloBI CSV file."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/ants_no_duplicates.csv",
        help="Path to the input CSV file (default: data/processed/ants_no_duplicates.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/clean_ants.csv",
        help="Path to the output CSV file (default: data/processed/clean_ants.csv)"
    )
    parser.add_argument(
        "--columns", "-c",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="List of columns to keep (default: predefined relevant columns)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the first rows of the output dataframe"
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    clean_df = df[args.columns]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    clean_df.to_csv(args.output, index=False)

    if args.verbose:
        print(clean_df.head())


if __name__ == "__main__":
    main()