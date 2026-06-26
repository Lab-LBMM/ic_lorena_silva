import argparse
import os
import pandas as pd
__version__ = "1.0.0"


DEFAULT_INVALID_TERMS = ["animalia", "plantae", "fungi", "unknown", "no name"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and clean taxon names from a processed GloBI CSV file."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/clean_ants.csv",
        help="Path to the input CSV file (default: data/processed/clean_ants.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/valid_ants.csv",
        help="Path to the output CSV file (default: data/processed/valid_ants.csv)"
    )
    parser.add_argument(
        "--invalid-terms", "-t",
        nargs="+",
        default=DEFAULT_INVALID_TERMS,
        help="List of taxon names to exclude (default: animalia plantae fungi unknown 'no name')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print row counts before and after validation"
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

    if args.verbose:
        print(f"Total before: {len(df)}")

    valid_df = df.dropna(subset=["source_taxon_name", "target_taxon_name"])

    valid_df = valid_df[
        ~valid_df["source_taxon_name"].str.lower().isin(args.invalid_terms) &
        ~valid_df["target_taxon_name"].str.lower().isin(args.invalid_terms)
    ]

    if args.verbose:
        print(f"Total after: {len(valid_df)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    valid_df.to_csv(args.output, index=False)
    print(f"File saved in {args.output}")


if __name__ == "__main__":
    main()