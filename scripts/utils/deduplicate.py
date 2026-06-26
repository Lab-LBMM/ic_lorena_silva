import argparse
import os
import pandas as pd
__version__ = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove duplicate interactions from a GloBI processed CSV file."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/ants_interactions.csv",
        help="Path to the input CSV file (default: data/processed/ants_interactions.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/ants_no_duplicates.csv",
        help="Path to the output CSV file (default: data/processed/ants_no_duplicates.csv)"
    )
    parser.add_argument(
        "--subset", "-s",
        nargs="+",
        default=["source_taxon_name", "interaction_type", "target_taxon_name"],
        help="Columns to consider when identifying duplicates (default: source_taxon_name interaction_type target_taxon_name)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print row counts before and after deduplication"
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

    df_no_duplicates = df.drop_duplicates(subset=args.subset)

    if args.verbose:
        print(f"Total after: {len(df_no_duplicates)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_no_duplicates.to_csv(args.output, index=False)
    print(f"File saved in {args.output}")


if __name__ == "__main__":
    main()