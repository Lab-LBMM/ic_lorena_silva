import argparse
import os
import pandas as pd
__version__ = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract unique interaction pairs from a processed GloBI CSV file."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/valid_ants.csv",
        help="Path to the input CSV file (default: data/processed/valid_ants.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/pairs_unique.csv",
        help="Path to the output CSV file (default: data/processed/pairs_unique.csv)"
    )
    parser.add_argument(
        "--separator", "-s",
        default=" | ",
        help="Separator used to join the pair fields (default: ' | ')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print total count and sample of unique pairs"
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

    df["pair_interaction"] = (
        df["source_taxon_name"] + args.separator +
        df["interaction_type"] + args.separator +
        df["target_taxon_name"]
    )

    pairs_unique = df["pair_interaction"].drop_duplicates().reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pairs_unique.to_csv(args.output, index=False)

    if args.verbose:
        print(f"Total unique pairs: {len(pairs_unique)}")
        print(pairs_unique.head(10))


if __name__ == "__main__":
    main()