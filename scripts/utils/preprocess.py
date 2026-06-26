import argparse
import os
import pandas as pd
__version__ = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw GloBI interactions CSV and save to processed directory."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/interactions.csv",
        help="Path to the raw input CSV file (default: data/raw/interactions.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/ants_interactions.csv",
        help="Path to the output CSV file (default: data/processed/ants_interactions.csv)"
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"File saved in {args.output}")


if __name__ == "__main__":
    main()