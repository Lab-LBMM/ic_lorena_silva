import argparse
import pandas as pd
__version__ = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter interactions between ants (Formicidae) and microorganisms from a GloBI dataset."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/ants_interactions.csv",
        help="Path to the input CSV file (default: data/processed/ants_interactions.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/ants_microorganisms.csv",
        help="Path to the output CSV file (default: data/processed/ants_microorganisms.csv)"
    )
    parser.add_argument(
        "--ants", "-a",
        default="Formicidae",
        help="Taxon name to filter as ants (default: Formicidae)"
    )
    parser.add_argument(
        "--microorganisms", "-m",
        nargs="+",
        default=["Fungi", "Bacteria"],
        help="List of taxon names to filter as microorganisms (default: Fungi Bacteria)"
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


def main():
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    filter_condition = (
        (contains_term(df["source_taxon_path"], args.ants) &
         contains_term(df["target_taxon_path"], args.microorganisms)) |
        (contains_term(df["source_taxon_path"], args.microorganisms) &
         contains_term(df["target_taxon_path"], args.ants))
    )

    filtered_df = df[filter_condition]

    filtered_df.to_csv(args.output, index=False)
    print(f"\nFile saved in {args.output}")


if __name__ == "__main__":
    main()