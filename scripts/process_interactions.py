#!/usr/bin/env python3

import argparse
import json
import sys

import pandas as pd

__version__ = "1.2.0"

ORIGIN_COLUMNS = [
    "source_taxon_name",
    "interaction_type",
    "target_taxon_name",
    "study_citation",
    "study_url",
    "study_source_archive_uri"
]

DEFAULT_INVALID_TERMS = ["animalia", "plantae", "fungi", "unknown", "no name"]

DEFAULT_RULES = [
    ["Rede trófica", ["trophich", "globalbioticinteractions"]],
    ["Observação de ocorrência", [
        "inaturalist", "vertnet", "bold", "scan", "fmnh", "mcz",
        "ucsb-izc", "ecdysis", "osal", "emtuckerlab"
    ]],
    ["Repositório de dados", ["zenodo", "10.5285", "10.15468"]],
    ["Científica (artigo com DOI)", ["doi"]],
    ["Catálogo institucional", ["gbif", "museum", "catalog", "collection", "specimen"]],
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for filtering, cleaning, and extracting unique Subject–Relation–Object pairs from interaction datasets, with source type classification."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--invalid-terms", "-t", nargs="+", default=DEFAULT_INVALID_TERMS)
    parser.add_argument("--focal-taxon", "-a", default="Formicidae")
    parser.add_argument("--interacting-taxa", "-m", nargs="+", default=["Fungi", "Bacteria"])
    parser.add_argument("--separator", "-s", default=" , ")
    parser.add_argument("--output-origin", default=None)
    parser.add_argument("--classify-source", action="store_true")
    parser.add_argument("--url-column", default="study_url")
    parser.add_argument("--rules-file", default=None)
    parser.add_argument("--empty-category", default="Mineração de texto")
    parser.add_argument("--fallback-category", default="Outro / não classificado")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--classified-output", default=None)
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def contains_term(series, terms):
    if isinstance(terms, str):
        terms = [terms]
    return series.str.contains("|".join(terms), na=False, case=False)


def validate_taxon(df, invalid_terms):
    print(f"[validate_taxon] Total before: {len(df)}")
    df = df.dropna(subset=["source_taxon_name", "target_taxon_name"])
    df = df[
        ~df["source_taxon_name"].str.lower().isin(invalid_terms) &
        ~df["target_taxon_name"].str.lower().isin(invalid_terms)
    ]
    print(f"[validate_taxon] Total after: {len(df)}")
    return df


def deduplicate(df):
    print(f"[deduplicate] Total before: {len(df)}")
    df = df.drop_duplicates(
        subset=["source_taxon_name", "interaction_type", "target_taxon_name"]
    )
    print(f"[deduplicate] Total after: {len(df)}")
    return df


def filter_interactions(df, focal_taxon, interacting_taxa):
    print(f"[filter_interactions] Total before: {len(df)}")
    condition = (
        (contains_term(df["source_taxon_path"], focal_taxon) &
         contains_term(df["target_taxon_path"], interacting_taxa)) |
        (contains_term(df["source_taxon_path"], interacting_taxa) &
         contains_term(df["target_taxon_path"], focal_taxon))
    )
    df = df[condition]
    print(f"[filter_interactions] Total after: {len(df)}")
    return df


def extract_pairs(df, separator):
    return (
        df["source_taxon_name"] + separator +
        df["interaction_type"] + separator +
        df["target_taxon_name"]
    ).drop_duplicates().reset_index(drop=True)


def load_rules(rules_file):
    if rules_file is None:
        print("[load_rules] Usando regras padrão embutidas")
        return DEFAULT_RULES
    with open(rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)
    print(f"[load_rules] Regras carregadas de {rules_file}: {len(rules)} categorias")
    return rules


def classify_value(value, rules, empty_category, fallback_category):
    if pd.isna(value):
        return empty_category
    text = str(value).strip().lower()
    if text == "" or text == "nan":
        return empty_category
    for category, keywords in rules:
        if any(kw.lower() in text for kw in keywords):
            return category
    return fallback_category


def classify_dataframe(df, url_column, rules, empty_category, fallback_category):
    if url_column not in df.columns:
        sys.exit(f"Erro: coluna '{url_column}' não encontrada no CSV.")
    print(f"[classify_dataframe] Classificando {len(df)} registros pela coluna '{url_column}'")
    df = df.copy()
    df["categoria"] = df[url_column].apply(
        lambda v: classify_value(v, rules, empty_category, fallback_category)
    )
    return df


def build_summary(df):
    tabela = df["categoria"].value_counts().reset_index()
    tabela.columns = ["Tipo de fonte", "n"]
    tabela["%"] = (tabela["n"] / len(df) * 100).round(1)
    print(f"[build_summary] {len(tabela)} categorias encontradas")
    return tabela


def run_classification(df_origin, args):
    rules = load_rules(args.rules_file)
    df_classified = classify_dataframe(
        df_origin, args.url_column, rules,
        args.empty_category, args.fallback_category
    )
    tabela = build_summary(df_classified)

    print("=== TABELA DE TIPO DE FONTE ===")
    print(tabela.to_string(index=False))

    if args.classified_output:
        df_classified.to_csv(args.classified_output, index=False)

    if args.summary_output:
        tabela.to_csv(args.summary_output, index=False)


def main():
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)

    df = validate_taxon(df, args.invalid_terms)
    df = deduplicate(df)
    df = filter_interactions(df, args.focal_taxon, args.interacting_taxa)

    if args.output_origin:
        df[ORIGIN_COLUMNS].to_csv(args.output_origin, index=False)
        print(f"[output_origin] Full dataset saved in {args.output_origin}")

    if args.classify_source:
        run_classification(df[ORIGIN_COLUMNS], args)

    pairs = extract_pairs(df, args.separator)
    pairs.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()