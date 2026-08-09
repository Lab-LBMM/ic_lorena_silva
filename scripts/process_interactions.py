#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile

import duckdb
import pandas as pd

__version__ = "1.6.0"

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
    ["GloBi", ["trophich", "globalbioticinteractions"]],
    ["Occurrence Observation", [
        "inaturalist", "vertnet", "bold", "scan", "fmnh", "mcz",
        "ucsb-izc", "ecdysis", "osal", "emtuckerlab"
    ]],
    ["Data Repository", ["zenodo", "10.5285", "10.15468"]],
    ["Scientific (article with DOI)", ["doi"]],
    ["Institutional Catalog", ["gbif", "museum", "catalog", "collection", "specimen"]],
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
    parser.add_argument("--any-side", action="store_true")
    parser.add_argument("--separator", "-s", default=" , ")
    parser.add_argument("--output-origin", default=None)
    parser.add_argument("--classify-source", action="store_true")
    parser.add_argument("--url-column", default="study_url")
    parser.add_argument("--rules-file", default=None)
    parser.add_argument("--empty-category", default="Without URL")
    parser.add_argument("--fallback-category", default="Other / not classified")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--classified-output", default=None)
    parser.add_argument("--memory-limit", default="4GB",
                         help="Limite de memória para o DuckDB (ex: '4GB', '3GB'). Default: 4GB")
    parser.add_argument("--threads", type=int, default=2,
                         help="Número de threads para o DuckDB. Default: 2")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def esc(s):
    return str(s).replace("'", "''")


def build_like_or(column, terms):
    return " OR ".join(f"{column} ILIKE '%{esc(t)}%'" for t in terms)


def filter_with_duckdb(input_path, invalid_terms, focal_taxon, interacting_taxa, any_side,
                        memory_limit, threads, tmp_filtered_path):
    """
    Executa o filtro via DuckDB e escreve o resultado direto em disco via COPY,
    evitando materializar o resultado inteiro em memória (pandas .df()).
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    invalid_terms_lower = [t.lower() for t in invalid_terms]
    invalid_list_sql = ", ".join(f"'{esc(t)}'" for t in invalid_terms_lower)

    if any_side:
        interaction_condition = (
            f"( sourceTaxonPathNames ILIKE '%{esc(focal_taxon)}%' "
            f"  OR targetTaxonPathNames ILIKE '%{esc(focal_taxon)}%' )"
        )
    else:
        interacting_taxa = interacting_taxa if isinstance(interacting_taxa, list) else [interacting_taxa]
        interaction_condition = (
            f"( ( sourceTaxonPathNames ILIKE '%{esc(focal_taxon)}%' "
            f"    AND ({build_like_or('targetTaxonPathNames', interacting_taxa)}) ) "
            f" OR ( targetTaxonPathNames ILIKE '%{esc(focal_taxon)}%' "
            f"       AND ({build_like_or('sourceTaxonPathNames', interacting_taxa)}) ) )"
        )

    query = f"""
        SELECT
            sourceTaxonName AS source_taxon_name,
            interactionTypeName AS interaction_type,
            targetTaxonName AS target_taxon_name,
            referenceCitation AS study_citation,
            referenceUrl AS study_url,
            sourceArchiveURI AS study_source_archive_uri
        FROM read_csv('{esc(input_path)}', AUTO_DETECT=TRUE)
        WHERE sourceTaxonName IS NOT NULL
          AND targetTaxonName IS NOT NULL
          AND LOWER(sourceTaxonName) NOT IN ({invalid_list_sql})
          AND LOWER(targetTaxonName) NOT IN ({invalid_list_sql})
          AND {interaction_condition}
    """

    print(f"[filter_with_duckdb] Executando filtro via DuckDB (streaming, memory_limit={memory_limit}, threads={threads})...")
    con.execute(f"""
        COPY (
            {query}
        ) TO '{tmp_filtered_path}' (FORMAT CSV, HEADER)
    """)
    con.close()

    # Conta linhas sem carregar tudo em memória (usa o próprio DuckDB)
    con2 = duckdb.connect()
    n_rows = con2.execute(
        f"SELECT COUNT(*) FROM read_csv('{esc(tmp_filtered_path)}', AUTO_DETECT=TRUE)"
    ).fetchone()[0]
    con2.close()
    print(f"[filter_with_duckdb] Total após filtro: {n_rows}")
    return tmp_filtered_path


def deduplicate_streaming(filtered_csv_path, deduped_csv_path, memory_limit, threads):
    """
    Deduplicação feita via SQL (DuckDB), sem carregar tudo em pandas.
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    print("[deduplicate_streaming] Deduplicando via DuckDB...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT ON (source_taxon_name, interaction_type, target_taxon_name) *
            FROM read_csv('{esc(filtered_csv_path)}', AUTO_DETECT=TRUE)
        ) TO '{deduped_csv_path}' (FORMAT CSV, HEADER)
    """)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)"
    ).fetchone()[0]
    con.close()
    print(f"[deduplicate_streaming] Total after: {n_rows}")
    return deduped_csv_path


def extract_pairs_streaming(deduped_csv_path, output_path, separator, memory_limit, threads):
    """
    Gera os pares únicos Source-Relation-Target direto via SQL, sem passar por pandas.
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    sep_escaped = esc(separator)
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                source_taxon_name || '{sep_escaped}' || interaction_type || '{sep_escaped}' || target_taxon_name AS pair
            FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
        ) TO '{output_path}' (FORMAT CSV, HEADER)
    """)
    con.close()
    print(f"[extract_pairs_streaming] Pares salvos em {output_path}")


def load_rules(rules_file):
    if rules_file is None:
        print("[load_rules] Using default rules")
        return DEFAULT_RULES
    with open(rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)
    print(f"[load_rules] Rules loaded from {rules_file}: {len(rules)} categories")
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


def classify_dataframe(df, url_column, fallback_column, rules, empty_category, fallback_category):
    if url_column not in df.columns:
        sys.exit(f"Error: column '{url_column}' not found in CSV.")

    print(f"[classify_dataframe] Classifying {len(df)} records by column '{url_column}' "
          f"(with fallback to '{fallback_column}' when empty)")

    df = df.copy()

    def resolve_value(row):
        primary = row.get(url_column)
        if pd.isna(primary) or str(primary).strip() == "" or str(primary).strip().lower() == "nan":
            return row.get(fallback_column)
        return primary

    df["_source_value"] = df.apply(resolve_value, axis=1)
    df["category"] = df["_source_value"].apply(
        lambda v: classify_value(v, rules, empty_category, fallback_category)
    )
    df = df.drop(columns=["_source_value"])
    return df


def build_summary(df):
    tabela = df["category"].value_counts().reset_index()
    tabela.columns = ["Source Type", "n"]
    tabela["%"] = (tabela["n"] / len(df) * 100).round(1)
    print(f"[build_summary] {len(tabela)} categories found")
    return tabela


def run_classification(deduped_csv_path, args):
    df_origin = pd.read_csv(deduped_csv_path, usecols=lambda c: c in ORIGIN_COLUMNS)

    rules = load_rules(args.rules_file)
    df_classified = classify_dataframe(
        df_origin, args.url_column, "study_source_archive_uri", rules,
        args.empty_category, args.fallback_category
    )
    tabela = build_summary(df_classified)

    print("=== SOURCE TYPE TABLE ===")
    print(tabela.to_string(index=False))

    if args.classified_output:
        df_classified.to_csv(args.classified_output, index=False)

    if args.summary_output:
        tabela.to_csv(args.summary_output, index=False)

    return df_classified


def main():
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        filtered_path = os.path.join(tmp_dir, "filtered.csv")
        deduped_path = os.path.join(tmp_dir, "deduped.csv")

        filter_with_duckdb(
            args.input, args.invalid_terms, args.focal_taxon, args.interacting_taxa,
            args.any_side, args.memory_limit, args.threads, filtered_path
        )

        deduplicate_streaming(filtered_path, deduped_path, args.memory_limit, args.threads)

        if args.output_origin:
            # Copia só as colunas de origem, direto via DuckDB, sem pandas
            con = duckdb.connect()
            con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
            cols_sql = ", ".join(ORIGIN_COLUMNS)
            con.execute(f"""
                COPY (
                    SELECT {cols_sql} FROM read_csv('{esc(deduped_path)}', AUTO_DETECT=TRUE)
                ) TO '{esc(args.output_origin)}' (FORMAT CSV, HEADER)
            """)
            con.close()
            print(f"[output_origin] Full dataset saved in {args.output_origin}")

        if args.classify_source:
            run_classification(deduped_path, args)

        extract_pairs_streaming(deduped_path, args.output, args.separator,
                                 args.memory_limit, args.threads)


if __name__ == "__main__":
    main()