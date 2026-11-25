#!/usr/bin/env python3

import argparse
import os
import tempfile

import duckdb
import pandas as pd

__version__ = "2.3.0"

ORIGIN_COLUMNS = [
    "source_taxon_name",
    "interaction_type",
    "target_taxon_name",
    "study_citation",
    "study_source_archive_uri"
]

DEFAULT_INVALID_TERMS = ["animalia", "plantae", "fungi", "unknown", "no name"]

PMC_ARTICLE_PREFIX = "https://pmc.ncbi.nlm.nih.gov/articles/"
NUCCORE_PREFIX = "http://www.ncbi.nlm.nih.gov/nuccore/"

TRACEABLE_LABEL = "Traceable (PMC article)"
MAYBE_TRACEABLE_LABEL = "Possibly traceable (nuccore) - manual check required"
NOT_TRACEABLE_LABEL = "Not traceable"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for filtering, cleaning, and extracting unique Subject-Relation-Object pairs from interaction datasets, with source traceability classification."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--invalid-terms", "-t", nargs="+", default=DEFAULT_INVALID_TERMS)
    parser.add_argument("--focal-taxon", "-a", default="Formicidae")
    parser.add_argument("--interacting-taxa", "-m", nargs="+", default=["Fungi", "Bacteria"])
    parser.add_argument("--any-side", action="store_true")
    parser.add_argument("--separator", "-s", default=" , ")
    parser.add_argument("--output-origin", default=None)
    parser.add_argument("--output-pmcid-list", "-p", default=None,
                        help="Path to save a TXT file containing only unique PMCIDs (e.g., PMC1234567).")
    parser.add_argument("--output-triplets", "-t-csv", default=None,
                        help="Path to save a CSV file with PMCIDs and SRO triplets (pmid;sourceTaxonName;interactionTypeName;targetTaxonName).")
    parser.add_argument("--strict-species", action="store_true",
                        help="Discard unspecific single-word taxa (e.g., keep only binomial species names).")
    parser.add_argument("--memory-limit", default="4GB",
                        help="Memory limit for DuckDB (e.g., '4GB', '3GB'). Default: 4GB")
    parser.add_argument("--threads", type=int, default=2,
                        help="Number of threads for DuckDB. Default: 2")
    parser.add_argument("--stats", action="store_true",
                        help="Compute detailed pipeline statistics (funnel, top species, "
                             "interaction types, deduplication collapse) and generate PNG charts.")
    parser.add_argument("--stats-dir", default=None,
                        help="Folder to save the statistics PNG charts. "
                             "Default: a 'stats' subfolder next to --output.")
    parser.add_argument("--stats-top-n", type=int, default=15,
                        help="How many items to show/plot in rankings (top species, interaction "
                             "types, etc). Default: 15")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def esc(s):
    return str(s).replace("'", "''")


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_like_or(column, terms):
    return " OR ".join(f"{column} ILIKE '%{esc(t)}%'" for t in terms)


def count_csv_rows(csv_path, memory_limit, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{esc(csv_path)}', AUTO_DETECT=TRUE)"
    ).fetchone()[0]
    con.close()
    return n_rows


def filter_with_duckdb(input_path, invalid_terms, focal_taxon, interacting_taxa, any_side,
                       strict_species, memory_limit, threads, tmp_filtered_path):
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

    species_strict_condition = ""
    if strict_species:
        species_strict_condition = """
            AND length(trim(sourceTaxonName)) - length(replace(trim(sourceTaxonName), ' ', '')) >= 1
            AND length(trim(targetTaxonName)) - length(replace(trim(targetTaxonName), ' ', '')) >= 1
        """

    query = f"""
        SELECT
            trim(sourceTaxonName) AS source_taxon_name,
            trim(interactionTypeName) AS interaction_type,
            trim(targetTaxonName) AS target_taxon_name,
            trim(referenceCitation) AS study_citation,
            trim(sourceArchiveURI) AS study_source_archive_uri
        FROM read_csv('{esc(input_path)}', AUTO_DETECT=TRUE)
        WHERE sourceTaxonName IS NOT NULL
          AND targetTaxonName IS NOT NULL
          AND LOWER(sourceTaxonName) NOT IN ({invalid_list_sql})
          AND LOWER(targetTaxonName) NOT IN ({invalid_list_sql})
          AND {interaction_condition}
          {species_strict_condition}
    """

    print(f"[filter_with_duckdb] Running filter via DuckDB (streaming, memory_limit={memory_limit}, threads={threads})...")
    con.execute(f"""
        COPY (
            {query}
        ) TO '{tmp_filtered_path}' (FORMAT CSV, HEADER)
    """)
    con.close()

    con2 = duckdb.connect()
    n_rows = con2.execute(
        f"SELECT COUNT(*) FROM read_csv('{esc(tmp_filtered_path)}', AUTO_DETECT=TRUE)"
    ).fetchone()[0]
    con2.close()
    print(f"[filter_with_duckdb] Total after filter: {n_rows}")
    return tmp_filtered_path, n_rows


def deduplicate_streaming(filtered_csv_path, deduped_csv_path, memory_limit, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    print("[deduplicate_streaming] Deduplicating via DuckDB with specificity ranking...")
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT *,
                    (CASE WHEN length(source_taxon_name) - length(replace(source_taxon_name, ' ', '')) >= 1 THEN 1 ELSE 0 END +
                     CASE WHEN length(target_taxon_name) - length(replace(target_taxon_name, ' ', '')) >= 1 THEN 1 ELSE 0 END) AS spec_score,
                    (CASE WHEN study_citation LIKE '{PMC_ARTICLE_PREFIX}%' THEN 2 
                          WHEN study_citation LIKE '{NUCCORE_PREFIX}%' THEN 1 ELSE 0 END) AS trace_score,
                    ROW_NUMBER() OVER (
                        PARTITION BY source_taxon_name, interaction_type, target_taxon_name, study_citation
                        ORDER BY spec_score DESC, trace_score DESC
                    ) AS rn
                FROM read_csv('{esc(filtered_csv_path)}', AUTO_DETECT=TRUE)
            )
            SELECT source_taxon_name, interaction_type, target_taxon_name, study_citation, study_source_archive_uri
            FROM ranked
            WHERE rn = 1
        ) TO '{deduped_csv_path}' (FORMAT CSV, HEADER)
    """)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)"
    ).fetchone()[0]
    con.close()
    print(f"[deduplicate_streaming] Total after: {n_rows}")
    return deduped_csv_path, n_rows


def extract_pairs_streaming(deduped_csv_path, output_path, separator, memory_limit, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    sep_escaped = esc(separator)
    ensure_parent_dir(output_path)
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                source_taxon_name || '{sep_escaped}' || interaction_type || '{sep_escaped}' || target_taxon_name AS pair
            FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
        ) TO '{output_path}' (FORMAT CSV, HEADER)
    """)
    con.close()
    print(f"[extract_pairs_streaming] Pairs saved to {output_path}")


def extract_pmc_ids_to_txt(deduped_csv_path, output_txt_path, memory_limit, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    ensure_parent_dir(output_txt_path)

    df_pmcids = con.execute(f"""
        SELECT DISTINCT 
            regexp_extract(study_citation, '(PMC[0-9]+)', 1) AS pmcid
        FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
        WHERE study_citation LIKE '{PMC_ARTICLE_PREFIX}%'
          AND regexp_extract(study_citation, '(PMC[0-9]+)', 1) != ''
        ORDER BY pmcid
    """).df()
    con.close()

    pmc_list = df_pmcids['pmcid'].dropna().tolist()

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for pmc_id in pmc_list:
            f.write(f"{pmc_id}\n")

    print(f"[extract_pmc_ids_to_txt] Saved {len(pmc_list)} unique PMCIDs to {output_txt_path}")


def write_triplets_with_pmcid(deduped_csv_path, output_csv_path, memory_limit, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    ensure_parent_dir(output_csv_path)

    print("[write_triplets_with_pmcid] Exporting PMC triplets CSV...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                regexp_extract(study_citation, '(PMC[0-9]+)', 1) AS pmid,
                source_taxon_name AS sourceTaxonName,
                interaction_type AS interactionTypeName,
                target_taxon_name AS targetTaxonName
            FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
            WHERE study_citation LIKE '{PMC_ARTICLE_PREFIX}%'
              AND regexp_extract(study_citation, '(PMC[0-9]+)', 1) != ''
            ORDER BY pmid, sourceTaxonName
        ) TO '{esc(output_csv_path)}' (FORMAT CSV, HEADER, DELIMITER ';')
    """)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{esc(output_csv_path)}', AUTO_DETECT=TRUE, delim=';')"
    ).fetchone()[0]
    con.close()
    print(f"[write_triplets_with_pmcid] Saved {n_rows} PMC-SRO triplets to {output_csv_path}")


def classify_traceability(citation):
    if pd.isna(citation):
        return NOT_TRACEABLE_LABEL
    text = str(citation).strip()
    if text.startswith(PMC_ARTICLE_PREFIX):
        return TRACEABLE_LABEL
    if text.startswith(NUCCORE_PREFIX):
        return MAYBE_TRACEABLE_LABEL
    return NOT_TRACEABLE_LABEL


def write_origin_with_traceability(deduped_csv_path, output_path, memory_limit, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    cols_sql = ", ".join(ORIGIN_COLUMNS)
    df = con.execute(f"""
        SELECT {cols_sql} FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
    """).df()
    con.close()

    df["_traceability"] = df["study_citation"].apply(classify_traceability)

    traceable_df = df[df["_traceability"] == TRACEABLE_LABEL].drop(columns=["_traceability"])
    maybe_df = df[df["_traceability"] == MAYBE_TRACEABLE_LABEL].drop(columns=["_traceability"])
    n_discarded = len(df) - len(traceable_df) - len(maybe_df)

    print(f"[write_origin_with_traceability] Traceable (PMC): {len(traceable_df)}")
    print(f"[write_origin_with_traceability] Possibly traceable (nuccore): {len(maybe_df)}")
    print(f"[write_origin_with_traceability] Not traceable (discarded): {n_discarded}")

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        traceable_df.to_csv(f, index=False)
        f.write("\n")
        f.write(f"# {MAYBE_TRACEABLE_LABEL}\n")
        maybe_df.to_csv(f, index=False)

    print(f"[write_origin_with_traceability] File saved to {output_path}")

    return {
        TRACEABLE_LABEL: len(traceable_df),
        MAYBE_TRACEABLE_LABEL: len(maybe_df),
        NOT_TRACEABLE_LABEL: n_discarded,
    }


def compute_funnel_stats(n_input, n_filtered, n_deduped):
    retention_pct = round(n_filtered / n_input * 100, 2) if n_input else 0.0
    dedup_removed_pct = round((n_filtered - n_deduped) / n_filtered * 100, 2) if n_filtered else 0.0
    print("=== PIPELINE FUNNEL ===")
    print(f"Input (raw GloBI): {n_input}")
    print(f"After filter (focal/interacting taxa): {n_filtered} ({retention_pct}% of raw)")
    print(f"After deduplication: {n_deduped} ({dedup_removed_pct}% removed as duplicate)")


def compute_dedup_collapse_stats(filtered_csv_path, memory_limit, threads, top_n):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    triad_counts = con.execute(f"""
        SELECT source_taxon_name, interaction_type, target_taxon_name,
               COUNT(*) AS n_raw_records
        FROM read_csv('{esc(filtered_csv_path)}', AUTO_DETECT=TRUE)
        GROUP BY source_taxon_name, interaction_type, target_taxon_name
        ORDER BY n_raw_records DESC
        LIMIT {top_n}
    """).df()

    species_counts = con.execute(f"""
        WITH all_species AS (
            SELECT source_taxon_name AS species
            FROM read_csv('{esc(filtered_csv_path)}', AUTO_DETECT=TRUE)
            UNION ALL
            SELECT target_taxon_name AS species
            FROM read_csv('{esc(filtered_csv_path)}', AUTO_DETECT=TRUE)
        )
        SELECT species, COUNT(*) AS n_appearances
        FROM all_species
        GROUP BY species
        ORDER BY n_appearances DESC
        LIMIT {top_n}
    """).df()
    con.close()

    print(f"\n=== TOP {top_n} PAIRS THAT COLLAPSED THE MOST DURING DEDUPLICATION ===")
    print(triad_counts.to_string(index=False))
    print(f"\n=== TOP {top_n} SPECIES BY NUMBER OF RAW APPEARANCES (source + target, before dedup) ===")
    print(species_counts.to_string(index=False))

    return triad_counts, species_counts


def compute_interaction_and_species_stats(deduped_csv_path, memory_limit, threads, top_n):
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    interaction_dist = con.execute(f"""
        SELECT interaction_type, COUNT(*) AS n
        FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
        GROUP BY interaction_type
        ORDER BY n DESC
    """).df()

    top_source = con.execute(f"""
        SELECT source_taxon_name AS species, COUNT(*) AS n
        FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
        GROUP BY source_taxon_name
        ORDER BY n DESC
        LIMIT {top_n}
    """).df()

    top_target = con.execute(f"""
        SELECT target_taxon_name AS species, COUNT(*) AS n
        FROM read_csv('{esc(deduped_csv_path)}', AUTO_DETECT=TRUE)
        GROUP BY target_taxon_name
        ORDER BY n DESC
        LIMIT {top_n}
    """).df()
    con.close()

    print(f"\n=== INTERACTION TYPE DISTRIBUTION (final set, top {top_n}) ===")
    print(interaction_dist.head(top_n).to_string(index=False))
    print(f"\n=== TOP {top_n} SOURCE SPECIES (final set) ===")
    print(top_source.to_string(index=False))
    print(f"\n=== TOP {top_n} TARGET SPECIES (final set) ===")
    print(top_target.to_string(index=False))

    return interaction_dist, top_source, top_target


def save_bar_chart(labels, values, title, xlabel, ylabel, output_path, horizontal=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_bar_chart] matplotlib not installed -- skipped chart generation.")
        return None

    labels = [str(v) for v in labels]
    fig_height = max(4, 0.4 * len(labels)) if horizontal else 5
    fig, ax = plt.subplots(figsize=(9, fig_height))

    if horizontal:
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel(ylabel)
    else:
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")

    ax.set_title(title)
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[save_bar_chart] Chart saved to {output_path}")
    return output_path


def run_stats(args, n_input, n_filtered, n_deduped, filtered_path, deduped_path,
              traceability_counts):
    stats_dir = args.stats_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output)) or ".", "stats"
    )
    top_n = args.stats_top_n

    compute_funnel_stats(n_input, n_filtered, n_deduped)

    triad_counts, species_counts = compute_dedup_collapse_stats(
        filtered_path, args.memory_limit, args.threads, top_n
    )

    interaction_dist, top_source, top_target = compute_interaction_and_species_stats(
        deduped_path, args.memory_limit, args.threads, top_n
    )

    if traceability_counts:
        save_bar_chart(
            list(traceability_counts.keys()), list(traceability_counts.values()),
            "Origin traceability (study_citation)", "Category", "Number of records",
            os.path.join(stats_dir, "02_traceability.png"), horizontal=True,
        )

    save_bar_chart(
        interaction_dist["interaction_type"].head(top_n).tolist(),
        interaction_dist["n"].head(top_n).tolist(),
        f"Top {top_n} interaction types (final set)", "Interaction type", "Number of records",
        os.path.join(stats_dir, "03_interaction_types.png"), horizontal=True,
    )

    save_bar_chart(
        top_source["species"].tolist(), top_source["n"].tolist(),
        f"Top {top_n} source species (final set)", "Species", "Number of records",
        os.path.join(stats_dir, "04_top_source_species.png"), horizontal=True,
    )

    save_bar_chart(
        top_target["species"].tolist(), top_target["n"].tolist(),
        f"Top {top_n} target species (final set)", "Species", "Number of records",
        os.path.join(stats_dir, "05_top_target_species.png"), horizontal=True,
    )

    save_bar_chart(
        species_counts["species"].tolist(), species_counts["n_appearances"].tolist(),
        f"Top {top_n} species by raw appearances (before dedup)", "Species", "Number of appearances",
        os.path.join(stats_dir, "06_species_collapsed_in_dedup.png"), horizontal=True,
    )


def main():
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        filtered_path = os.path.join(tmp_dir, "filtered.csv")
        deduped_path = os.path.join(tmp_dir, "deduped.csv")

        n_input = count_csv_rows(args.input, args.memory_limit, args.threads) if args.stats else None

        filtered_path, n_filtered = filter_with_duckdb(
            args.input, args.invalid_terms, args.focal_taxon, args.interacting_taxa,
            args.any_side, args.strict_species, args.memory_limit, args.threads, filtered_path
        )

        deduped_path, n_deduped = deduplicate_streaming(
            filtered_path, deduped_path, args.memory_limit, args.threads
        )

        traceability_counts = None
        if args.output_origin:
            traceability_counts = write_origin_with_traceability(
                deduped_path, args.output_origin, args.memory_limit, args.threads
            )

        if args.output_pmcid_list:
            extract_pmc_ids_to_txt(
                deduped_path, args.output_pmcid_list, args.memory_limit, args.threads
            )

        if args.output_triplets:
            write_triplets_with_pmcid(
                deduped_path, args.output_triplets, args.memory_limit, args.threads
            )

        extract_pairs_streaming(deduped_path, args.output, args.separator,
                                 args.memory_limit, args.threads)

        if args.stats:
            run_stats(
                args, n_input, n_filtered, n_deduped, filtered_path, deduped_path,
                traceability_counts
            )


if __name__ == "__main__":
    main()