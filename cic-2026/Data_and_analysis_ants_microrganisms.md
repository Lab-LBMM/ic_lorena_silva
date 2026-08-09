# Data and Analysis – Ants and Microorganisms

I downloaded the GloBI data in CSV format.

I used the script from the previous page (**Version 1.5.0**) with the following parameters:

```bash
python3 scripts/process_interactions.py \
  --input "/mnt/c/Users/loren/Downloads/interactions.csv/interactions.csv" \
  --focal-taxon "Formicidae" \
  --memory-limit "3GB" \
  --threads 2 \
  --classify-source \
  --output "/mnt/c/Users/loren/Downloads/resultados_globi/pares_formigas_microorganismos.csv" \
  --output-origin "/mnt/c/Users/loren/Downloads/resultados_globi/origem_formigas_microorganismos.csv"
```

The resulting dataset contained **355 interactions** involving SRO triples between ants and microorganisms, along with source information collected throughout the processing pipeline:

```text
[filter_with_duckdb] Executing filter via DuckDB (streaming, memory_limit=3GB, threads=2)...
[filter_with_duckdb] Total after filter: 1169
[deduplicate_streaming] Deduplicating via DuckDB...
[deduplicate_streaming] Total after: 355
[output_origin] Full dataset saved in /mnt/c/Users/loren/Downloads/resultados_globi/origem_formigas_microorganismos.csv
[load_rules] Using default rules
[classify_dataframe] Classifying 355 records by column 'study_url' (with fallback to 'study_source_archive_uri' when empty)
[build_summary] 3 categories found
=== SOURCE TYPE TABLE ===
           Source Type   n    %
Other / not classified 166 46.8
                 GloBi 140 39.4
       Data Repository  49 13.8
[extract_pairs_streaming] Pairs saved to /mnt/c/Users/loren/Downloads/resultados_globi/pares_formigas_microorganismos.csv
```

### Manual Curation

After manual curation, **179 association pairs with traceable text sources** were identified.