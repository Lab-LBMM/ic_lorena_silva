This repository contains the code, data, and documentation for the Undergraduate Research Project **“Fine-tuning large language models for the extraction of ecological relationships involving microorganisms and ants”** (FAPESP 2026/12637-3).

The project aims to evaluate and fine-tune large language models (LLMs) for the automatic extraction of ecological relationships between ants (Formicidae) and microorganisms from scientific literature, building a reproducible pipeline that spans from the curation of raw data (GloBI, BioC-PMC) to the construction of a knowledge graph of the identified interactions.

## Team

* **Advisor:** Prof. Dr. Renato Augusto Corrêa dos Santos (CEIS/IB/UNESP Rio Claro)
* **External collaborator:** Prof. Dr. Ruben Interian (Institute of Computing/UNICAMP)
* **Undergraduate Research Fellow:** Lorena Silva (Biological Sciences, UNESP Rio Claro)

## Methodology

The project follows a three-phase methodology:

1. **Baseline evaluation** — selection and screening of interaction data (GloBI) and evaluation of candidate LLMs based on eligibility criteria (license, availability of model weights, support for LoRA/QLoRA, computational feasibility, active maintenance, and technical documentation).
2. **Fine-tuning (LoRA/QLoRA)** — supervised fine-tuning of the selected model(s) for relationship extraction from narrative text (BioC-PMC corpus), using annotations based on the SRO (Subject–Relation–Object) scheme.
3. **Generalization testing and knowledge graph construction** — evaluation of the fine-tuned model on unseen data and organization of the extracted relationships into a knowledge graph.

## Requirements

* Python 3.x with the libraries listed in each `scripts/` subdirectory.

## Installation

Clone this repository:

```bash
git clone https://github.com/Lab-LBMM/ic_lorena_silva.git
cd ic_lorena_silva
```

## Repository Structure

```text
scripts/
  process_interactions.py   # main data curation and processing pipeline
cic/                         # step-by-step analysis documentation exported from Notion,
                             # corresponding to the work presented at the CIC
  poster/                    # poster presented at the CIC (PDF)
```

### Main Script

* `process_interactions.py` — pipeline for filtering, cleaning, and extracting unique Subject–Relation–Object pairs from raw interaction data (GloBI), with optional classification of the source type for each record. The pipeline processes the data using DuckDB (streaming, without loading the entire dataset into memory) and performs the following steps:

  1. **Filtering** — removes invalid terms (e.g., `animalia`, `plantae`, `unknown`) and selects interactions between the focal taxon (default: `Formicidae`) and the taxa of interest (default: `Fungi`, `Bacteria`), either on either side of the interaction or only when the focal taxon occurs on one side (`--any-side`).
  2. **Deduplication** — removes duplicate records based on the combination of source taxon, interaction type, and target taxon.
  3. **SRO pair extraction** — generates a list of unique Source–Relation–Target pairs.
  4. **Optional source classification (`--classify-source`)** — classifies each record by source type (e.g., GloBI, occurrence observation, data repository, scientific article with DOI, institutional catalog) based on keyword rules applied to the study URL, with a fallback to the source archive identifier. It also generates a summary table with the count and percentage for each category.

**Main parameters:** `--input`/`-i`, `--output`/`-o`, `--focal-taxon`/`-a`, `--interacting-taxa`/`-m`, `--any-side`, `--output-origin` (saves the complete dataset, including source metadata, before pair extraction), `--classify-source`, `--summary-output`, `--classified-output`, `--memory-limit`, `--threads`.

**Current version:** `1.6.0`.

## Input Data

The pipeline uses raw ecological interaction data extracted from [GloBI](https://www.globalbioticinteractions.org/) and, during the fine-tuning stages, scientific article texts from the [BioC-PMC](https://www.ncbi.nlm.nih.gov/pmc/tools/textmining/) corpus.

## CIC IGCE 2026

The [`cic/analysis`](./cic/analysis) directory contains the step-by-step analysis workflow presented at the CIC, exported from the documentation maintained in the laboratory's Notion workspace. A PDF copy of the poster presented at the **UNESP Undergraduate Research Conference (CIC)** is available in [`cic/poster/`](./cic/poster).

The poster summarizes the results of the GloBI data screening for insects and microorganisms and the evaluation of candidate LLMs.

## Questions or Contributions?

For suggestions, bug reports, or collaboration, feel free to open an [issue](https://github.com/Lab-LBMM/ic_lorena_silva/issues).
