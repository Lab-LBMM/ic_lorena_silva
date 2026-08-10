# Fine-tuning Large Language Models for the Extraction of Ecological Relationships Involving Microorganisms and Ants

This repository contains the code, data, and documentation for the Undergraduate Research Project **“Fine-tuning large language models for the extraction of ecological relationships involving microorganisms and ants”** (FAPESP 2026/12637-3).

The project aims to evaluate and fine-tune large language models (LLMs) for the automatic extraction of ecological relationships between ants (Formicidae) and microorganisms from scientific literature, building a reproducible pipeline that spans from the curation of raw data (GloBI, BioC-PMC) to the construction of a knowledge graph of the identified interactions.

## Team

* **Undergraduate Research Fellow:** Lorena Silva (Biological Sciences, UNESP Rio Claro)
* **Advisor:** Prof. Dr. Renato Augusto Corrêa dos Santos (CEIS/IB/UNESP Rio Claro)
* **External Collaborator:** Prof. Dr. Ruben Interian (Institute of Computing/UNICAMP)

## Methodology

The project follows a three-phase methodology:

1. **Baseline Evaluation** — selection and screening of interaction data (GloBI) and evaluation of candidate LLMs based on eligibility criteria (license, availability of model weights, support for LoRA/QLoRA, computational feasibility, active maintenance, and technical documentation).
2. **Fine-tuning (LoRA/QLoRA)** — supervised fine-tuning of the selected model(s) for relationship extraction from narrative text (BioC-PMC corpus), using annotations based on the SRO (Subject–Relation–Object) scheme.
3. **Generalization Testing and Knowledge Graph Construction** — evaluation of the fine-tuned model on unseen data and organization of the extracted relationships into a knowledge graph.

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

## Documentation

https://github.com/Lab-LBMM/ic_lorena_silva/git

## Acknowledgments

This work was carried out with support from the São Paulo Research Foundation (FAPESP), Brazil (grant numbers 2024/19418-0 and 2026/12637-3).

## Questions or Contributions?

For suggestions, bug reports, or collaboration, feel free to open an [issue](https://github.com/Lab-LBMM/ic_lorena_silva/issues).
