import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

BASE_ONTOLOGY = """
# ONTOLOGY DEFINITIONS & STRICT EXCLUSIONS
1. TAXON: Biological entities (ants, bacteria, yeasts, plants, insects).
   -> EXCLUDE: Names of researchers, geographic locations, chemical compounds, anatomical parts.
2. CULTIVATED_FUNGUS: Specific fungal crop maintained/farmed by ants.
3. BEHAVIOR: Specific actions of ants or microbes (e.g., weeding, grooming).
4. LOCATION: Geographic areas, biomes, or sampling locations.

# RELATION ONTOLOGY (SRO Triples)
- [TAXON] -> CULTIVATES -> [CULTIVATED_FUNGUS]
- [TAXON] -> PARASITIZES -> [TAXON or CULTIVATED_FUNGUS]
- [TAXON/CULTIVATED_FUNGUS] -> MUTUALISM_WITH -> [TAXON/CULTIVATED_FUNGUS]
- [TAXON] -> COMPETES_WITH -> [TAXON]
"""

PROMPTS = {
    "P1_Strict_Ontology": f"You are an Ontology-Guided Information Extraction System specialized in myrmecology.\n{BASE_ONTOLOGY}\nOutput ONLY valid JSON.",
    "P2_Step_Reasoning": f"You are an expert bio-curator performing Two-Stage Information Extraction.\n{BASE_ONTOLOGY}\nDocument reasoning in 'scratchpad' before outputting JSON.",
    "P3_Evidence_Comprehension": f"You are a Machine Reading Comprehension model.\n{BASE_ONTOLOGY}\nFor every relation, 'evidence' MUST be an exact quote from the text.",
}

FEW_SHOT_EXAMPLES = [
    {
        "user": "<text>Fungus-growing ants (Attini) maintain an obligate mutualism with Leucoagaricus gongylophorus, which they cultivate for food. Parasitic microfungi in the genus Escovopsis frequently infect these gardens.</text>",
        "assistant": json.dumps({
            "scratchpad": "Identified Attini (TAXON), Leucoagaricus gongylophorus (CULTIVATED_FUNGUS), Escovopsis (TAXON).",
            "entities": [
                {"text": "Attini", "type": "TAXON"},
                {"text": "Leucoagaricus gongylophorus", "type": "CULTIVATED_FUNGUS"},
                {"text": "Escovopsis", "type": "TAXON"},
            ],
            "relations": [
                {
                    "subject": "Attini",
                    "relation": "CULTIVATES",
                    "object": "Leucoagaricus gongylophorus",
                    "evidence": "cultivate for food",
                    "negated": False,
                },
                {
                    "subject": "Escovopsis",
                    "relation": "PARASITIZES",
                    "object": "Leucoagaricus gongylophorus",
                    "evidence": "parasitic microfungi in the genus Escovopsis frequently infect these gardens",
                    "negated": False,
                },
            ],
        }),
    }
]


class Entity(BaseModel):
    text: str
    type: Literal["TAXON", "CULTIVATED_FUNGUS", "BEHAVIOR", "LOCATION"]


class Relation(BaseModel):
    subject: str
    relation: Literal["CULTIVATES", "PARASITIZES", "COMPETES_WITH", "MUTUALISM_WITH"]
    object: str
    evidence: str
    negated: bool = False


class NEROutput(BaseModel):
    scratchpad: Optional[str] = None
    entities: list[Entity]


class REOutput(BaseModel):
    scratchpad: Optional[str] = None
    relations: list[Relation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run information extraction ablation grid search across LLMs and prompts."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.1", "qwen2.5", "gemma2", "mistral-nemo", "deepseek-r1:1.5b"],
        help="Models to evaluate (default: llama3.1 qwen2.5 gemma2 mistral-nemo deepseek-r1:1.5b)",
    )
    parser.add_argument(
        "--shot-modes",
        nargs="+",
        choices=["zero_shot", "few_shot"],
        default=["zero_shot", "few_shot"],
        help="Shot modes to evaluate (default: zero_shot few_shot)",
    )
    parser.add_argument(
        "--db-path",
        default="data/data_base/literature.db",
        help="Path to the SQLite database (default: data/data_base/literature.db)",
    )
    parser.add_argument(
        "--article-limit",
        type=int,
        default=5,
        help="Maximum number of articles to query from the database (default: 5)",
    )
    return parser.parse_args()


def build_messages(system_prompt: str, user_text: str, shot_mode: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if shot_mode == "few_shot":
        for shot in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": shot["user"]})
            messages.append({"role": "assistant", "content": shot["assistant"]})
    messages.append({"role": "user", "content": f"<text>{user_text}</text>"})
    return messages


def extract_ner_decomposed(chunk: str, model: str, system_prompt: str, shot_mode: str) -> list[dict]:
    sys_ner = f"{system_prompt}\nTask: Extract ONLY valid biological entities."
    messages = build_messages(sys_ner, chunk, shot_mode)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            timeout=30.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ner", "schema": NEROutput.model_json_schema()},
            },
        )
        data = NEROutput.model_validate_json(resp.choices[0].message.content)
        return [e.model_dump() for e in data.entities]
    except Exception:
        return []


def extract_re_decomposed(
    chunk: str, entities: list[dict], model: str, system_prompt: str, shot_mode: str
) -> list[dict]:
    if not entities:
        return []

    sys_re = (
        f"{system_prompt}\n"
        f"Task: Extract ONLY valid relationships between these pre-extracted entities: {json.dumps(entities)}"
    )
    messages = build_messages(sys_re, chunk, shot_mode)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            timeout=30.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "re", "schema": REOutput.model_json_schema()},
            },
        )
        data = REOutput.model_validate_json(resp.choices[0].message.content)
        return [r.model_dump() for r in data.relations]
    except Exception:
        return []


def split_text(text: str, max_length: int = 2500) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < max_length:
            current += f"\n\n{p}"
        else:
            if current:
                chunks.append(current.strip())
            current = p
    if current:
        chunks.append(current.strip())
    return chunks


def process_chunk_with_timeout(
    chunk: str, model: str, prompt_text: str, shot_mode: str, limit_seconds: int = 60
) -> Optional[tuple[list[dict], list[dict]]]:
    def _task():
        ents = extract_ner_decomposed(chunk, model, prompt_text, shot_mode)
        rels = extract_re_decomposed(chunk, ents, model, prompt_text, shot_mode)
        return ents, rels

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_task)
        try:
            return future.result(timeout=limit_seconds)
        except TimeoutError:
            return None


def run_ablation_grid_search(
    models: list[str], shot_modes: list[str], db_path: str, article_limit: int
):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT pmcid, abstract, content FROM pcw_literature LIMIT ?",
        conn,
        params=(article_limit,),
    )
    conn.close()

    for model in models:
        slug = model.replace(":", "_")

        for shot_mode in shot_modes:
            for prompt_name, prompt_text in PROMPTS.items():
                print(f"\nRUNNING: Model [{model}] | Prompt [{prompt_name}] | Mode [{shot_mode}]")

                output_folder = f"data/processed/graphs/{prompt_name.split('_')[0].lower()}"
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(
                    output_folder, f"extracted_graphs_{slug}_{prompt_name}_{shot_mode}.csv"
                )

                results = []
                start_time = time.time()

                for idx, row in df.iterrows():
                    pmcid = row["pmcid"]
                    text = f"{row['abstract'] or ''}\n\n{row['content'] or ''}".strip()
                    chunks = split_text(text)

                    all_entities, all_relations = [], []

                    print(f"PMCID {pmcid} ({idx + 1}/{len(df)}) - {len(chunks)} chunks total:")

                    for c_i, chunk in enumerate(chunks):
                        t0 = time.time()

                        ents = extract_ner_decomposed(chunk, model, prompt_text, shot_mode)
                        rels = extract_re_decomposed(chunk, ents, model, prompt_text, shot_mode)

                        dt = round(time.time() - t0, 2)
                        print(
                            f"  - Chunk {c_i + 1}/{len(chunks)}: {len(ents)} ents, {len(rels)} rels ({dt}s)"
                        )

                        all_entities.extend(ents)
                        all_relations.extend(rels)

                    graph = {"entities": all_entities, "relations": all_relations}
                    results.append({
                        "pmcid": pmcid,
                        "model": model,
                        "prompt": prompt_name,
                        "shot_mode": shot_mode,
                        "graph_json": json.dumps(graph, ensure_ascii=False),
                    })

                pd.DataFrame(results).to_csv(output_file, index=False)
                total_time = round(time.time() - start_time, 2)
                print(f"  -> Completed in {total_time}s | Saved to: {output_file}")


if __name__ == "__main__":
    args = parse_args()
    run_ablation_grid_search(
        models=args.models,
        shot_modes=args.shot_modes,
        db_path=args.db_path,
        article_limit=args.article_limit,
    )