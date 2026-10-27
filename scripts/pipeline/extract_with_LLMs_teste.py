import sqlite3
import pandas as pd
import time
import json
import os
import re
import subprocess
from typing import Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ==========================================
# 1. CONFIGURATIONS AND HTTP CLIENT
# ==========================================
MODELS = ["qwen2.5", "llama3.1", "mistral-nemo", "deepseek-r1:1.5b", "gemma2"]
SHOT_MODE = "few_shot"         # 'zero_shot' or 'few_shot'
DECOMPOSED_PIPELINE = False    # If True, runs NER and RE in 2 separate API calls

BLOCK_SIZE = 2200              # Reduced to avoid overflow in dense texts
SENTENCE_OVERLAP = 1     
MAX_ATTEMPTS = 3             

client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="local",
    timeout=180.0               # Extended timeout to 3 minutes
)

def get_max_tokens(model_id):
    """Ensures extended window for reasoning and long citations."""
    if "deepseek" in model_id.lower() or "r1" in model_id.lower():
        return 8192
    return 6000

def ensure_local_model(model_name):
    print(f"\n[*] Preparing model '{model_name}' in Ollama...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
    except Exception as e:
        print(f"[!] Error loading model {model_name}: {e}")
        exit(1)

# ==========================================
# 2. DATA MODELS (PYDANTIC)
# ==========================================
class Entity(BaseModel):
    text: str
    type: Literal["TAXON", "CULTIVATED_FUNGUS", "BEHAVIOR", "LOCATION"]

class Relation(BaseModel):
    subject: str
    relation: Literal["CULTIVATES", "PARASITIZES", "COMPETES_WITH", "MUTUALISM_WITH"]
    object: str
    evidence: str
    negated: bool

class Extraction(BaseModel):
    scratchpad: Optional[str] = None
    entities: list[Entity]
    relations: list[Relation]

class NERExtraction(BaseModel):
    entities: list[Entity]

class REExtraction(BaseModel):
    relations: list[Relation]

RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "extraction", "schema": Extraction.model_json_schema()}}
RESPONSE_FORMAT_NER = {"type": "json_schema", "json_schema": {"name": "ner_extraction", "schema": NERExtraction.model_json_schema()}}
RESPONSE_FORMAT_RE = {"type": "json_schema", "json_schema": {"name": "re_extraction", "schema": REExtraction.model_json_schema()}}

# ==========================================
# 3. PROMPTS AND UNIFIED ONTOLOGY
# ==========================================
BASE_ONTOLOGY = """
# ONTOLOGY DEFINITIONS & STRICT EXCLUSIONS
1. TAXON: Biological entities (ants, bacteria, yeasts, plants, insects).
   -> EXCLUDE: Names of researchers, geographic locations, chemical compounds (e.g., antibiotics), anatomical parts, or laboratory equipment/software.
2. CULTIVATED_FUNGUS: The specific fungal crop maintained, farmed, or eaten by ants/beetles.
   -> EXCLUDE: Pathogenic/environmental fungi, seeds, eggs, or general diets.
3. BEHAVIOR: Specific actions of ants or microbes (e.g., weeding, grooming, foraging).
   -> EXCLUDE: General biological concepts or anatomical terms.
4. LOCATION: Specific geographic areas, biomes, or sampling locations.

# RELATION ONTOLOGY (SRO Triples)
- [TAXON] -> CULTIVATES -> [CULTIVATED_FUNGUS]
- [TAXON] -> PARASITIZES -> [TAXON or CULTIVATED_FUNGUS]
- [TAXON/CULTIVATED_FUNGUS] -> MUTUALISM_WITH -> [TAXON/CULTIVATED_FUNGUS]
- [TAXON] -> COMPETES_WITH -> [TAXON]
"""

PROMPTS = {
    "P1_Strict_Ontology": f"You are an Ontology-Guided Information Extraction System specialized in myrmecology.\n{BASE_ONTOLOGY}\nOutput ONLY valid JSON.",
    "P2_Step_Reasoning": f"You are an expert bio-curator performing Two-Stage Information Extraction.\n{BASE_ONTOLOGY}\nDocument reasoning in 'scratchpad' before outputting JSON.",
    "P3_Evidence_Comprehension": f"You are a Machine Reading Comprehension model.\n{BASE_ONTOLOGY}\nFor every relation, 'evidence' MUST be an exact quote from the text."
}

FULL_SHOTS = [
    {
        "user": "<text>Fungus-growing ants (Attini) maintain an obligate mutualism with Leucoagaricus gongylophorus, which they cultivate for food. Parasitic microfungi in the genus Escovopsis frequently infect these gardens.</text>",
        "assistant": json.dumps({
            "scratchpad": "Identified Attini, Leucoagaricus gongylophorus, Escovopsis.",
            "entities": [{"text": "Attini", "type": "TAXON"}, {"text": "Leucoagaricus gongylophorus", "type": "CULTIVATED_FUNGUS"}, {"text": "Escovopsis", "type": "TAXON"}],
            "relations": [
                {"subject": "Attini", "relation": "CULTIVATES", "object": "Leucoagaricus gongylophorus", "evidence": "maintain an obligate mutualism with Leucoagaricus gongylophorus, which they cultivate for food", "negated": False},
                {"subject": "Escovopsis", "relation": "PARASITIZES", "object": "Leucoagaricus gongylophorus", "evidence": "parasitic microfungi in the genus Escovopsis frequently infect these gardens", "negated": False}
            ]
        })
    }
]

def build_messages(target_text, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    if SHOT_MODE == "few_shot":
        for shot in FULL_SHOTS:
            messages.append({"role": "user", "content": shot["user"]})
            messages.append({"role": "assistant", "content": shot["assistant"]})
    messages.append({"role": "user", "content": f"<text>{target_text}</text>"})
    return messages

# ==========================================
# 4. TEXT PROCESSING AND AI EXECUTION
# ==========================================
_SPLIT_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def split_text_into_blocks(full_text, max_size=BLOCK_SIZE, overlap=SENTENCE_OVERLAP):
    sentences = []
    for paragraph in re.split(r"\n\s*\n", full_text):
        for s in _SPLIT_SENTENCES.split(paragraph.strip()):
            s = s.strip()
            while len(s) > max_size:
                sentences.append(s[:max_size])
                s = s[max_size:]
            if s: sentences.append(s)

    blocks, current = [], []
    current_size = 0
    for s in sentences:
        if current and current_size + len(s) + 1 > max_size:
            blocks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current_size = sum(len(x) + 1 for x in current)
        current.append(s)
        current_size += len(s) + 1
    if current: blocks.append(" ".join(current))
    return blocks

def clean_response(content):
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"```(?:json)?", "", content)
    start, end = content.find("{"), content.rfind("}")
    return content[start:end + 1] if start != -1 and end != -1 else content.strip()

def extract_block(block, model_id, system_prompt):
    messages = build_messages(block, system_prompt)
    info = {"attempts": 0, "finish_reason": None, "error": None, "time_s": 0.0}
    start_time = time.time()
    data = None

    for attempt in range(MAX_ATTEMPTS):
        info["attempts"] = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.0 if attempt == 0 else 0.2,
                max_tokens=get_max_tokens(model_id),
                response_format=RESPONSE_FORMAT,
            )
        except Exception as e:
            info["error"] = f"Call error: {e}"
            time.sleep(3)
            continue

        choice = resp.choices[0]
        raw = choice.message.content or ""
        info["finish_reason"] = choice.finish_reason

        if choice.finish_reason == "length":
            info["error"] = "truncated response (finish_reason=length)"
            break

        content = clean_response(raw)
        try:
            data = Extraction.model_validate_json(content).model_dump()
            info["error"] = None
            break
        except ValidationError as e:
            info["error"] = f"Invalid JSON: {str(e)[:150]}"
            continue

    info["time_s"] = round(time.time() - start_time, 2)
    return data, info

def extract_two_stages_decomposed(block, model_id):
    """
    Executes 2 independent API calls:
    Stage 1: Identifies Entities only (NER).
    Stage 2: Receives the block + Entities extracted in Stage 1 and identifies Relations (RE).
    """
    # --- STAGE 1: NER ---
    prompt_ner = f"Extract ONLY valid biological entities from the text.\n{BASE_ONTOLOGY}\nOutput ONLY valid JSON."
    messages_ner = [{"role": "system", "content": prompt_ner}, {"role": "user", "content": f"<text>{block}</text>"}]
    
    try:
        resp_ner = client.chat.completions.create(
            model=model_id,
            messages=messages_ner,
            temperature=0.0,
            max_tokens=get_max_tokens(model_id),
            response_format=RESPONSE_FORMAT_NER
        )
        content_ner = clean_response(resp_ner.choices[0].message.content or "")
        data_ner = NERExtraction.model_validate_json(content_ner).model_dump()
    except Exception as e:
        return None, f"Stage 1 (NER) failed: {e}"

    found_entities = data_ner.get("entities", [])
    if not found_entities:
        return {"entities": [], "relations": []}, None

    # --- STAGE 2: RE ---
    prompt_re = f"""Given the following text AND the pre-extracted entities, identify ONLY valid relationships between these entities.

Extracted Entities: {json.dumps(found_entities)}

{BASE_ONTOLOGY}
Output ONLY valid JSON.
"""
    messages_re = [{"role": "system", "content": prompt_re}, {"role": "user", "content": f"<text>{block}</text>"}]

    try:
        resp_re = client.chat.completions.create(
            model=model_id,
            messages=messages_re,
            temperature=0.0,
            max_tokens=get_max_tokens(model_id),
            response_format=RESPONSE_FORMAT_RE
        )
        content_re = clean_response(resp_re.choices[0].message.content or "")
        data_re = REExtraction.model_validate_json(content_re).model_dump()
    except Exception as e:
        return None, f"Stage 2 (RE) failed: {e}"

    return {
        "entities": found_entities,
        "relations": data_re.get("relations", [])
    }, None

def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def validate_block(data, block):
    block_norm = _norm(block)
    ents_ok, discarded = [], []

    for e in data.get("entities", []):
        if e.get("text") and e["text"] in block:
            ents_ok.append(e)
        else:
            reason = "differs_only_in_formatting" if e.get("text") and _norm(e["text"]) in block_norm else "missing_in_text"
            discarded.append({"type": "entity", "reason": reason, "item": e})

    texts = {e["text"] for e in ents_ok}
    rels_ok = []
    for r in data.get("relations", []):
        if not (r.get("evidence") and r["evidence"] in block):
            reason = "evidence_differs_only_in_formatting" if r.get("evidence") and _norm(r["evidence"]) in block_norm else "evidence_missing_in_text"
        elif r.get("subject") not in texts or r.get("object") not in texts:
            reason = "subject_or_object_without_valid_entity"
        else:
            rels_ok.append(r)
            continue
        discarded.append({"type": "relation", "reason": reason, "item": r})

    discards = {
        "entities": len(data.get("entities", [])) - len(ents_ok),
        "relations": len(data.get("relations", [])) - len(rels_ok),
    }
    return {"entities": ents_ok, "relations": rels_ok}, discards, discarded

def merge_validated_results(data_list):
    entities, relations = {}, {}
    for data in data_list:
        for e in data.get("entities", []):
            entities[(e["text"], e["type"])] = e
        for r in data.get("relations", []):
            key = (r["subject"], r["relation"], r["object"], r["evidence"], r["negated"])
            relations[key] = r
    return {"entities": list(entities.values()), "relations": list(relations.values())}

# ==========================================
# 5. GRID SEARCH LOOP
# ==========================================
def _safe_text(v): return "" if pd.isna(v) else str(v)

def _append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def run_grid_search(db_path, article_limit=5):
    conn = sqlite3.connect(db_path)
    df_articles = pd.read_sql_query(f"SELECT pmcid, pmid, title, abstract, content FROM pcw_literature LIMIT {int(article_limit)}", conn)
    conn.close()

    for model in MODELS:
        ensure_local_model(model)
        _slug = model.replace(":", "_")

        for prompt_name, prompt_text in PROMPTS.items():
            print(f"\n" + "="*50)
            print(f"🚀 RUNNING: Model [{model}] | Prompt [{prompt_name}] | Shots [{SHOT_MODE}]")
            print("="*50)

            output_folder = f"data/processed/graphs/{prompt_name.split('_')[0].lower()}"
            os.makedirs(output_folder, exist_ok=True)
            
            output_file = os.path.join(output_folder, f"extracted_graphs_{_slug}_{prompt_name}.csv")
            failed_file = os.path.join(output_folder, f"failed_graphs_{_slug}_{prompt_name}.csv")
            metrics_file = os.path.join(output_folder, f"block_metrics_{_slug}_{prompt_name}.jsonl")

            processed_articles = set()
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                processed_articles = set(pd.read_csv(output_file)["pmcid"].astype(str))

            for idx, row in df_articles.iterrows():
                pmcid = str(row["pmcid"])
                if pmcid in processed_articles: continue

                print(f"\n[{idx+1}/{len(df_articles)}] PMCID: {pmcid}")
                text = f"{_safe_text(row['abstract'])}\n\n{_safe_text(row['content'])}".strip()
                blocks = split_text_into_blocks(text)

                validated, failed_blocks = [], 0
                total_discards = {"entities": 0, "relations": 0}

                for i, block in enumerate(blocks):
                    print(f"  -> Block {i+1}/{len(blocks)} ({len(block)} chars)...", end="", flush=True)

                    if DECOMPOSED_PIPELINE:
                        data, dec_error = extract_two_stages_decomposed(block, model)
                        info = {"time_s": 0.0, "error": dec_error}
                    else:
                        data, info = extract_block(block, model, prompt_text)

                    metric = {"pmcid": pmcid, "block": i, "n_chars": len(block), **info}
                    if data is None:
                        failed_blocks += 1
                        print(f" [FAILED: {info['error']}]")
                    else:
                        ok, discards, details = validate_block(data, block)
                        validated.append(ok)
                        for k in total_discards: total_discards[k] += discards[k]
                        metric.update({
                            "n_entities": len(ok["entities"]),
                            "n_relations": len(ok["relations"]),
                            "discarded_entities": discards["entities"],
                            "discarded_relations": discards["relations"],
                        })
                        print(f" [OK: {info.get('time_s', 0)}s]")

                    with open(metrics_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metric, ensure_ascii=False) + "\n")

                row_data = {
                    "pmcid": pmcid,
                    "pmid": row["pmid"],
                    "title": row["title"],
                    "model": model,
                    "prompt": prompt_name,
                    "graph_json": json.dumps(merge_validated_results(validated), ensure_ascii=False),
                    "n_blocks": len(blocks),
                    "failed_blocks": failed_blocks,
                    "discarded_entities": total_discards["entities"],
                    "discarded_relations": total_discards["relations"],
                }

                if failed_blocks == 0:
                    _append_csv(output_file, row_data)
                    processed_articles.add(pmcid)
                    print(f"  -> Saved to {output_file}")
                else:
                    _append_csv(failed_file, row_data)
                    print(f"  -> Saved to failed files.")

if __name__ == "__main__":
    run_grid_search(db_path="data/data_base/literature.db", article_limit=5)