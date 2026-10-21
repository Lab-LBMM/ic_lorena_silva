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
# 1. CONFIGURAÇÕES E CLIENTE HTTP
# ==========================================
MODELOS = ["qwen2.5", "llama3.1", "mistral-nemo", "deepseek-r1:1.5b", "gemma2"]
MODO_SHOTS = "few_shot"        # 'zero_shot' ou 'few_shot'
PIPELINE_DECOMPOSTO = False    # Se True, executa NER e RE em 2 chamadas separadas de API

TAMANHO_BLOCO = 2200           # Reduzido para evitar estouro em textos densos
SOBREPOSICAO_SENTENCAS = 1     
MAX_TENTATIVAS = 3             

client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="local",
    timeout=180.0               # Timeout estendido para 3 minutos
)

def obter_max_tokens(modelo_id):
    """Garante janela estendida para raciocínio e citações longas."""
    if "deepseek" in modelo_id.lower() or "r1" in modelo_id.lower():
        return 8192
    return 6000

def garantir_modelo_local(nome_modelo):
    print(f"\n[*] Preparando modelo '{nome_modelo}' no Ollama...")
    try:
        subprocess.run(["ollama", "pull", nome_modelo], check=True)
    except Exception as e:
        print(f"[!] Erro ao carregar modelo {nome_modelo}: {e}")
        exit(1)

# ==========================================
# 2. MODELOS DE DADOS (PYDANTIC)
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
# 3. PROMPTS E ONTOLOGIA UNIFORMIZADA
# ==========================================
ONTOLOGIA_BASE = """
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
    "P1_Ontologia_Rigida": f"You are an Ontology-Guided Information Extraction System specialized in myrmecology.\n{ONTOLOGIA_BASE}\nOutput ONLY valid JSON.",
    "P2_Raciocinio_Etapas": f"You are an expert bio-curator performing Two-Stage Information Extraction.\n{ONTOLOGIA_BASE}\nDocument reasoning in 'scratchpad' before outputting JSON.",
    "P3_Compreensao_Evidencia": f"You are a Machine Reading Comprehension model.\n{ONTOLOGIA_BASE}\nFor every relation, 'evidence' MUST be an exact quote from the text."
}

SHOTS_COMPLETOS = [
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

def construir_mensagens(texto_alvo, system_prompt):
    mensagens = [{"role": "system", "content": system_prompt}]
    if MODO_SHOTS == "few_shot":
        for shot in SHOTS_COMPLETOS:
            mensagens.append({"role": "user", "content": shot["user"]})
            mensagens.append({"role": "assistant", "content": shot["assistant"]})
    mensagens.append({"role": "user", "content": f"<text>{texto_alvo}</text>"})
    return mensagens

# ==========================================
# 4. TRATAMENTO DE TEXTO E EXECUÇÃO DA IA
# ==========================================
_SPLIT_SENTENCAS = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def dividir_texto_em_blocos(texto_completo, tamanho_maximo=TAMANHO_BLOCO, sobreposicao=SOBREPOSICAO_SENTENCAS):
    sentencas = []
    for paragrafo in re.split(r"\n\s*\n", texto_completo):
        for s in _SPLIT_SENTENCAS.split(paragrafo.strip()):
            s = s.strip()
            while len(s) > tamanho_maximo:
                sentencas.append(s[:tamanho_maximo])
                s = s[tamanho_maximo:]
            if s: sentencas.append(s)

    blocos, atual = [], []
    tamanho_atual = 0
    for s in sentencas:
        if atual and tamanho_atual + len(s) + 1 > tamanho_maximo:
            blocos.append(" ".join(atual))
            atual = atual[-sobreposicao:] if sobreposicao else []
            tamanho_atual = sum(len(x) + 1 for x in atual)
        atual.append(s)
        tamanho_atual += len(s) + 1
    if atual: blocos.append(" ".join(atual))
    return blocos

def limpar_resposta(conteudo):
    conteudo = re.sub(r"<think>.*?</think>", "", conteudo, flags=re.DOTALL)
    conteudo = re.sub(r"```(?:json)?", "", conteudo)
    ini, fim = conteudo.find("{"), conteudo.rfind("}")
    return conteudo[ini:fim + 1] if ini != -1 and fim != -1 else conteudo.strip()

def extrair_bloco(bloco, modelo_id, system_prompt):
    mensagens = construir_mensagens(bloco, system_prompt)
    info = {"tentativas": 0, "finish_reason": None, "erro": None, "tempo_s": 0.0}
    inicio = time.time()
    dados = None

    for tentativa in range(MAX_TENTATIVAS):
        info["tentativas"] = tentativa + 1
        try:
            resp = client.chat.completions.create(
                model=modelo_id,
                messages=mensagens,
                temperature=0.0 if tentativa == 0 else 0.2,
                max_tokens=obter_max_tokens(modelo_id),
                response_format=RESPONSE_FORMAT,
            )
        except Exception as e:
            info["erro"] = f"Erro na chamada: {e}"
            time.sleep(3)
            continue

        escolha = resp.choices[0]
        bruta = escolha.message.content or ""
        info["finish_reason"] = escolha.finish_reason

        if escolha.finish_reason == "length":
            info["erro"] = "resposta truncada (finish_reason=length)"
            break

        conteudo = limpar_resposta(bruta)
        try:
            dados = Extraction.model_validate_json(conteudo).model_dump()
            info["erro"] = None
            break
        except ValidationError as e:
            info["erro"] = f"JSON inválido: {str(e)[:150]}"
            continue

    info["tempo_s"] = round(time.time() - inicio, 2)
    return dados, info

def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def validar_bloco(dados, bloco):
    bloco_norm = _norm(bloco)
    ents_ok, descartados = [], []

    for e in dados.get("entities", []):
        if e.get("text") and e["text"] in bloco:
            ents_ok.append(e)
        else:
            motivo = "so_difere_em_formatacao" if e.get("text") and _norm(e["text"]) in bloco_norm else "ausente_no_texto"
            descartados.append({"tipo": "entidade", "motivo": motivo, "item": e})

    textos = {e["text"] for e in ents_ok}
    rels_ok = []
    for r in dados.get("relations", []):
        if not (r.get("evidence") and r["evidence"] in bloco):
            motivo = "evidencia_so_difere_em_formatacao" if r.get("evidence") and _norm(r["evidence"]) in bloco_norm else "evidencia_ausente_no_texto"
        elif r.get("subject") not in textos or r.get("object") not in textos:
            motivo = "sujeito_ou_objeto_sem_entidade_valida"
        else:
            rels_ok.append(r)
            continue
        descartados.append({"tipo": "relacao", "motivo": motivo, "item": r})

    descartes = {
        "entidades": len(dados.get("entities", [])) - len(ents_ok),
        "relacoes": len(dados.get("relations", [])) - len(rels_ok),
    }
    return {"entities": ents_ok, "relations": rels_ok}, descartes, descartados

def fundir_resultados_validados(lista_de_dados):
    entidades, relacoes = {}, {}
    for dados in lista_de_dados:
        for e in dados.get("entities", []):
            entidades[(e["text"], e["type"])] = e
        for r in dados.get("relations", []):
            chave = (r["subject"], r["relation"], r["object"], r["evidence"], r["negated"])
            relacoes[chave] = r
    return {"entities": list(entidades.values()), "relations": list(relacoes.values())}

# ==========================================
# 5. LOOP DA BUSCA EM GRADE (GRID SEARCH)
# ==========================================
def _texto_seguro(v): return "" if pd.isna(v) else str(v)

def _anexar_csv(caminho, linha):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    pd.DataFrame([linha]).to_csv(caminho, mode="a", header=not os.path.exists(caminho), index=False)

def executar_grid_search(caminho_db, limite_artigos=5):
    conn = sqlite3.connect(caminho_db)
    df_artigos = pd.read_sql_query(f"SELECT pmcid, pmid, title, abstract, content FROM pcw_literature LIMIT {int(limite_artigos)}", conn)
    conn.close()

    for modelo in MODELOS:
        garantir_modelo_local(modelo)
        _slug = modelo.replace(":", "_")

        for nome_prompt, texto_prompt in PROMPTS.items():
            print(f"\n" + "="*50)
            print(f"🚀 EXECUTANDO: Modelo [{modelo}] | Prompt [{nome_prompt}] | Shots [{MODO_SHOTS}]")
            print("="*50)

            pasta_saida = f"data/processed/grafos/{nome_prompt.split('_')[0].lower()}"
            ficheiro_saida = os.path.join(pasta_saida, f"grafos_extraidos_{_slug}_{nome_prompt}.csv")
            ficheiro_falhas = os.path.join(pasta_saida, f"grafos_com_falhas_{_slug}_{nome_prompt}.csv")
            ficheiro_metricas = os.path.join(pasta_saida, f"metricas_blocos_{_slug}_{nome_prompt}.jsonl")

            artigos_processados = set()
            if os.path.exists(ficheiro_saida) and os.path.getsize(ficheiro_saida) > 0:
                artigos_processados = set(pd.read_csv(ficheiro_saida)["pmcid"].astype(str))

            for idx, row in df_artigos.iterrows():
                pmcid = str(row["pmcid"])
                if pmcid in artigos_processados: continue

                print(f"\n[{idx+1}/{len(df_artigos)}] PMCID: {pmcid}")
                texto = f"{_texto_seguro(row['abstract'])}\n\n{_texto_seguro(row['content'])}".strip()
                blocos = dividir_texto_em_blocos(texto)

                validados, blocos_falhos = [], 0
                descartes_total = {"entidades": 0, "relacoes": 0}

                for i, bloco in enumerate(blocos):
                    print(f"  -> Bloco {i+1}/{len(blocos)} ({len(bloco)} chars)...", end="", flush=True)

                    if PIPELINE_DECOMPOSTO:
                        dados, erro_dec = extrair_duas_etapas_decompostas(bloco, modelo)
                        info = {"tempo_s": 0.0, "erro": erro_dec}
                    else:
                        dados, info = extrair_bloco(bloco, modelo, texto_prompt)

                    metrica = {"pmcid": pmcid, "bloco": i, "n_chars": len(bloco), **info}
                    if dados is None:
                        blocos_falhos += 1
                        print(f" [FALHOU: {info['erro']}]")
                    else:
                        ok, descartes, detalhes = validar_bloco(dados, bloco)
                        validados.append(ok)
                        for k in descartes_total: descartes_total[k] += descartes[k]
                        metrica.update({
                            "n_entidades": len(ok["entities"]),
                            "n_relacoes": len(ok["relations"]),
                            "descartes_entidades": descartes["entidades"],
                            "descartes_relacoes": descartes["relacoes"],
                        })
                        print(f" [OK: {info.get('tempo_s', 0)}s]")

                    with open(ficheiro_metricas, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metrica, ensure_ascii=False) + "\n")

                linha = {
                    "pmcid": pmcid,
                    "pmid": row["pmid"],
                    "titulo": row["title"],
                    "modelo": modelo,
                    "prompt": nome_prompt,
                    "grafo_json": json.dumps(fundir_resultados_validados(validados), ensure_ascii=False),
                    "n_blocos": len(blocos),
                    "blocos_falhos": blocos_falhos,
                    "descartes_entidades": descartes_total["entidades"],
                    "descartes_relacoes": descartes_total["relacoes"],
                }

                if blocos_falhos == 0:
                    _anexar_csv(ficheiro_saida, linha)
                    artigos_processados.add(pmcid)
                    print(f"  -> Salvo em {ficheiro_saida}")
                else:
                    _anexar_csv(ficheiro_falhas, linha)
                    print(f"  -> Salvo em falhas.")

if __name__ == "__main__":
    executar_grid_search(caminho_db="data/data_base/literature.db", limite_artigos=5)