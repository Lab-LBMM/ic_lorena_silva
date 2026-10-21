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
# 1. CONFIGURAÇÕES DA BUSCA EM GRADE (GRID SEARCH)
# ==========================================
# Lista de todos os modelos locais do Ollama a testar
MODELOS = ["deepseek-r1:1.5b"]

NUM_SHOTS = 3
TAMANHO_BLOCO = 3000           
SOBREPOSICAO_SENTENCAS = 1     
MAX_TOKENS = 4000              
MAX_TENTATIVAS = 3             

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

def garantir_modelo_local(nome_modelo):
    """Verifica e baixa o modelo no Ollama se não existir."""
    print(f"\n[*] Preparando ambiente para o modelo '{nome_modelo}'...")
    try:
        subprocess.run(["ollama", "pull", nome_modelo], check=True)
        print(f"[*] Modelo '{nome_modelo}' pronto e carregado na cache!\n")
    except FileNotFoundError:
        print("[!] Ollama não encontrado.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[!] Erro ao baixar o modelo: {e}")
        exit(1)

# ==========================================
# 2. FORMATO QUE A IA TEM QUE DEVOLVER (PYDANTIC)
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
    scratchpad: Optional[str] = None # Permite que modelos de raciocínio pensem sem quebrar o código
    entities: list[Entity]
    relations: list[Relation]

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "extraction", "schema": Extraction.model_json_schema()},
}

# ==========================================
# 3. DICIONÁRIO DOS SYSTEM PROMPTS
# ==========================================
PROMPTS = {
    "P1_Ontologia_Rigida": """You are an Ontology-Guided Information Extraction System specialized in myrmecology and microbial symbiosis. Your task is to map text spans to a strict ontology.

# ONTOLOGY DEFINITIONS & EXCLUSIONS
1. TAXON: Biological entities (ants, bacteria, yeasts, plants). EXCLUDE: Names of researchers, locations, chemical compounds, or anatomy.
2. CULTIVATED_FUNGUS: The specific fungal crop maintained by ants. EXCLUDE: Pathogenic or environmental fungi.
3. BEHAVIOR: Specific actions of ants or microbes (e.g., weeding, grooming). EXCLUDE: General biological processes.
4. LOCATION: Specific geographic areas or biomes.

# RELATION ONTOLOGY (SRO Triples)
- [TAXON] -> CULTIVATES -> [CULTIVATED_FUNGUS]
- [TAXON] -> PARASITIZES -> [TAXON or CULTIVATED_FUNGUS]
- [TAXON/CULTIVATED_FUNGUS] -> MUTUALISM_WITH -> [TAXON/CULTIVATED_FUNGUS]
- [TAXON] -> COMPETES_WITH -> [TAXON]

# RULES
Output ONLY a valid JSON object. Entities must be exact substrings.
{
  "entities": [{"text": "string", "type": "string"}],
  "relations": [{"subject": "string", "relation": "string", "object": "string", "evidence": "string", "negated": boolean}]
}
""",

    "P2_Raciocinio_Etapas": """You are an expert bio-curator performing a Two-Stage Information Extraction on ecological texts about Formicidae. 

# STEP 1: NAMED ENTITY RECOGNITION (NER)
Identify entities belonging strictly to: [TAXON, CULTIVATED_FUNGUS, BEHAVIOR, LOCATION]. 

# STEP 2: RELATION EXTRACTION (RE)
Determine if any pairs are explicitly linked by: [CULTIVATES, PARASITIZES, COMPETES_WITH, MUTUALISM_WITH].

# OUTPUT FORMAT
Document your Step 1 and Step 2 thought process inside the "scratchpad" key before generating the JSON arrays. Output ONLY valid JSON.
{
  "scratchpad": "Step 1: Found X. Step 2: X parasitizes Y because...",
  "entities": [{"text": "exact_span", "type": "TAXON"}],
  "relations": [{"subject": "exact_span", "relation": "PARASITIZES", "object": "exact_span", "evidence": "quote", "negated": false}]
}
""",

    "P3_Compreensao_Evidencia": """You are an advanced Machine Reading Comprehension (MRC) model extracting ecological relationships.

# EXTRACTION QUERIES
Q1: Which exact text spans represent entities? (Types: TAXON, CULTIVATED_FUNGUS, BEHAVIOR, LOCATION).
Q2: Does the text explicitly describe a relationship? (Relations: CULTIVATES, PARASITIZES, COMPETES_WITH, MUTUALISM_WITH).

# STRICT EVIDENCE RULE
For every relation, you MUST populate the "evidence" field with the exact sequential words from the text that prove the relationship. If you cannot find an exact quote, DO NOT extract the relation. Set "negated" to true ONLY if explicitly denied.

# RESPONSE
Output only valid JSON. No markdown.
{
  "entities": [{"text": "span", "type": "TYPE"}],
  "relations": [{"subject": "span", "relation": "TYPE", "object": "span", "evidence": "exact quote", "negated": false}]
}
"""
}

SHOTS = [
    {
        "user": "The common ant Camponotus rufipes is host to the specialized parasite Ophiocordyceps camponoti-rufipedis in the Atlantic rainforests of Brazil.",
        "assistant": '{"scratchpad": "Found Camponotus rufipes (TAXON), Ophiocordyceps camponoti-rufipedis (TAXON), and Brazil (LOCATION). The text states Ophiocordyceps is a parasite to Camponotus.", "entities": [{"text": "Camponotus rufipes", "type": "TAXON"}, {"text": "Ophiocordyceps camponoti-rufipedis", "type": "TAXON"}, {"text": "Brazil", "type": "LOCATION"}], "relations": [{"subject": "Ophiocordyceps camponoti-rufipedis", "relation": "PARASITIZES", "object": "Camponotus rufipes", "evidence": "host to the specialized parasite", "negated": false}]}',
    }
]

def formatar_entrada(texto):
    return f"<text>{texto}</text>"

def construir_mensagens(texto_alvo, system_prompt):
    mensagens = [{"role": "system", "content": system_prompt}]
    for shot in SHOTS[:NUM_SHOTS]:
        mensagens.append({"role": "user", "content": formatar_entrada(shot["user"])})
        mensagens.append({"role": "assistant", "content": shot["assistant"]})
    mensagens.append({"role": "user", "content": formatar_entrada(texto_alvo)})
    return mensagens

# =======================
# 4. QUEBRAR O TEXTO
# =======================
_SPLIT_SENTENCAS = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def dividir_texto_em_blocos(texto_completo, tamanho_maximo=TAMANHO_BLOCO, sobreposicao=SOBREPOSICAO_SENTENCAS):
    sentencas = []
    for paragrafo in re.split(r"\n\s*\n", texto_completo):
        for s in _SPLIT_SENTENCAS.split(paragrafo.strip()):
            s = s.strip()
            while len(s) > tamanho_maximo:
                sentencas.append(s[:tamanho_maximo])
                s = s[tamanho_maximo:]
            if s:
                sentencas.append(s)

    blocos, atual = [], []
    tamanho_atual = 0
    for s in sentencas:
        if atual and tamanho_atual + len(s) + 1 > tamanho_maximo:
            blocos.append(" ".join(atual))
            atual = atual[-sobreposicao:] if sobreposicao else []
            tamanho_atual = sum(len(x) + 1 for x in atual)
        atual.append(s)
        tamanho_atual += len(s) + 1
    if atual:
        blocos.append(" ".join(atual))
    return blocos

# ==========================================
# 5. MANDA PRA IA, LIMPA O LIXO E JUNTA
# ==========================================
def limpar_resposta(conteudo):
    conteudo = re.sub(r"<think>.*?</think>", "", conteudo, flags=re.DOTALL)
    conteudo = re.sub(r"```(?:json)?", "", conteudo)
    ini, fim = conteudo.find("{"), conteudo.rfind("}")
    return conteudo[ini:fim + 1] if ini != -1 and fim != -1 else conteudo.strip()

def extrair_bloco(bloco, modelo_id, system_prompt, max_tentativas=MAX_TENTATIVAS):
    mensagens = construir_mensagens(bloco, system_prompt)
    info = {"tentativas": 0, "finish_reason": None, "erro": None, "tempo_s": 0.0}
    inicio = time.time()
    dados = None

    for tentativa in range(max_tentativas):
        info["tentativas"] = tentativa + 1
        try:
            resp = client.chat.completions.create(
                model=modelo_id,
                messages=mensagens,
                temperature=0.0 if tentativa == 0 else 0.2,
                max_tokens=MAX_TOKENS,
                response_format=RESPONSE_FORMAT,
            )
        except Exception as e:
            msg = str(e).lower()
            if "timeout" in msg or "connection" in msg:
                espera = 5 * (tentativa + 1)
                info["erro"] = f"servidor lento/indisponível: {e}"
                print(f"    [!] Servidor demorou. Pausando {espera}s (tentativa {tentativa + 1}/{max_tentativas})...")
                time.sleep(espera)
                continue
            info["erro"] = f"erro fatal: {e}"
            break

        escolha = resp.choices[0]
        bruta = escolha.message.content or ""
        info["finish_reason"] = escolha.finish_reason

        if escolha.finish_reason == "length":
            info["erro"] = "resposta truncada (finish_reason=length)"
            info["resposta_bruta"] = bruta[:1500]
            break
        conteudo = limpar_resposta(bruta)
        try:
            dados = Extraction.model_validate_json(conteudo).model_dump()
            info["erro"] = None
            info.pop("resposta_bruta", None)
            break
        except ValidationError as e:
            info["erro"] = f"JSON/schema inválido: {str(e)[:200]}"
            info["resposta_bruta"] = bruta[:1500]
            continue

    info["tempo_s"] = round(time.time() - inicio, 2)
    return dados, info

def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def validar_bloco(dados, bloco):
    bloco_norm = _norm(bloco)
    ents_ok, descartados = [], []

    for e in dados.get("entities", []):
        if e["text"] and e["text"] in bloco:
            ents_ok.append(e)
        else:
            motivo = "so_difere_em_formatacao" if e["text"] and _norm(e["text"]) in bloco_norm else "ausente_no_texto"
            descartados.append({"tipo": "entidade", "motivo": motivo, "item": e})

    textos = {e["text"] for e in ents_ok}
    rels_ok = []
    for r in dados.get("relations", []):
        if not (r["evidence"] and r["evidence"] in bloco):
            motivo = "evidencia_so_difere_em_formatacao" if r["evidence"] and _norm(r["evidence"]) in bloco_norm else "evidencia_ausente_no_texto"
        elif r["subject"] not in textos or r["object"] not in textos:
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
# 6. FUNÇÃO PRINCIPAL DA BUSCA EM GRADE
# ==========================================
def _texto_seguro(valor):
    return "" if pd.isna(valor) else str(valor)

def _anexar_csv(caminho, linha):
    pd.DataFrame([linha]).to_csv(caminho, mode="a", header=not os.path.exists(caminho), index=False)

def executar_grid_search(caminho_db, limite_artigos=30):
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando GRID SEARCH no banco de dados...")
    
    # 1. Carregar os artigos do Gabarito para a memória
    conn = sqlite3.connect(caminho_db)
    query = f"SELECT pmcid, pmid, title, abstract, content FROM pcw_literature LIMIT {int(limite_artigos)}"
    df_artigos = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"-> {len(df_artigos)} artigos carregados para teste cruzado.")

    # 2. Percorrer os modelos
    for modelo in MODELOS:
        garantir_modelo_local(modelo)
        _slug = modelo.replace(":", "_")
        
        # 3. Percorrer os prompts
        for nome_prompt, texto_prompt in PROMPTS.items():
            print(f"\n" + "="*50)
            print(f"EXECUTANDO COMBINAÇÃO: Modelo [{modelo}] | Prompt [{nome_prompt}]")
            print("="*50)
            
            ficheiro_saida = f"grafos_extraidos_{_slug}_{nome_prompt}.csv"
            ficheiro_falhas = f"grafos_com_falhas_{_slug}_{nome_prompt}.csv"
            ficheiro_metricas = f"metricas_blocos_{_slug}_{nome_prompt}.jsonl"

            # Ler processados apenas para esta combinação específica
            artigos_processados = set()
            if os.path.exists(ficheiro_saida) and os.path.getsize(ficheiro_saida) > 0:
                artigos_processados = set(pd.read_csv(ficheiro_saida)["pmcid"].astype(str))

            contador = 0
            for _, row in df_artigos.iterrows():
                contador += 1
                pmcid = str(row["pmcid"])
                if pmcid in artigos_processados:
                    continue

                print(f"\n[{contador}/{len(df_artigos)}] Processando PMCID: {pmcid}")
                texto = f"{_texto_seguro(row['abstract'])}\n\n{_texto_seguro(row['content'])}".strip()
                blocos = dividir_texto_em_blocos(texto)

                validados, blocos_falhos = [], 0
                descartes_total = {"entidades": 0, "relacoes": 0}

                for i, bloco in enumerate(blocos):
                    print(f"  -> Bloco {i + 1}/{len(blocos)} ({len(bloco)} chars)...", end="", flush=True)
                    dados, info = extrair_bloco(bloco, modelo, texto_prompt)

                    metrica = {"pmcid": pmcid, "bloco": i, "n_chars": len(bloco), **info}
                    if dados is None:
                        blocos_falhos += 1
                        print(f" [FALHOU: {info['erro']}]")
                    else:
                        ok, descartes, detalhes = validar_bloco(dados, bloco)
                        validados.append(ok)
                        for k in descartes_total:
                            descartes_total[k] += descartes[k]
                        metrica.update({
                            "n_entidades": len(ok["entities"]),
                            "n_relacoes": len(ok["relations"]),
                            "descartes_entidades": descartes["entidades"],
                            "descartes_relacoes": descartes["relacoes"],
                            "descartados": detalhes,
                        })
                        print(f" [OK: {info['tempo_s']}s]")
                        
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
                    print(f"  -> Grafo salvo! ({len(validados)} blocos fundidos)")
                else:
                    _anexar_csv(ficheiro_falhas, linha)
                    print(f"  -> {blocos_falhos}/{len(blocos)} blocos falharam; salvo em falhas.")

if __name__ == "__main__":
    # Testaremos 5 artigos de acordo com a nossa estratégia metodológica
    executar_grid_search(caminho_db="data/data_base/literature.db", limite_artigos=5)

#tempo --> tá demorando 45min aprox. por modelo pra rodar 15 requisições