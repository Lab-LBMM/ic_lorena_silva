import sqlite3
import pandas as pd
import time
import json
import os
import re
import subprocess
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ==========================================
# 1. CONFIGURAÇÕES GERAIS E AMBIENTE LOCAL
# ==========================================
MODELO_ID = "gemma2"   # ex: "llama3.1", "qwen2.5", "gemma2"
NUM_SHOTS = 3
TAMANHO_BLOCO = 3000           # caracteres; blocos curtos combinam com os shots
SOBREPOSICAO_SENTENCAS = 1     # nº de sentenças repetidas entre blocos consecutivos
MAX_TOKENS = 4000              # modelos de raciocínio (r1) precisam de bem mais (ex: 12000)
MAX_TENTATIVAS = 3

_slug = MODELO_ID.replace(":", "_")
FICHEIRO_SAIDA = f"grafos_extraidos_{_slug}.csv"          # só artigos 100% concluídos
FICHEIRO_FALHAS = f"grafos_com_falhas_{_slug}.csv"        # artigos com bloco(s) falho(s)
FICHEIRO_METRICAS = f"metricas_blocos_{_slug}.jsonl"      # 1 linha por bloco


def garantir_modelo_local(nome_modelo):
    print(f"[*] A verificar/descarregar o modelo '{nome_modelo}' no Ollama...")
    try:
        subprocess.run(["ollama", "pull", nome_modelo], check=True)
        print(f"[*] Modelo '{nome_modelo}' pronto!\n")
    except FileNotFoundError:
        print("[!] Ollama não encontrado. Instale com: curl -fsSL https://ollama.com/install.sh | sh")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[!] Erro ao baixar o modelo: {e}")
        exit(1)

# garantir_modelo_local(MODELO_ID)

# ATENÇÃO: o contexto (num_ctx) do Ollama NÃO é definido aqui. Configure via Modelfile
# (PARAMETER num_ctx 8192) ou variável de ambiente, senão o prompt pode ser truncado em silêncio.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

# ==========================================
# 2. SCHEMA (structured output)
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
    entities: list[Entity]
    relations: list[Relation]


# Se o Ollama da sua versão rejeitar json_schema, troque por {"type": "json_object"}
# (a validação Pydantic continua protegendo o pipeline).
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "extraction", "schema": Extraction.model_json_schema()},
}

# ==========================================
# 3. PROMPT E SHOTS
# ==========================================
SYSTEM_PROMPT = """You are an expert in Natural Language Processing and Ecology, specialized in Information Extraction from scientific literature regarding symbiotic relationships involving ants (Formicidae) and associated microorganisms.
Your task is to analyze the provided text and extract Named Entities (NER) and Ecological Relations (RE) structured as Subject-Relation-Object (SRO) triples.
The input text is always delimited by <text></text> tags.

Entity Rules: Classify entities ONLY into the following categories: [TAXON, CULTIVATED_FUNGUS, BEHAVIOR, LOCATION]. The "text" field must contain the exact substring present in the original text.

Relation Rules: Identify ecological connections between entities using ONLY the following predicates: [CULTIVATES, PARASITIZES, COMPETES_WITH, MUTUALISM_WITH]. The relation must have a logical direction (subject -> relation -> object). The "subject" and "object" fields must be exactly the "text" of an extracted entity.
The "evidence" field must contain the exact text snippet that justifies the relation.
The "negated" field must be boolean (true) ONLY if the text explicitly states that the interaction does NOT occur. Otherwise, it must be (false).

Output Restriction: Return ONLY a valid JSON object conforming strictly to the requested schema. Do not include greetings, explanatory text, markdown code blocks, or any text outside the JSON brackets. If no entities or relations are found, return empty lists []."""

# "user" guarda só o texto cru; o delimitador é aplicado por formatar_entrada().
# TODO: revisar o shot 3 (MUTUALISM_WITH vs CULTIVATES) e adicionar shots com
# negated=true, COMPETES_WITH, BEHAVIOR e resultado vazio.
SHOTS = [
    {
        "user": "The common ant Camponotus rufipes is host to the specialized parasite Ophiocordyceps camponoti-rufipedis in the Atlantic rainforests of Brazil.",
        "assistant": '{"entities": [{"text": "Camponotus rufipes", "type": "TAXON"}, {"text": "Ophiocordyceps camponoti-rufipedis", "type": "TAXON"}, {"text": "Brazil", "type": "LOCATION"}], "relations": [{"subject": "Ophiocordyceps camponoti-rufipedis", "relation": "PARASITIZES", "object": "Camponotus rufipes", "evidence": "host to the specialized parasite", "negated": false}]}',
    },
    {
        "user": "Similarly, Ophiocordyceps kniphofioides infects Cephalotes atratus (Formicidae: Myrmicinae), Paraponera clavata , and Dinoponera longipes (Formicidae: Ponerinae) ants.",
        "assistant": '{"entities": [{"text": "Ophiocordyceps kniphofioides", "type": "TAXON"}, {"text": "Cephalotes atratus", "type": "TAXON"}, {"text": "Paraponera clavata", "type": "TAXON"}, {"text": "Dinoponera longipes", "type": "TAXON"}], "relations": [{"subject": "Ophiocordyceps kniphofioides", "relation": "PARASITIZES", "object": "Cephalotes atratus", "evidence": "infects", "negated": false}, {"subject": "Ophiocordyceps kniphofioides", "relation": "PARASITIZES", "object": "Paraponera clavata", "evidence": "infects", "negated": false}, {"subject": "Ophiocordyceps kniphofioides", "relation": "PARASITIZES", "object": "Dinoponera longipes", "evidence": "infects", "negated": false}]}',
    },
    {
        "user": "One fungal symbiont associated with A. pubescens was isolated and identified as L. gongylophorus.",
        "assistant": '{"entities": [{"text": "A. pubescens", "type": "TAXON"}, {"text": "L. gongylophorus", "type": "CULTIVATED_FUNGUS"}], "relations": [{"subject": "A. pubescens", "relation": "MUTUALISM_WITH", "object": "L. gongylophorus", "evidence": "fungal symbiont associated with", "negated": false}]}',
    },
]


def formatar_entrada(texto):
    return f"<text>{texto}</text>"


def construir_mensagens(texto_alvo):
    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in SHOTS[:NUM_SHOTS]:
        mensagens.append({"role": "user", "content": formatar_entrada(shot["user"])})
        mensagens.append({"role": "assistant", "content": shot["assistant"]})
    mensagens.append({"role": "user", "content": formatar_entrada(texto_alvo)})
    return mensagens

# ==========================================
# 4. DIVISÃO DO TEXTO (por sentenças, com sobreposição)
# ==========================================
_SPLIT_SENTENCAS = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def dividir_texto_em_blocos(texto_completo, tamanho_maximo=TAMANHO_BLOCO,
                            sobreposicao=SOBREPOSICAO_SENTENCAS):
    sentencas = []
    for paragrafo in re.split(r"\n\s*\n", texto_completo):
        for s in _SPLIT_SENTENCAS.split(paragrafo.strip()):
            s = s.strip()
            # sentença gigante: corta duro para respeitar o limite
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
# 5. EXTRAÇÃO, VALIDAÇÃO E FUSÃO
# ==========================================
def limpar_resposta(conteudo):
    """Remove <think>, cercas markdown e texto fora do primeiro {...} final."""
    conteudo = re.sub(r"<think>.*?</think>", "", conteudo, flags=re.DOTALL)
    conteudo = re.sub(r"```(?:json)?", "", conteudo)
    ini, fim = conteudo.find("{"), conteudo.rfind("}")
    return conteudo[ini:fim + 1] if ini != -1 and fim != -1 else conteudo.strip()


def extrair_bloco(bloco, max_tentativas=MAX_TENTATIVAS):
    """Retorna (dados_validados_pelo_schema | None, info_do_bloco)."""
    mensagens = construir_mensagens(bloco)
    info = {"tentativas": 0, "finish_reason": None, "erro": None, "tempo_s": 0.0}
    inicio = time.time()
    dados = None

    for tentativa in range(max_tentativas):
        info["tentativas"] = tentativa + 1
        try:
            resp = client.chat.completions.create(
                model=MODELO_ID,
                messages=mensagens,
                temperature=0.0 if tentativa == 0 else 0.2,  # varia um pouco nas retentativas
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
            info["resposta_bruta"] = bruta[:1500]   # para diagnosticar (think longo? loop?)
            break  # repetir com o mesmo limite não resolve

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
    """
    Mantém só itens ancorados no texto do bloco.
    Retorna (dados_ok, contagem_de_descartes, lista_detalhada_de_descartados).
    O 'motivo' distingue erro de cópia (formatação) de alucinação.
    """
    bloco_norm = _norm(bloco)
    ents_ok, descartados = [], []

    for e in dados["entities"]:
        if e["text"] and e["text"] in bloco:
            ents_ok.append(e)
        else:
            motivo = ("so_difere_em_formatacao"
                      if e["text"] and _norm(e["text"]) in bloco_norm
                      else "ausente_no_texto")
            descartados.append({"tipo": "entidade", "motivo": motivo, "item": e})

    textos = {e["text"] for e in ents_ok}
    rels_ok = []
    for r in dados["relations"]:
        if not (r["evidence"] and r["evidence"] in bloco):
            motivo = ("evidencia_so_difere_em_formatacao"
                      if r["evidence"] and _norm(r["evidence"]) in bloco_norm
                      else "evidencia_ausente_no_texto")
        elif r["subject"] not in textos or r["object"] not in textos:
            motivo = "sujeito_ou_objeto_sem_entidade_valida"
        else:
            rels_ok.append(r)
            continue
        descartados.append({"tipo": "relacao", "motivo": motivo, "item": r})

    descartes = {
        "entidades": len(dados["entities"]) - len(ents_ok),
        "relacoes": len(dados["relations"]) - len(rels_ok),
    }
    return {"entities": ents_ok, "relations": rels_ok}, descartes, descartados


def fundir_resultados_validados(lista_de_dados):
    """Une resultados já validados de vários blocos, removendo duplicatas."""
    entidades, relacoes = {}, {}
    for dados in lista_de_dados:
        for e in dados["entities"]:
            entidades[(e["text"], e["type"])] = e
        for r in dados["relations"]:
            chave = (r["subject"], r["relation"], r["object"], r["evidence"], r["negated"])
            relacoes[chave] = r
    return {"entities": list(entidades.values()), "relations": list(relacoes.values())}

# ==========================================
# 6. ORQUESTRAÇÃO COM CHECKPOINTING SEGURO
# ==========================================
def _texto_seguro(valor):
    return "" if pd.isna(valor) else str(valor)


def _anexar_csv(caminho, linha):
    pd.DataFrame([linha]).to_csv(caminho, mode="a", header=not os.path.exists(caminho), index=False)


def _ler_processados():
    if os.path.exists(FICHEIRO_SAIDA) and os.path.getsize(FICHEIRO_SAIDA) > 0:
        return set(pd.read_csv(FICHEIRO_SAIDA)["pmcid"].astype(str))
    return set()


def processar_banco_sqlite(caminho_db, limite_artigos=None, tamanho_lote=50):
    artigos_processados = _ler_processados()
    if artigos_processados:
        print(f"Retomando... {len(artigos_processados)} artigos já concluídos.")

    conn = sqlite3.connect(caminho_db)
    query = "SELECT pmcid, pmid, title, abstract, content FROM pcw_literature"
    if limite_artigos:
        query += f" LIMIT {int(limite_artigos)}"

    print(f"Iniciando extração (Modelo: {MODELO_ID} | Shots: {NUM_SHOTS}).")
    contador = 0
    for df_chunk in pd.read_sql_query(query, conn, chunksize=tamanho_lote):
        for _, row in df_chunk.iterrows():
            contador += 1
            pmcid = str(row["pmcid"])
            if pmcid in artigos_processados:
                continue

            print(f"\n[{contador}] Processando PMCID: {pmcid}")
            texto = f"{_texto_seguro(row['abstract'])}\n\n{_texto_seguro(row['content'])}".strip()
            blocos = dividir_texto_em_blocos(texto)

            validados, blocos_falhos = [], 0
            descartes_total = {"entidades": 0, "relacoes": 0}

            for i, bloco in enumerate(blocos):
                print(f"  -> Bloco {i + 1}/{len(blocos)} ({len(bloco)} caracteres)...")
                dados, info = extrair_bloco(bloco)

                metrica = {"pmcid": pmcid, "bloco": i, "n_chars": len(bloco), **info}
                if dados is None:
                    blocos_falhos += 1
                    print(f"    [!] Bloco falhou: {info['erro']}")
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
                with open(FICHEIRO_METRICAS, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrica, ensure_ascii=False) + "\n")

            linha = {
                "pmcid": pmcid,
                "pmid": row["pmid"],
                "titulo": row["title"],
                "modelo": MODELO_ID,
                "grafo_json": json.dumps(fundir_resultados_validados(validados), ensure_ascii=False),
                "n_blocos": len(blocos),
                "blocos_falhos": blocos_falhos,
                "descartes_entidades": descartes_total["entidades"],
                "descartes_relacoes": descartes_total["relacoes"],
            }

            if blocos_falhos == 0:
                _anexar_csv(FICHEIRO_SAIDA, linha)
                artigos_processados.add(pmcid)
                print("  -> Grafo salvo com sucesso!")
            else:
                # não entra no checkpoint principal: será reprocessado na próxima execução
                _anexar_csv(FICHEIRO_FALHAS, linha)
                print(f"  -> {blocos_falhos}/{len(blocos)} blocos falharam; salvo em {FICHEIRO_FALHAS} "
                      f"e NÃO marcado como concluído.")

    conn.close()
    print(f"\nFim. Concluídos em: {FICHEIRO_SAIDA} | Falhas em: {FICHEIRO_FALHAS} | Métricas: {FICHEIRO_METRICAS}")


if __name__ == "__main__":
    processar_banco_sqlite(caminho_db="data/data_base/literature.db", limite_artigos=5)