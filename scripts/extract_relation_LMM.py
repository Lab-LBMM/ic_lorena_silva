import sqlite3
import pandas as pd
import time
import json
import os
import re
import subprocess
from openai import OpenAI

# ==========================================
# 1. CONFIGURAÇÕES GERAIS E AMBIENTE LOCAL
# ==========================================
# Altere este ID conforme o modelo que deseja testar (ex: "llama3.1", "qwen2.5", "gemma2", "deepseek-r1:8b")
MODELO_ID = "qwen2.5"
NUM_SHOTS = 3
FICHEIRO_SAIDA = f"grafos_extraidos_{MODELO_ID.replace(':', '_')}.csv"

def garantir_modelo_local(nome_modelo):
    """
    Verifica e descarrega os pesos do modelo via Ollama antes de iniciar a extração.
    """
    print(f"[*] A verificar/descarregar o modelo '{nome_modelo}' no Ollama...")
    try:
        # Executa o comando de pull no terminal de forma silenciosa
        subprocess.run(["ollama", "pull", nome_modelo], check=True)
        print(f"[*] Modelo '{nome_modelo}' pronto e carregado no cache local!\n")
    except FileNotFoundError:
        print("[!] Erro crítico: O Ollama não foi encontrado no sistema.")
        print("Instale o Ollama usando: curl -fsSL https://ollama.com/install.sh | sh")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[!] Erro ao tentar baixar os pesos do modelo: {e}")
        exit(1)

# Assegura o download dos pesos na hora de rodar
#garantir_modelo_local(MODELO_ID)

# Conecta ao servidor local do Ollama (localhost)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="local" # A chave é ignorada em execuções locais
)

SYSTEM_PROMPT = """You are an expert in Natural Language Processing and Ecology, specialized in Information Extraction from scientific literature regarding symbiotic relationships involving ants (Formicidae) and associated microorganisms.
Your task is to analyze the provided text and extract Named Entities (NER) and Ecological Relations (RE) structured as Subject-Relation-Object (SRO) triples.

Entity Rules: Classify entities ONLY into the following categories: [TAXON, CULTIVATED_FUNGUS, BEHAVIOR, LOCATION]. The "text" field must contain the exact substring present in the original text.

Relation Rules: Identify ecological connections between entities using ONLY the following predicates: [CULTIVATES, PARASITIZES, COMPETES_WITH, MUTUALISM_WITH]. The relation must have a logical direction (subject -> relation -> object).
The "evidence" field must contain the exact text snippet that justifies the relation.
The "negated" field must be boolean (true) ONLY if the text explicitly states that the interaction does NOT occur. Otherwise, it must be (false).

Output Restriction: Return ONLY a valid JSON object conforming strictly to the requested schema. Do not include greetings, explanatory text, markdown code blocks, or any text outside the JSON brackets. If no entities or relations are found, return empty lists []."""

SHOTS = [
    { 
        "user": "Text: 'The common ant Camponotus rufipes is host to the specialized parasite Ophiocordyceps camponoti-rufipedis in the Atlantic rainforests of Brazil.'",
        "assistant": '{"entities": [{"text": "Camponotus rufipes", "type": "TAXON"}, {"text": "Ophiocordyceps camponoti-rufipedis", "type": "TAXON"}, {"text": "Brazil", "type": "LOCATION"}], "relations": [{"subject": "Ophiocordyceps camponoti-rufipedis", "relation": "PARASITIZES", "object": "Camponotus rufipes", "evidence": "host to the specialized parasite", "negated": false}]}'
    },
    { 
        "user": "Text: 'Similarly, Ophiocordyceps kniphofioides infects Cephalotes atratus (Formicidae: Myrmicinae), Paraponera clavata , and Dinoponera longipes (Formicidae: Ponerinae) ants.'",
        "assistant": '{"entities": [{"text": "Ophiocordyceps kniphofioides", "type": "TAXON"}, {"text": "Cephalotes atratus", "type": "TAXON"}, {"text": "Paraponera clavata", "type": "TAXON"}, {"text": "Dinoponera longipes", "type": "TAXON"}], "relations": [{"subject": "Ophiocordyceps kniphofioides", "relation": "PARASITIZES", "object": "Cephalotes atratus", "evidence": "infects", "negated": false}, {"subject": "Ophiocordyceps kniphofioides", "relation": "PARASITIZES", "object": "Paraponera clavata", "evidence": "infects", "negated": false}, {"subject": "Ophiocordyceps kniphofioides", "relation": "PARASITIZES", "object": "Dinoponera longipes", "evidence": "infects", "negated": false}]}'
    },
    { 
        "user": "Text: 'One fungal symbiont associated with A. pubescens was isolated and identified as L. gongylophorus.'",
        "assistant": '{"entities": [{"text": "A. pubescens", "type": "TAXON"}, {"text": "L. gongylophorus", "type": "CULTIVATED_FUNGUS"}], "relations": [{"subject": "A. pubescens", "relation": "MUTUALISM_WITH", "object": "L. gongylophorus", "evidence": "fungal symbiont associated with", "negated": false}]}'
    }
]

# ==========================================
# 2. FUNÇÕES DE PROCESSAMENTO
# ==========================================
def construir_mensagens(texto_alvo):
    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(min(NUM_SHOTS, len(SHOTS))):
        mensagens.append({"role": "user", "content": SHOTS[i]["user"]})
        mensagens.append({"role": "assistant", "content": SHOTS[i]["assistant"]})
    mensagens.append({"role": "user", "content": f"Text: '{texto_alvo}'"})
    return mensagens

def dividir_texto_em_blocos(texto_completo, tamanho_maximo=15000):
    paragrafos = re.split(r'\n\s*\n', str(texto_completo))
    blocos = []
    bloco_atual = ""
    
    for p in paragrafos:
        if len(bloco_atual) + len(p) < tamanho_maximo:
            bloco_atual += p + "\n\n"
        else:
            if bloco_atual:
                blocos.append(bloco_atual.strip())
            bloco_atual = p + "\n\n"
            
    if bloco_atual.strip():
        blocos.append(bloco_atual.strip())
    return blocos

def extrair_com_retentativa(texto_alvo, max_tentativas=3):
    mensagens = construir_mensagens(texto_alvo)
    
    for tentativa in range(max_tentativas):
        try:
            response = client.chat.completions.create(
                model=MODELO_ID,
                messages=mensagens,
                temperature=0.0,
                max_tokens=1500, # <--- ADICIONE ESTA LINHA AQUI
                response_format={ "type": "json_object" } 
            )
            return response.choices[0].message.content
            
        except Exception as e:
            msg_erro = str(e).lower()
            if "timeout" in msg_erro or "connection" in msg_erro:
                espera = 5 * (tentativa + 1)
                print(f"    [!] O servidor local demorou a responder. Pausando por {espera}s (Tentativa {tentativa+1}/{max_tentativas})...")
                time.sleep(espera)
            else:
                print(f"    [!] Erro fatal no bloco: {e}")
                return None
    return None

def fundir_jsons_do_artigo(lista_de_jsons_texto):
    entidades_finais, relacoes_finais = [], []
    for resposta_texto in lista_de_jsons_texto:
        if resposta_texto:
            try:
                dados = json.loads(resposta_texto)
                if "entities" in dados:
                    entidades_finais.extend(dados["entities"])
                if "relations" in dados:
                    relacoes_finais.extend(dados["relations"])
            except json.JSONDecodeError:
                continue
                
    entidades_unicas = [json.loads(t) for t in set(json.dumps(d, sort_keys=True) for d in entidades_finais)]
    relacoes_unicas = [json.loads(t) for t in set(json.dumps(d, sort_keys=True) for d in relacoes_finais)]
    
    return {"entities": entidades_unicas, "relations": relacoes_unicas}

# ==========================================
# 3. ORQUESTRAÇÃO COM CHECKPOINTING
# ==========================================
def processar_banco_sqlite(caminho_db, limite_artigos=None, tamanho_lote=50):
    artigos_processados = set()
    if os.path.exists(FICHEIRO_SAIDA):
        df_existente = pd.read_csv(FICHEIRO_SAIDA)
        artigos_processados = set(df_existente['pmcid'].astype(str))
        print(f"Retomando progresso... {len(artigos_processados)} artigos já extraídos.")

    conn = sqlite3.connect(caminho_db)
    query = "SELECT pmcid, pmid, title, abstract, content FROM pcw_literature"
    if limite_artigos:
        query += f" LIMIT {limite_artigos}"
    
    df_iterator = pd.read_sql_query(query, conn, chunksize=tamanho_lote)
    print(f"Iniciando extração (Modelo Local: {MODELO_ID} | Shots: {NUM_SHOTS}).")
    
    contador = 0
    for df_chunk in df_iterator:
        for index, row in df_chunk.iterrows():
            contador += 1
            pmcid = str(row['pmcid'])
            
            if pmcid in artigos_processados:
                continue
                
            print(f"\n[{contador}] Processando PMCID: {pmcid}")
            
            texto_completo = f"{row['abstract']}\n\n{row['content']}"
            blocos = dividir_texto_em_blocos(texto_completo)
            resultados_blocos = []
            
            for i, bloco in enumerate(blocos):
                print(f"  -> Analisando bloco {i+1}/{len(blocos)} (Inferência na GPU Local)...")
                json_do_bloco = extrair_com_retentativa(bloco)
                resultados_blocos.append(json_do_bloco)
                
            json_final = fundir_jsons_do_artigo(resultados_blocos)
            
            novo_dado = pd.DataFrame([{
                "pmcid": pmcid,
                "pmid": row['pmid'],
                "titulo": row['title'],
                "modelo": MODELO_ID,
                "grafo_json": json.dumps(json_final)
            }])
            
            novo_dado.to_csv(FICHEIRO_SAIDA, mode='a', header=not os.path.exists(FICHEIRO_SAIDA), index=False)
            print(f"  -> Grafo salvo com sucesso!")

    conn.close()
    print(f"\nProcessamento 100% concluído! Resultados consolidados em: {FICHEIRO_SAIDA}")

if __name__ == "__main__":
    # Certifique-se de que o servidor do Ollama está a correr no seu sistema operativo.
    processar_banco_sqlite(caminho_db="data/data_base/literature.db", limite_artigos=5)