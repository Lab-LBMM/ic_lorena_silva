import pandas as pd
import json

def normalizar_texto(texto):
    """Remove espaços duplos e converte para minúsculas para evitar falsos erros de formatação."""
    return " ".join(str(texto).strip().lower().split())

def extrair_conjuntos(caminho_csv):
    """Lê o CSV e separa todas as Entidades e Relações em conjuntos matemáticos (sets)."""
    entidades = set()
    relacoes = set()
    
    try:
        df = pd.read_csv(caminho_csv)
        for _, row in df.iterrows():
            try:
                grafo = json.loads(row['grafo_json'])
                
                # 1. Extrair Entidades: Guarda como tupla (nome, tipo)
                for e in grafo.get('entities', []):
                    entidades.add((normalizar_texto(e['text']), normalizar_texto(e['type'])))
                    
                # 2. Extrair Relações: Guarda como tupla (sujeito, relacao, objeto)
                for r in grafo.get('relations', []):
                    relacoes.add((
                        normalizar_texto(r['subject']),
                        normalizar_texto(r['relation']),
                        normalizar_texto(r['object'])
                    ))
            except Exception:
                continue # Ignora linhas com erro de parse ou vazias
                
        return entidades, relacoes
    except FileNotFoundError:
        print(f"[Erro] Ficheiro não encontrado: {caminho_csv}")
        return set(), set()

def calcular_metricas(tp, fp, fn):
    """Calcula Precisão, Recall e F1-Score evitando divisão por zero."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def avaliar_modelo(caminho_gabarito, caminho_ia, nome_teste):
    print(f"\n" + "="*60)
    print(f"📊 RESULTADOS: {nome_teste}")
    print("="*60)
    
    gold_ents, gold_rels = extrair_conjuntos(caminho_gabarito)
    ai_ents, ai_rels = extrair_conjuntos(caminho_ia)
    
    if not gold_ents and not gold_rels:
        print("[!] Aviso: O Gabarito está vazio. Certifique-se de que o CSV possui a coluna 'grafo_json'.")
        return

    # --- CÁLCULO NER (ENTIDADES) ---
    tp_ent = len(gold_ents.intersection(ai_ents))
    fp_ent = len(ai_ents - gold_ents)
    fn_ent = len(gold_ents - ai_ents)
    p_ent, r_ent, f1_ent = calcular_metricas(tp_ent, fp_ent, fn_ent)

    # --- CÁLCULO RE (RELAÇÕES) ---
    tp_rel = len(gold_rels.intersection(ai_rels))
    fp_rel = len(ai_rels - gold_rels)
    fn_rel = len(gold_rels - ai_rels)
    p_rel, r_rel, f1_rel = calcular_metricas(tp_rel, fp_rel, fn_rel)

    # --- IMPRESSÃO DOS RESULTADOS ---
    print("\n--- 1. NER (Reconhecimento de Entidades) ---")
    print(f"✅ Verdadeiros Positivos (Acertos):    {tp_ent}")
    print(f"❌ Falsos Positivos (Alucinações):     {fp_ent}")
    print(f"⚠️ Falsos Negativos (Omissões):        {fn_ent}")
    print(f"-> Precisão: {p_ent:.2%} | Recall: {r_ent:.2%} | F1-Score: {f1_ent:.2%}")

    print("\n--- 2. RE (Extração de Relações) ---")
    print(f"✅ Verdadeiros Positivos (Acertos):    {tp_rel}")
    print(f"❌ Falsos Positivos (Alucinações):     {fp_rel}")
    print(f"⚠️ Falsos Negativos (Omissões):        {fn_rel}")
    print(f"-> Precisão: {p_rel:.2%} | Recall: {r_rel:.2%} | F1-Score: {f1_rel:.2%}\n")

if __name__ == "__main__":
    # O seu Gabarito (Gold Standard) deve ter a mesma estrutura de um ficheiro de saída gerado pela IA.
    ARQUIVO_GABARITO = "meu_gabarito_5_artigos.csv"
    
    # Exemplo de como chamar a avaliação para os ficheiros que a sua Grid Search gerou:
    avaliar_modelo(ARQUIVO_GABARITO, "grafos_extraidos_qwen2.5_P1_Ontologia_Rigida.csv", "Qwen 2.5 - P1 (Ditador)")
    avaliar_modelo(ARQUIVO_GABARITO, "grafos_extraidos_qwen2.5_P2_Raciocinio_Etapas.csv", "Qwen 2.5 - P2 (Raciocínio)")
    avaliar_modelo(ARQUIVO_GABARITO, "grafos_extraidos_qwen2.5_P3_Compreensao_Evidencia.csv", "Qwen 2.5 - P3 (Compreensão)")