import argparse
import csv
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def normalizar_texto(texto):
    """Remove espaços extras, quebras de linha e converte para minúsculas de forma segura."""
    if texto is None:
        return ""
    return " ".join(str(texto).strip().lower().split())


def extrair_conjuntos(caminho_csv):
    """Lê o CSV usando csv.DictReader com suporte a JSONs multilinha."""
    entidades = set()
    relacoes = set()

    if not os.path.exists(caminho_csv):
        print(f"[Erro] Ficheiro não encontrado: {caminho_csv}")
        return set(), set()

    try:
        with open(caminho_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grafo_raw = (row.get("grafo_json") or "").strip()
                if not grafo_raw:
                    continue

                try:
                    grafo = json.loads(grafo_raw, strict=False)

                    # 1. Extrair Entidades: (texto, tipo)
                    for e in grafo.get("entities") or []:
                        if isinstance(e, dict):
                            txt = normalizar_texto(e.get("text"))
                            tipo = normalizar_texto(e.get("type"))
                            if txt:
                                entidades.add((txt, tipo))

                    # 2. Extrair Relações: (sujeito, relação, objeto)
                    for r in grafo.get("relations") or []:
                        if isinstance(r, dict):
                            subj = normalizar_texto(r.get("subject"))
                            rel = normalizar_texto(r.get("relation"))
                            obj = normalizar_texto(r.get("object"))
                            if subj and obj:
                                relacoes.add((subj, rel, obj))
                except (json.JSONDecodeError, TypeError):
                    continue

        return entidades, relacoes
    except Exception as e:
        print(f"[Erro] Falha ao processar {caminho_csv}: {e}")
        return set(), set()


def calcular_metricas(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def avaliar_modelo(caminho_gabarito, caminho_ia, nome_teste):
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS: {nome_teste}")
    print("=" * 60)

    if not os.path.exists(caminho_ia):
        print(f"⚠️ Arquivo da IA não existe: {caminho_ia}")
        return 0, 0, 0, 0, 0, 0

    gold_ents, gold_rels = extrair_conjuntos(caminho_gabarito)
    ai_ents, ai_rels = extrair_conjuntos(caminho_ia)

    if not gold_ents and not gold_rels:
        print("[!] Aviso: O Gabarito está vazio ou não pôde ser lido.")
        return 0, 0, 0, 0, 0, 0

    # NER (Entidades)
    tp_ent = len(gold_ents.intersection(ai_ents))
    fp_ent = len(ai_ents - gold_ents)
    fn_ent = len(gold_ents - ai_ents)
    p_ent, r_ent, f1_ent = calcular_metricas(tp_ent, fp_ent, fn_ent)

    # RE (Relações)
    tp_rel = len(gold_rels.intersection(ai_rels))
    fp_rel = len(ai_rels - gold_rels)
    fn_rel = len(gold_rels - ai_rels)
    p_rel, r_rel, f1_rel = calcular_metricas(tp_rel, fp_rel, fn_rel)

    print("\n--- 1. NER (Reconhecimento de Entidades) ---")
    print(f"✅ Acertos (TP): {tp_ent} | ❌ Alucinações (FP): {fp_ent} | ⚠️ Omissões (FN): {fn_ent}")
    print(f"-> Precisão: {p_ent:.2%} | Recall: {r_ent:.2%} | F1-Score: {f1_ent:.2%}")

    print("\n--- 2. RE (Extração de Relações) ---")
    print(f"✅ Acertos (TP): {tp_rel} | ❌ Alucinações (FP): {fp_rel} | ⚠️ Omissões (FN): {fn_rel}")
    print(f"-> Precisão: {p_rel:.2%} | Recall: {r_rel:.2%} | F1-Score: {f1_rel:.2%}\n")

    return p_ent, r_ent, f1_ent, p_rel, r_rel, f1_rel


def gerar_grafico_metricas_rosa(lista_resultados, caminho_saida="grafico_metricas_rosa.png"):
    """Gera o gráfico comparativo em tons de rosa com os resultados passados."""
    if not lista_resultados:
        print("[!] Nenhum dado de métrica fornecido para gerar o gráfico.")
        return

    df = pd.DataFrame(lista_resultados)
    sns.set_theme(style="whitegrid", font="sans-serif")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    paleta_rosa = ["#FFB6C1", "#FF69B4", "#C71585", "#DB7093"]

    barplot = sns.barplot(
        data=df,
        x="Configuração",
        y="Valor",
        hue="Métrica",
        palette=paleta_rosa,
        edgecolor="#8B004F",
        linewidth=0.8,
        ax=ax,
    )

    plt.title("Ablation Study — Desempenho de Extração de Informação", fontsize=14, fontweight="bold", pad=18, color="#4A0E2E")
    plt.xlabel("Configuração (Modelo + Prompt)", fontsize=11, fontweight="bold", color="#4A0E2E", labelpad=10)
    plt.ylabel("Pontuação (%)", fontsize=11, fontweight="bold", color="#4A0E2E", labelpad=10)
    plt.ylim(0, 105)

    for p in barplot.patches:
        altura = p.get_height()
        if altura > 0:
            barplot.annotate(
                f"{altura:.1f}%",
                (p.get_x() + p.get_width() / 2.0, altura),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4A0E2E",
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xticks(rotation=15, ha="right", fontsize=9, color="#4A0E2E")
    plt.yticks(color="#4A0E2E")

    legend = ax.legend(title="Métricas", frameon=True, facecolor="#FFF0F5", edgecolor="#FFB6C1")
    legend.get_title().set_color("#4A0E2E")
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✨ Gráfico rosa gerado com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação de Grafos de Conhecimento com Métricas de NER e RE")
    
    parser.add_argument(
        "--gabarito", "-g", 
        required=True, 
        help="Caminho para o CSV do Gabarito (Gold Standard)"
    )
    parser.add_argument(
        "--entrada", "-i", 
        nargs="+", 
        required=True, 
        help="Um ou mais arquivos CSVs gerados pelos modelos/prompts"
    )
    parser.add_argument(
        "--saida-grafico", "-o", 
        default="grafico_metricas_rosa.png", 
        help="Caminho para salvar o gráfico de saída (padrão: grafico_metricas_rosa.png)"
    )

    args = parser.parse_args()

    historico_metricas = []

    for caminho_ia in args.entrada:
        # Pega apenas o nome limpo do arquivo para a legenda do gráfico
        nome_arquivo = Path(caminho_ia).stem
        nome_config = nome_arquivo.replace("grafos_extraidos_", "")

        p_ent, r_ent, f1_ent, p_rel, r_rel, f1_rel = avaliar_modelo(
            args.gabarito, caminho_ia, nome_config
        )

        if any([p_ent, r_ent, f1_ent, f1_rel]):
            historico_metricas.extend([
                {"Configuração": nome_config, "Métrica": "Precision (NER)", "Valor": p_ent * 100},
                {"Configuração": nome_config, "Métrica": "Recall (NER)", "Valor": r_ent * 100},
                {"Configuração": nome_config, "Métrica": "F1-Score (NER)", "Valor": f1_ent * 100},
                {"Configuração": nome_config, "Métrica": "F1-Score (RE)", "Valor": f1_rel * 100},
            ])

    if historico_metricas:
        gerar_grafico_metricas_rosa(historico_metricas, caminho_saida=args.saida_grafico)