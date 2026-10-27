import argparse
import csv
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def extract_sets(csv_path):
    entities = set()
    relations = set()

    if not os.path.exists(csv_path):
        print(f"[Error] File not found: {csv_path}")
        return set(), set()

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                graph_raw = (row.get("grafo_json") or "").strip()
                if not graph_raw:
                    continue

                try:
                    graph = json.loads(graph_raw, strict=False)

                    # 1. Extract Entities: (text, type)
                    for e in graph.get("entities") or []:
                        if isinstance(e, dict):
                            txt = normalize_text(e.get("text"))
                            entity_type = normalize_text(e.get("type"))
                            if txt:
                                entities.add((txt, entity_type))

                    # 2. Extract Relations: (subject, relation, object)
                    for r in graph.get("relations") or []:
                        if isinstance(r, dict):
                            subj = normalize_text(r.get("subject"))
                            rel = normalize_text(r.get("relation"))
                            obj = normalize_text(r.get("object"))
                            if subj and obj:
                                relations.add((subj, rel, obj))
                except (json.JSONDecodeError, TypeError):
                    continue

        return entities, relations
    except Exception as e:
        print(f"[Error] Failed to process {csv_path}: {e}")
        return set(), set()


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_model(gold_standard_path, ai_path, test_name):
    print("\n" + "=" * 60)
    print(f"RESULTS: {test_name}")
    print("=" * 60)

    if not os.path.exists(ai_path):
        print(f"JSON file does not exist: {ai_path}")
        return 0, 0, 0, 0, 0, 0

    gold_ents, gold_rels = extract_sets(gold_standard_path)
    ai_ents, ai_rels = extract_sets(ai_path)

    if not gold_ents and not gold_rels:
        print("The Gold Standard is empty or could not be read.")
        return 0, 0, 0, 0, 0, 0

    # NER (Entities)
    tp_ent = len(gold_ents.intersection(ai_ents))
    fp_ent = len(ai_ents - gold_ents)
    fn_ent = len(gold_ents - ai_ents)
    p_ent, r_ent, f1_ent = calculate_metrics(tp_ent, fp_ent, fn_ent)

    # RE (Relations)
    tp_rel = len(gold_rels.intersection(ai_rels))
    fp_rel = len(ai_rels - gold_rels)
    fn_rel = len(gold_rels - ai_rels)
    p_rel, r_rel, f1_rel = calculate_metrics(tp_rel, fp_rel, fn_rel)

    print("\n--- 1. NER (Named Entity Recognition) ---")
    print(f"True Positives (TP): {tp_ent} | False Positives (FP): {fp_ent} | False Negatives (FN): {fn_ent}")
    print(f"-> Precision: {p_ent:.2%} | Recall: {r_ent:.2%} | F1-Score: {f1_ent:.2%}")

    print("\n--- 2. RE (Relation Extraction) ---")
    print(f"True Positives (TP): {tp_rel} | False Positives (FP): {fp_rel} | False Negatives (FN): {fn_rel}")
    print(f"-> Precision: {p_rel:.2%} | Recall: {r_rel:.2%} | F1-Score: {f1_rel:.2%}\n")

    return p_ent, r_ent, f1_ent, p_rel, r_rel, f1_rel


def generate_metrics_chart(results_list, output_path="metrics_chart.png"):
    if not results_list:
        print("No metric data provided to generate the chart.")
        return

    df = pd.DataFrame(results_list)
    sns.set_theme(style="whitegrid", font="sans-serif")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    pink_palette = ["#FFB6C1", "#FF69B4", "#C71585", "#DB7093"]

    barplot = sns.barplot(
        data=df,
        x="Configuration",
        y="Value",
        hue="Metric",
        palette=pink_palette,
        edgecolor="#8B004F",
        linewidth=0.8,
        ax=ax,
    )

    plt.title("Ablation Study — Information Extraction Performance", fontsize=14, fontweight="bold", pad=18, color="#4A0E2E")
    plt.xlabel("Configuration (Model + Prompt)", fontsize=11, fontweight="bold", color="#4A0E2E", labelpad=10)
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold", color="#4A0E2E", labelpad=10)
    plt.ylim(0, 105)

    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            barplot.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
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

    legend = ax.legend(title="Metrics", frameon=True, facecolor="#FFF0F5", edgecolor="#FFB6C1")
    legend.get_title().set_color("#4A0E2E")
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Chart generated successfully at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Validation with NER and RE Metrics")
    
    parser.add_argument(
        "--gold-standard", "-g", 
        required=True, 
        help="Path to the Gold Standard CSV"
    )
    parser.add_argument(
        "--input", "-i", 
        nargs="+", 
        required=True, 
        help="One or more CSV files generated by the models/prompts"
    )
    parser.add_argument(
        "--output-chart", "-o", 
        default="metrics_chart.png", 
        help="Path to save the output chart (default: metrics_chart.png)"
    )

    args = parser.parse_args()

    metrics_history = []

    for ai_path in args.input:
        file_name = Path(ai_path).stem
        config_name = file_name.replace("grafos_extraidos_", "")

        p_ent, r_ent, f1_ent, p_rel, r_rel, f1_rel = evaluate_model(
            args.gold_standard, ai_path, config_name
        )

        if any([p_ent, r_ent, f1_ent, f1_rel]):
            metrics_history.extend([
                {"Configuration": config_name, "Metric": "Precision (NER)", "Value": p_ent * 100},
                {"Configuration": config_name, "Metric": "Recall (NER)", "Value": r_ent * 100},
                {"Configuration": config_name, "Metric": "F1-Score (NER)", "Value": f1_ent * 100},
                {"Configuration": config_name, "Metric": "F1-Score (RE)", "Value": f1_rel * 100},
            ])

    if metrics_history:
        generate_metrics_chart(metrics_history, output_path=args.output_chart)