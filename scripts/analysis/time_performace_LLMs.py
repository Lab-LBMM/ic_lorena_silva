import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

SLOW_THRESHOLD = 60.0
TIMEOUT_LINE = 90.0
DPI = 300
IMG_FORMAT = "png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse LLM benchmark logs and generate comparative charts."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="benchmark.log",
        help="Path to the input benchmark log (or txt) file (default: benchmark.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default="benchmark_charts",
        help="Directory where charts and summary CSV will be saved (default: benchmark_charts)",
    )
    return parser.parse_args()


def parse_log_file(filepath: str) -> pd.DataFrame:
    """Parses the benchmark log file and returns a block-level DataFrame."""
    re_run = re.compile(
        r"RUNNING:\s*Modelo\s*\[(?P<modelo>.*?)\]\s*\|\s*Prompt\s*\[(?P<prompt>.*?)\]\s*\|\s*Mode\s*\[(?P<mode>.*?)\]"
    )
    re_pmcid = re.compile(r"PMCID\s+(?P<pmcid>PMC\d+)")
    re_bloco = re.compile(
        r"Bloco\s+(?P<bloco_idx>\d+)/(?P<total_blocos>\d+):\s*(?P<ents>\d+)\s*ents,\s*(?P<rels>\d+)\s*rels\s*\((?P<tempo>[\d.]+)s\)"
    )

    records = []
    current_model = None
    current_prompt = None
    current_mode = None
    current_pmcid = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m_run = re_run.search(line)
            if m_run:
                current_model = m_run.group("modelo").strip()
                current_prompt = m_run.group("prompt").strip()
                current_mode = m_run.group("mode").strip()
                continue

            m_pmcid = re_pmcid.search(line)
            if m_pmcid:
                current_pmcid = m_pmcid.group("pmcid").strip()
                continue

            m_bloco = re_bloco.search(line)
            if m_bloco and current_model:
                ents = int(m_bloco.group("ents"))
                rels = int(m_bloco.group("rels"))
                elapsed = float(m_bloco.group("tempo"))

                if ents == 0 and rels == 0 and elapsed >= SLOW_THRESHOLD:
                    status = "Timeout / Slow Failure (>=60s)"
                elif ents == 0 and rels == 0:
                    status = "Fast Empty (<60s)"
                elif ents > 0 and rels == 0 and elapsed >= SLOW_THRESHOLD:
                    status = "Truncated (Entities Only >=60s)"
                else:
                    status = "Success (Ents & Rels)"

                records.append({
                    "Model": current_model,
                    "Prompt": current_prompt,
                    "Mode": current_mode,
                    "PMCID": current_pmcid,
                    "Block": int(m_bloco.group("bloco_idx")),
                    "Entities": ents,
                    "Relations": rels,
                    "Total_Items": ents + rels,
                    "Time_s": elapsed,
                    "Status": status,
                })

    return pd.DataFrame(records)


def generate_report_and_charts(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    summary = (
        df.groupby(["Model", "Prompt", "Mode"], as_index=False)
        .agg(
            Blocks=("Block", "count"),
            Total_Entities=("Entities", "sum"),
            Total_Relations=("Relations", "sum"),
            Total_Time_s=("Time_s", "sum"),
            Avg_Block_Time_s=("Time_s", "mean"),
            Failures=("Status", lambda s: (s.str.contains("Timeout|Truncated")).sum()),
        )
    )
    summary["Items_per_Minute"] = (
        (summary["Total_Entities"] + summary["Total_Relations"])
        / (summary["Total_Time_s"] / 60.0)
    )
    summary["Failure_Rate_%"] = (summary["Failures"] / summary["Blocks"]) * 100

    print("\n=== BENCHMARK SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

    models = sorted(df["Model"].unique())
    prompts = sorted(df["Prompt"].unique())
    modes = sorted(df["Mode"].unique())
    n_modes = len(modes)

    model_palette = dict(zip(models, sns.color_palette("tab10", len(models))))

    df["Model_Mode"] = df.apply(
        lambda r: f"{r['Model']} [{r['Mode']}]" if n_modes > 1 else r["Model"],
        axis=1,
    )
    summary["Model_Mode"] = summary.apply(
        lambda r: f"{r['Model']} [{r['Mode']}]" if n_modes > 1 else r["Model"],
        axis=1,
    )
    y_order = sorted(df["Model_Mode"].unique(), reverse=True)

    # ------------------------------------------------------------------
    # CHART 1: Extraction Volume (Entities & Relations by Prompt & Mode)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, n_modes, figsize=(7.5 * n_modes, 11), squeeze=False, sharey="row"
    )

    metrics_c1 = [
        ("Total_Entities", "Total Extracted Entities"),
        ("Total_Relations", "Total Extracted Relations"),
    ]

    for row_idx, (col_metric, base_title) in enumerate(metrics_c1):
        max_val = summary[col_metric].max()
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx, col_idx]
            sub = summary[summary["Mode"] == mode]

            sns.barplot(
                data=sub,
                x="Prompt",
                y=col_metric,
                hue="Model",
                order=prompts,
                hue_order=models,
                palette=model_palette,
                edgecolor="black",
                linewidth=0.8,
                ax=ax,
            )
            ax.set_title(f"{base_title} ({mode})", fontweight="bold", pad=12)
            ax.set_xlabel("")
            ax.set_ylabel("Count" if col_idx == 0 else "")
            ax.set_ylim(0, max_val * 1.16 if max_val > 0 else 1)
            ax.tick_params(axis="x", rotation=12)

            for container in ax.containers:
                ax.bar_label(container, fmt="%d", padding=3, fontsize=8.5, fontweight="bold")

            if ax.get_legend():
                ax.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(models), 5),
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(
        os.path.join(output_dir, f"1_extraction_volume.{IMG_FORMAT}"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------------
    # CHART 2: Block Latency Distribution & Throughput (Items / Minute)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, n_modes, figsize=(7.5 * n_modes, 11), squeeze=False)
    max_throughput = summary["Items_per_Minute"].max()

    for col_idx, mode in enumerate(modes):
        # Row 0: Block Latency Boxplot
        ax_box = axes[0, col_idx]
        sub_df = df[df["Mode"] == mode]

        sns.boxplot(
            data=sub_df,
            x="Prompt",
            y="Time_s",
            hue="Model",
            order=prompts,
            hue_order=models,
            palette=model_palette,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.45},
            ax=ax_box,
        )
        ax_box.axhline(
            TIMEOUT_LINE,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"Timeout (~{TIMEOUT_LINE:g}s)",
        )
        ax_box.set_title(f"Block Latency Distribution ({mode})", fontweight="bold", pad=12)
        ax_box.set_xlabel("")
        ax_box.set_ylabel("Time per Block (s)" if col_idx == 0 else "")
        ax_box.tick_params(axis="x", rotation=12)
        if ax_box.get_legend():
            ax_box.get_legend().remove()

        # Row 1: Throughput Barplot
        ax_eff = axes[1, col_idx]
        sub_res = summary[summary["Mode"] == mode]

        sns.barplot(
            data=sub_res,
            x="Prompt",
            y="Items_per_Minute",
            hue="Model",
            order=prompts,
            hue_order=models,
            palette=model_palette,
            edgecolor="black",
            linewidth=0.8,
            ax=ax_eff,
        )
        ax_eff.set_title(
            f"Throughput: Extracted Items / Minute ({mode})", fontweight="bold", pad=12
        )
        ax_eff.set_xlabel("")
        ax_eff.set_ylabel("Items / Minute" if col_idx == 0 else "")
        ax_eff.set_ylim(0, max_throughput * 1.16 if max_throughput > 0 else 1)
        ax_eff.tick_params(axis="x", rotation=12)

        for container in ax_eff.containers:
            ax_eff.bar_label(container, fmt="%.1f", padding=3, fontsize=8.5, fontweight="bold")

        if ax_eff.get_legend():
            ax_eff.get_legend().remove()

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=f"Model (dashed red line = ~{TIMEOUT_LINE:g}s timeout threshold)",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(models), 5),
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(
        os.path.join(output_dir, f"2_latency_and_efficiency.{IMG_FORMAT}"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------------
    # CHART 3: Block Outcomes Breakdown (Horizontal Stacked Bars)
    # ------------------------------------------------------------------
    status_colors = {
        "Success (Ents & Rels)": "#2ecc71",
        "Fast Empty (<60s)": "#bdc3c7",
        "Truncated (Entities Only >=60s)": "#f39c12",
        "Timeout / Slow Failure (>=60s)": "#e74c3c",
    }
    status_order = list(status_colors.keys())

    fig_height = max(7.5, len(y_order) * 0.6)
    fig, axes = plt.subplots(
        1,
        len(prompts),
        figsize=(5.6 * len(prompts) + 2.8, fig_height),
        sharey=True,
        squeeze=False,
    )

    for idx, prompt in enumerate(prompts):
        ax = axes[0, idx]
        sub_p = df[df["Prompt"] == prompt]

        counts = (
            sub_p.groupby(["Model_Mode", "Status"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=y_order, columns=status_order, fill_value=0)
        )
        pct = counts.div(counts.sum(axis=1), axis=0) * 100

        left_acc = np.zeros(len(y_order))
        for status in status_order:
            vals = pct[status].values
            bars = ax.barh(
                y_order,
                vals,
                left=left_acc,
                color=status_colors[status],
                edgecolor="white",
                linewidth=0.8,
                height=0.72,
                label=status,
            )
            for bar, val in zip(bars, vals):
                if val >= 9.0:
                    text_color = "white" if "Timeout" in status else "#1a1a1a"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=text_color,
                    )
            left_acc += vals

        ax.set_title(prompt, fontweight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage of Blocks (%)")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Block Outcome",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    plt.savefig(
        os.path.join(output_dir, f"3_block_outcomes.{IMG_FORMAT}"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------------
    # CHART 4: Model x Prompt Heatmaps (Relations & Total Time)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(5.5, len(y_order) * 0.52))
    )

    piv_rels = summary.pivot(index="Model_Mode", columns="Prompt", values="Total_Relations")
    sns.heatmap(piv_rels, annot=True, fmt=".0f", cmap="Blues", cbar=True, ax=axes[0])
    axes[0].set_title("Heatmap: Total Extracted Relations", fontweight="bold", pad=12)
    axes[0].set_ylabel("Model [Mode]")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    piv_time = summary.pivot(index="Model_Mode", columns="Prompt", values="Total_Time_s")
    sns.heatmap(piv_time, annot=True, fmt=".1f", cmap="OrRd", cbar=True, ax=axes[1])
    axes[1].set_title("Heatmap: Total Execution Time (s)", fontweight="bold", pad=12)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"4_model_prompt_heatmaps.{IMG_FORMAT}"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nCharts and summary CSV saved successfully to: ./{output_dir}/")


def main():
    args = parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: File '{args.log_file}' not found.", file=sys.stderr)
        sys.exit(1)

    df_blocks = parse_log_file(args.log_file)
    if df_blocks.empty:
        print("Error: No valid blocks found in the log file.", file=sys.stderr)
        sys.exit(1)

    generate_report_and_charts(df_blocks, args.output_dir)


if __name__ == "__main__":
    main()