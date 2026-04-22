"""
Visualization script for fairness testing results.
Generates publication-quality figures for the report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 10


def plot_idi_comparison_bar(summary_df, output_dir="figures"):
    """
    Bar chart comparing mean IDI ratios across datasets/sensitive features.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = [
        f"{row['dataset']}\n({row['sensitive_feature']})"
        for _, row in summary_df.iterrows()
    ]
    baseline_means = summary_df["baseline_mean_idi"].values
    proposed_means = summary_df["proposed_mean_idi"].values
    baseline_stds = summary_df["baseline_std_idi"].values
    proposed_stds = summary_df["proposed_std_idi"].values

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars1 = ax.bar(
        x - width / 2, baseline_means, width,
        yerr=baseline_stds, capsize=3,
        label="Random Search (Baseline)", color="#4ECDC4", edgecolor="black", linewidth=0.5
    )
    bars2 = ax.bar(
        x + width / 2, proposed_means, width,
        yerr=proposed_stds, capsize=3,
        label="Two-Phase Directed Search (Proposed)", color="#FF6B6B", edgecolor="black", linewidth=0.5
    )

    ax.set_ylabel("IDI Ratio")
    ax.set_title("Individual Discrimination Instance Ratio: Baseline vs Proposed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Add significance markers
    for i, (_, row) in enumerate(summary_df.iterrows()):
        if row["significant_0.05"] == True:
            max_val = max(
                baseline_means[i] + baseline_stds[i],
                proposed_means[i] + proposed_stds[i]
            )
            ax.text(i, max_val + 0.02, "*", ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "idi_comparison_bar.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_boxplots(raw_df, output_dir="figures"):
    """
    Box plots showing distribution of IDI ratios across runs for each
    dataset/sensitive feature combination.
    """
    os.makedirs(output_dir, exist_ok=True)

    groups = raw_df.groupby(["dataset", "sensitive_feature"])
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(max(8, n_groups * 3), 5), sharey=False)
    if n_groups == 1:
        axes = [axes]

    for idx, ((ds, sf), group) in enumerate(groups):
        ax = axes[idx]
        data = [group["baseline_idi_ratio"].values, group["proposed_idi_ratio"].values]
        bp = ax.boxplot(
            data, labels=["Baseline", "Proposed"],
            patch_artist=True, widths=0.6
        )
        bp["boxes"][0].set_facecolor("#4ECDC4")
        bp["boxes"][1].set_facecolor("#FF6B6B")
        for box in bp["boxes"]:
            box.set_edgecolor("black")
            box.set_linewidth(0.8)

        ax.set_title(f"{ds}\n({sf})", fontsize=9)
        ax.set_ylabel("IDI Ratio" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Distribution of IDI Ratios Across 30 Runs", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "idi_boxplots.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_improvement_heatmap(summary_df, output_dir="figures"):
    """
    Heatmap showing percentage improvement of proposed over baseline
    across datasets and sensitive features.
    """
    os.makedirs(output_dir, exist_ok=True)

    pivot = summary_df.pivot_table(
        values="improvement_pct",
        index="dataset",
        columns="sensitive_feature",
        aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 2), max(4, pivot.shape[0] * 0.8)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add text annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9,
                        color="black" if abs(val) < 50 else "white")

    ax.set_title("Improvement in IDI Ratio (%) — Proposed vs Baseline")
    plt.colorbar(im, ax=ax, label="Improvement (%)")
    plt.tight_layout()
    path = os.path.join(output_dir, "improvement_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_time_comparison(summary_df, output_dir="figures"):
    """Bar chart comparing mean execution times."""
    os.makedirs(output_dir, exist_ok=True)

    labels = [
        f"{row['dataset']}\n({row['sensitive_feature']})"
        for _, row in summary_df.iterrows()
    ]
    baseline_times = summary_df["baseline_mean_time_s"].values
    proposed_times = summary_df["proposed_mean_time_s"].values

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    ax.bar(x - width / 2, baseline_times, width, label="Baseline", color="#4ECDC4", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, proposed_times, width, label="Proposed", color="#FF6B6B", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean Execution Time (seconds)")
    ax.set_title("Execution Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "time_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_all_figures(results_dir="results", figures_dir="figures"):
    """Generate all figures from saved results."""
    raw_path = os.path.join(results_dir, "raw_results.csv")
    summary_path = os.path.join(results_dir, "summary_statistics.csv")

    if not os.path.exists(raw_path):
        print(f"ERROR: Raw results not found at {raw_path}. Run experiments first.")
        return

    raw_df = pd.read_csv(raw_path)
    summary_df = pd.read_csv(summary_path)

    plot_idi_comparison_bar(summary_df, figures_dir)
    plot_boxplots(raw_df, figures_dir)
    plot_improvement_heatmap(summary_df, figures_dir)
    plot_time_comparison(summary_df, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    generate_all_figures()
