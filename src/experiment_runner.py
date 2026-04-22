"""
Experiment runner for fairness testing comparison.
Runs both baseline and proposed approach across multiple datasets,
multiple sensitive features, and multiple independent runs.
Performs Wilcoxon signed-rank test for statistical significance.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

from data_utils import load_and_preprocess_data, load_trained_model
from baseline_random_search import RandomSearch
from directed_search import TwoPhaseDirectedSearch
from config import DATASETS, EXPERIMENT_CONFIG


def run_single_experiment(
    dataset_name, dataset_cfg, sensitive_col, exp_cfg, run_id
):
    """
    Run one experiment: baseline + proposed, for one dataset, one sensitive
    feature, one run.
    
    Args:
        dataset_name: Name of the dataset.
        dataset_cfg: Dict with data_path, model_path, etc.
        sensitive_col: The sensitive column to test on.
        exp_cfg: Experiment configuration dict.
        run_id: Index of this run (added to seed for variation).
    
    Returns:
        Dict with results for both approaches.
    """
    seed = exp_cfg["random_state"] + run_id

    # Load data and model
    _, X_test, _, _ = load_and_preprocess_data(
        dataset_cfg["data_path"],
        dataset_cfg["target_column"],
        test_size=exp_cfg["test_size"],
        random_state=exp_cfg["random_state"],  # same split across runs
    )
    model = load_trained_model(dataset_cfg["model_path"])

    budget = exp_cfg["budget"]
    threshold = exp_cfg["threshold"]
    sensitive_columns = [sensitive_col]

    # --- Baseline: Random Search ---
    t0 = time.time()
    baseline = RandomSearch(model, X_test, sensitive_columns, threshold)
    baseline_result = baseline.run(budget, seed=seed)
    baseline_time = time.time() - t0

    # --- Proposed: Two-Phase Directed Search ---
    t0 = time.time()
    proposed = TwoPhaseDirectedSearch(
        model,
        X_test,
        sensitive_columns,
        threshold=threshold,
        local_radius=exp_cfg["local_search_radius"],
        local_steps=exp_cfg["local_search_steps"],
        global_fraction=exp_cfg["global_search_fraction"],
    )
    proposed_result = proposed.run(budget, seed=seed)
    proposed_time = time.time() - t0

    return {
        "dataset": dataset_name,
        "sensitive_feature": sensitive_col,
        "run_id": run_id,
        "baseline_idi_ratio": baseline_result["idi_ratio"],
        "baseline_disc_count": baseline_result["disc_count"],
        "baseline_total": baseline_result["total_generated"],
        "baseline_time_s": round(baseline_time, 3),
        "proposed_idi_ratio": proposed_result["idi_ratio"],
        "proposed_disc_count": proposed_result["disc_count"],
        "proposed_total": proposed_result["total_generated"],
        "proposed_time_s": round(proposed_time, 3),
    }


def run_all_experiments(datasets=None, num_runs=None, output_dir="results"):
    """
    Run experiments across all configured datasets and sensitive features.
    
    Args:
        datasets: List of dataset names to run (None = all).
        num_runs: Override number of runs (None = use config).
        output_dir: Directory to save results.
    
    Returns:
        pandas DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_cfg = EXPERIMENT_CONFIG.copy()
    if num_runs is not None:
        exp_cfg["num_runs"] = num_runs

    if datasets is None:
        datasets = list(DATASETS.keys())

    all_results = []
    total_experiments = 0

    for ds_name in datasets:
        if ds_name not in DATASETS:
            print(f"WARNING: Dataset '{ds_name}' not found in config, skipping.")
            continue

        ds_cfg = DATASETS[ds_name]

        # Check if files exist
        if not os.path.exists(ds_cfg["data_path"]):
            print(f"WARNING: Data file not found for '{ds_name}': {ds_cfg['data_path']}, skipping.")
            continue
        if not os.path.exists(ds_cfg["model_path"]):
            print(f"WARNING: Model file not found for '{ds_name}': {ds_cfg['model_path']}, skipping.")
            continue

        for sens_col in ds_cfg["sensitive_columns"]:
            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name} | Sensitive Feature: {sens_col}")
            print(f"{'='*60}")

            for run_id in range(exp_cfg["num_runs"]):
                print(f"  Run {run_id + 1}/{exp_cfg['num_runs']}...", end=" ", flush=True)
                result = run_single_experiment(
                    ds_name, ds_cfg, sens_col, exp_cfg, run_id
                )
                all_results.append(result)
                total_experiments += 1
                print(
                    f"Baseline IDI={result['baseline_idi_ratio']:.4f} | "
                    f"Proposed IDI={result['proposed_idi_ratio']:.4f}"
                )

    # Save raw results
    df = pd.DataFrame(all_results)
    raw_path = os.path.join(output_dir, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")
    print(f"Total experiments completed: {total_experiments}")

    return df


def compute_statistics(df, output_dir="results"):
    """
    Compute summary statistics and Wilcoxon signed-rank tests.
    
    Args:
        df: DataFrame from run_all_experiments.
        output_dir: Directory to save summary.
    
    Returns:
        Summary DataFrame.
    """
    summaries = []

    groups = df.groupby(["dataset", "sensitive_feature"])
    for (ds_name, sens_col), group in groups:
        baseline_idi = group["baseline_idi_ratio"].values
        proposed_idi = group["proposed_idi_ratio"].values

        # Wilcoxon signed-rank test (paired, two-sided)
        # Tests whether proposed IDI ratios are significantly different from baseline
        if len(baseline_idi) >= 5 and not np.all(baseline_idi == proposed_idi):
            stat, p_value = stats.wilcoxon(proposed_idi, baseline_idi, alternative="greater")
        else:
            stat, p_value = np.nan, np.nan

        # Effect size: Cliff's delta (non-parametric effect size)
        cliffs_d = cliffs_delta(proposed_idi, baseline_idi)

        # Vargha-Delaney A measure (probability that proposed > baseline)
        a12 = vargha_delaney(proposed_idi, baseline_idi)

        summary = {
            "dataset": ds_name,
            "sensitive_feature": sens_col,
            "n_runs": len(group),
            "baseline_mean_idi": round(np.mean(baseline_idi), 4),
            "baseline_std_idi": round(np.std(baseline_idi), 4),
            "proposed_mean_idi": round(np.mean(proposed_idi), 4),
            "proposed_std_idi": round(np.std(proposed_idi), 4),
            "improvement_pct": round(
                (np.mean(proposed_idi) - np.mean(baseline_idi))
                / max(np.mean(baseline_idi), 1e-10) * 100, 2
            ),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else "N/A",
            "wilcoxon_p_value": round(p_value, 6) if not np.isnan(p_value) else "N/A",
            "significant_0.05": p_value < 0.05 if not np.isnan(p_value) else "N/A",
            "cliffs_delta": round(cliffs_d, 4),
            "vargha_delaney_a12": round(a12, 4),
            "baseline_mean_time_s": round(group["baseline_time_s"].mean(), 3),
            "proposed_mean_time_s": round(group["proposed_time_s"].mean(), 3),
        }
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to {summary_path}")

    # Print formatted summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"\n{row['dataset']} ({row['sensitive_feature']}):")
        print(f"  Baseline IDI ratio:  {row['baseline_mean_idi']:.4f} +/- {row['baseline_std_idi']:.4f}")
        print(f"  Proposed IDI ratio:  {row['proposed_mean_idi']:.4f} +/- {row['proposed_std_idi']:.4f}")
        print(f"  Improvement:         {row['improvement_pct']:.2f}%")
        print(f"  Wilcoxon p-value:    {row['wilcoxon_p_value']}")
        print(f"  Significant (0.05):  {row['significant_0.05']}")
        print(f"  Cliff's delta:       {row['cliffs_delta']:.4f}")
        print(f"  A12:                 {row['vargha_delaney_a12']:.4f}")

    return summary_df


def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size.
    Ranges from -1 to 1. Positive means x tends to be larger than y.
    """
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def vargha_delaney(x, y):
    """
    Compute Vargha-Delaney A12 measure.
    A12 = 0.5 means no difference; >0.5 means x tends to be larger.
    """
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    equal = sum(1 for xi in x for yi in y if xi == yi)
    return (more + 0.5 * equal) / (n_x * n_y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run fairness testing experiments")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to run (default: all available)"
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of independent runs (default: from config)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    df = run_all_experiments(
        datasets=args.datasets,
        num_runs=args.runs,
        output_dir=args.output,
    )
    if len(df) > 0:
        compute_statistics(df, output_dir=args.output)
