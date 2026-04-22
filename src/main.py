"""
Main entry point for the Fairness Testing Tool.
Two-Phase Directed Search for AI Model Individual Fairness Testing.

Usage:
    python main.py --datasets kdd adult compas --runs 30
    python main.py --all --runs 30
    python main.py --visualize
"""

import argparse
import sys
import os

from config import DATASETS, EXPERIMENT_CONFIG
from experiment_runner import run_all_experiments, compute_statistics
from visualize_results import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="Two-Phase Directed Search for AI Model Fairness Testing"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Specific datasets to run (e.g., kdd adult compas)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all available datasets"
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help=f"Number of independent runs (default: {EXPERIMENT_CONFIG['num_runs']})"
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=f"Budget per run (default: {EXPERIMENT_CONFIG['budget']})"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate figures from existing results (skip experiments)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--figures", type=str, default="figures",
        help="Output directory for figures (default: figures)"
    )

    args = parser.parse_args()

    if args.visualize:
        print("Generating figures from existing results...")
        generate_all_figures(results_dir=args.output, figures_dir=args.figures)
        return

    # Determine which datasets to run
    if args.all:
        datasets = list(DATASETS.keys())
    elif args.datasets:
        datasets = args.datasets
    else:
        # Default: show available datasets and prompt
        print("Available datasets:")
        for name in DATASETS:
            cfg = DATASETS[name]
            exists = os.path.exists(cfg["data_path"]) and os.path.exists(cfg["model_path"])
            status = "FOUND" if exists else "NOT FOUND"
            print(f"  {name:15s} sensitive={cfg['sensitive_columns']}  [{status}]")
        print("\nUsage: python main.py --datasets kdd adult --runs 30")
        print("       python main.py --all --runs 30")
        return

    # Override budget if provided
    if args.budget is not None:
        EXPERIMENT_CONFIG["budget"] = args.budget

    print(f"Configuration:")
    print(f"  Datasets:      {datasets}")
    print(f"  Runs:          {args.runs or EXPERIMENT_CONFIG['num_runs']}")
    print(f"  Budget:        {EXPERIMENT_CONFIG['budget']}")
    print(f"  Threshold:     {EXPERIMENT_CONFIG['threshold']}")
    print(f"  Global ratio:  {EXPERIMENT_CONFIG['global_search_fraction']}")
    print(f"  Local radius:  {EXPERIMENT_CONFIG['local_search_radius']}")
    print(f"  Local steps:   {EXPERIMENT_CONFIG['local_search_steps']}")
    print()

    # Run experiments
    df = run_all_experiments(
        datasets=datasets, num_runs=args.runs, output_dir=args.output
    )

    if len(df) > 0:
        summary = compute_statistics(df, output_dir=args.output)
        print("\nGenerating figures...")
        generate_all_figures(results_dir=args.output, figures_dir=args.figures)
        print("\nDone! Check results/ and figures/ directories.")
    else:
        print("\nNo experiments were run. Check that dataset/model files exist.")


if __name__ == "__main__":
    main()
