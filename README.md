# Two-Phase Directed Search for AI Model Fairness Testing

An intelligent software engineering tool that improves upon Random Search for detecting individual discriminatory instances in pre-trained AI models.

## Overview

This tool implements a **Two-Phase Directed Search** approach for fairness testing:

- **Phase 1 (Global Search):** Randomly explores the input space to find initial discriminatory instances (seeds).
- **Phase 2 (Local Search):** Performs neighbourhood perturbation around discovered seeds to exploit the clustering property of discrimination regions, finding more discriminatory instances per unit of budget.

The baseline is **Random Search**, as provided in Lab 4 of the Intelligent Software Engineering module.

## Project Structure

```
fairness-testing/
├── src/
│   ├── config.py                  # Dataset and experiment configuration
│   ├── data_utils.py              # Data loading and preprocessing
│   ├── baseline_random_search.py  # Baseline: Random Search
│   ├── directed_search.py         # Proposed: Two-Phase Directed Search
│   ├── experiment_runner.py       # Experiment orchestration + statistics
│   ├── visualize_results.py       # Figure generation for report
│   └── main.py                    # Entry point
├── model/                         # Datasets (.csv) and pre-trained models (.h5)
├── results/                       # Raw and summary CSV results
├── figures/                       # Generated figures (PNG)
├── docs/
│   ├── requirements.pdf
│   ├── manual.pdf
│   └── replication.pdf
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets/models from:
# https://github.com/ideas-labo/ISE/tree/main/lab4
# Place them in the model/ directory

# Run experiments on specific datasets
cd src
python main.py --datasets kdd adult compas --runs 30

# Run on all datasets
python main.py --all --runs 30

# Generate figures only (from existing results)
python main.py --visualize
```

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

## Metric

**IDI Ratio** (Individual Discrimination Instance Ratio) = I / S

Where I = number of unique discriminatory instances found, S = total unique inputs generated (budget).

## Statistical Testing

Each experiment is repeated 30 times with different random seeds. Results are compared using the **Wilcoxon signed-rank test** (paired, one-sided) with significance level α = 0.05. Effect sizes are reported using **Cliff's delta** and **Vargha-Delaney A12**.

## Author

Sakshi Palve — MSc Advanced Computer Science, University of Birmingham (2025–2026)

## Module

Intelligent Software Engineering — Dr. Tao Chen and Dr. Rami Bahsoon
# fairness-testing
