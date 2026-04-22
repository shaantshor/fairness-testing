"""
Configuration for fairness testing experiments.
Defines datasets, sensitive features, and experiment parameters.

File paths match the project layout where datasets (.csv) and
pre-trained models (.h5) are both in the model/ directory.
"""

import os

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "model")

DATASETS = {
    "adult": {
        "data_path": os.path.join(BASE_PATH, "processed_adult.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_adult.h5"),
        "sensitive_columns": ["gender", "race", "age"],
        "target_column": "Class-label",
    },
    "compas": {
        "data_path": os.path.join(BASE_PATH, "processed_compas.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_compas.h5"),
        "sensitive_columns": ["Sex", "Race"],
        "target_column": "Recidivism",
    },
    "kdd": {
        "data_path": os.path.join(BASE_PATH, "processed_kdd.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_kdd.h5"),
        "sensitive_columns": ["sex", "race"],
        "target_column": "income",
    },
    "law_school": {
        "data_path": os.path.join(BASE_PATH, "processed_law_school.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_law_school.h5"),
        "sensitive_columns": ["male", "race"],
        "target_column": "pass_bar",
    },
    "dutch": {
        "data_path": os.path.join(BASE_PATH, "processed_dutch.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_dutch.h5"),
        "sensitive_columns": ["sex", "age"],
        "target_column": "occupation",
    },
    "credit": {
        "data_path": os.path.join(BASE_PATH, "processed_credit_with_numerical.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_credit.h5"),
        "sensitive_columns": ["SEX", "EDUCATION", "MARRIAGE"],
        "target_column": "class",
    },
    "german": {
        "data_path": os.path.join(BASE_PATH, "processed_german.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_german.h5"),
        "sensitive_columns": ["PersonStatusSex", "AgeInYears"],
        "target_column": "CREDITRATING",
    },
    "crime": {
        "data_path": os.path.join(BASE_PATH, "processed_communities_crime.csv"),
        "model_path": os.path.join(BASE_PATH, "model_processed_communities_crime.h5"),
        "sensitive_columns": ["Black", "FemalePctDiv"],
        "target_column": "class",
    },
}

# Experiment parameters
EXPERIMENT_CONFIG = {
    "budget": 1000,
    "num_runs": 30,
    "threshold": 0.05,
    "test_size": 0.3,
    "random_state": 42,
    "local_search_radius": 0.1,
    "local_search_steps": 10,
    "global_search_fraction": 0.3,
}
