"""
Data loading and preprocessing utilities.
Handles loading CSV datasets and pre-trained Keras models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras


def load_and_preprocess_data(data_path, target_column, test_size=0.3, random_state=42):
    """
    Load a CSV dataset and split into train/test sets.
    
    Args:
        data_path: Path to the CSV file.
        target_column: Name of the target/label column.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test as pandas DataFrames/Series.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_trained_model(model_path):
    """
    Load a pre-trained Keras model from an .h5 file.
    
    Args:
        model_path: Path to the .h5 model file.
    
    Returns:
        Compiled Keras model.
    """
    return keras.models.load_model(model_path)


def get_feature_bounds(X_test):
    """
    Compute per-feature min/max bounds from test data.
    
    Args:
        X_test: Test feature DataFrame.
    
    Returns:
        Dictionary mapping column name -> (min_val, max_val).
    """
    bounds = {}
    for col in X_test.columns:
        bounds[col] = (X_test[col].min(), X_test[col].max())
    return bounds


def get_unique_values(X_test, columns):
    """
    Get unique values for specified columns (used for sensitive features).
    
    Args:
        X_test: Test feature DataFrame.
        columns: List of column names.
    
    Returns:
        Dictionary mapping column name -> array of unique values.
    """
    return {col: X_test[col].unique() for col in columns if col in X_test.columns}
