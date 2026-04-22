"""
Baseline: Random Search for fairness testing.
Randomly generates input pairs and checks for individual discrimination.
This directly implements the lab 4 baseline approach.
"""

import numpy as np


class RandomSearch:
    """
    Random Search baseline for individual fairness testing.
    
    For each iteration, randomly selects a test sample, creates a paired
    sample by flipping the sensitive feature to a different value, applies
    small random perturbations to non-sensitive features on both samples,
    and checks whether the model produces different predictions (discrimination).
    """

    def __init__(self, model, X_test, sensitive_columns, threshold=0.05):
        """
        Args:
            model: Pre-trained Keras model.
            X_test: Test features (pandas DataFrame).
            sensitive_columns: List of sensitive feature column names.
            threshold: Prediction difference threshold for discrimination.
        """
        self.model = model
        self.X_test = X_test
        self.sensitive_columns = [c for c in sensitive_columns if c in X_test.columns]
        self.non_sensitive_columns = [
            c for c in X_test.columns if c not in self.sensitive_columns
        ]
        self.threshold = threshold

        # Precompute feature metadata
        self.unique_vals = {
            col: X_test[col].unique() for col in self.sensitive_columns
        }
        self.feature_bounds = {}
        self._col_idx = {}
        for i, col in enumerate(X_test.columns):
            self._col_idx[col] = i
            if col in self.non_sensitive_columns:
                self.feature_bounds[col] = (X_test[col].min(), X_test[col].max())

    def _generate_pair(self, rng):
        """Generate a single pair of test inputs (sample_a, sample_b)."""
        idx = rng.choice(len(self.X_test))
        # Convert to float numpy array to avoid dtype issues with perturbations
        sample_a = self.X_test.iloc[idx].values.astype(float).copy()
        sample_b = sample_a.copy()

        # Flip sensitive features to different values
        for col in self.sensitive_columns:
            ci = self._col_idx[col]
            unique = self.unique_vals[col]
            sample_b[ci] = rng.choice(unique)

        # Apply small perturbation to non-sensitive features on BOTH samples
        for col in self.non_sensitive_columns:
            ci = self._col_idx[col]
            min_val, max_val = self.feature_bounds[col]
            feature_range = max_val - min_val
            if feature_range == 0:
                continue
            perturbation = rng.uniform(-0.1 * feature_range, 0.1 * feature_range)
            sample_a[ci] = np.clip(sample_a[ci] + perturbation, min_val, max_val)
            sample_b[ci] = np.clip(sample_b[ci] + perturbation, min_val, max_val)

        return sample_a, sample_b

    def _is_discriminatory(self, sample_a, sample_b):
        """Check if a pair of inputs produces discriminatory predictions (batched for speed)."""
        import tensorflow as tf
        batch = np.vstack([sample_a.reshape(1, -1), sample_b.reshape(1, -1)])
        preds = self.model(tf.constant(batch, dtype=tf.float32), training=False).numpy()
        return abs(float(preds[0][0]) - float(preds[1][0])) > self.threshold

    def run(self, budget, seed=42):
        """
        Execute Random Search within a given budget.
        
        Args:
            budget: Maximum number of unique input pairs to generate.
            seed: Random seed for this run.
        
        Returns:
            dict with 'idi_ratio', 'disc_count', 'total_generated',
            and 'disc_pairs' (list of discriminatory input pairs).
        """
        rng = np.random.default_rng(seed)
        disc_count = 0
        disc_pairs = []
        seen = set()

        for _ in range(budget):
            sample_a, sample_b = self._generate_pair(rng)

            # Track uniqueness using a hash of the concatenated pair
            pair_key = (tuple(np.round(sample_a, 6)), tuple(np.round(sample_b, 6)))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            if self._is_discriminatory(sample_a, sample_b):
                disc_count += 1
                disc_pairs.append((sample_a.copy(), sample_b.copy()))

        total = len(seen)
        idi_ratio = disc_count / total if total > 0 else 0.0
        return {
            "idi_ratio": idi_ratio,
            "disc_count": disc_count,
            "total_generated": total,
            "disc_pairs": disc_pairs,
        }
