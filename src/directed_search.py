"""
Proposed Solution: Two-Phase Directed Search for fairness testing.

Inspired by AEQUITAS (Udeshi et al., 2018), this approach combines:
  Phase 1 (Global Search): Random exploration to discover initial discriminatory
      instances — identical to the baseline but uses only a fraction of the budget.
  Phase 2 (Local Search): Neighbourhood perturbation around discovered discriminatory
      instances to exploit the clustering property of discrimination regions.

Rationale:
  Neural network decision boundaries are locally smooth, so if an input pair triggers
  discrimination, nearby input pairs (with small perturbations to non-sensitive features)
  are also likely to trigger discrimination. By concentrating search effort around known
  discriminatory regions, we find more unique discriminatory instances per unit of budget
  than purely random sampling.

Design choices:
  - Budget split: a configurable fraction goes to global search; the rest to local search.
  - Local perturbation radius: smaller than global (fraction of feature range) to stay
    in the discriminatory neighbourhood.
  - Adaptive seeding: each newly found discriminatory instance in local search also
    becomes a seed for further local exploration (breadth-first expansion).
"""

import numpy as np
from collections import deque


class TwoPhaseDirectedSearch:
    """
    Two-Phase Directed Search for individual fairness testing.
    
    Phase 1 — Global: random sampling (same as baseline) to find initial seeds.
    Phase 2 — Local: neighbourhood perturbation around seeds to find more
              discriminatory instances efficiently.
    """

    def __init__(
        self,
        model,
        X_test,
        sensitive_columns,
        threshold=0.05,
        local_radius=0.1,
        local_steps=10,
        global_fraction=0.3,
    ):
        """
        Args:
            model: Pre-trained Keras model.
            X_test: Test features (pandas DataFrame).
            sensitive_columns: List of sensitive feature column names.
            threshold: Prediction difference threshold for discrimination.
            local_radius: Perturbation radius for local search (fraction of feature range).
            local_steps: Number of local perturbation attempts per seed.
            global_fraction: Fraction of total budget allocated to global phase.
        """
        self.model = model
        self.X_test = X_test
        self.sensitive_columns = [c for c in sensitive_columns if c in X_test.columns]
        self.non_sensitive_columns = [
            c for c in X_test.columns if c not in self.sensitive_columns
        ]
        self.threshold = threshold
        self.local_radius = local_radius
        self.local_steps = local_steps
        self.global_fraction = global_fraction

        # Precompute feature metadata
        self.unique_vals = {
            col: X_test[col].unique() for col in self.sensitive_columns
        }
        self.feature_bounds = {}
        self.col_indices = {}
        for i, col in enumerate(X_test.columns):
            if col in self.non_sensitive_columns:
                self.feature_bounds[col] = (X_test[col].min(), X_test[col].max())
            self.col_indices[col] = i

    def _is_discriminatory(self, sample_a, sample_b):
        """Check if a pair produces discriminatory predictions (batched for speed)."""
        import tensorflow as tf
        batch = np.vstack([sample_a.reshape(1, -1), sample_b.reshape(1, -1)])
        preds = self.model(tf.constant(batch, dtype=tf.float32), training=False).numpy()
        return abs(float(preds[0][0]) - float(preds[1][0])) > self.threshold

    def _generate_global_pair(self, rng):
        """Generate a random input pair (same as baseline)."""
        idx = rng.choice(len(self.X_test))
        sample_a = self.X_test.iloc[idx].values.copy().astype(float)
        sample_b = sample_a.copy()

        # Flip sensitive features
        for col in self.sensitive_columns:
            ci = self.col_indices[col]
            unique = self.unique_vals[col]
            sample_b[ci] = rng.choice(unique)

        # Random perturbation on non-sensitive features
        for col in self.non_sensitive_columns:
            ci = self.col_indices[col]
            min_val, max_val = self.feature_bounds[col]
            feature_range = max_val - min_val
            if feature_range == 0:
                continue
            perturbation = rng.uniform(-0.1 * feature_range, 0.1 * feature_range)
            sample_a[ci] = np.clip(sample_a[ci] + perturbation, min_val, max_val)
            sample_b[ci] = np.clip(sample_b[ci] + perturbation, min_val, max_val)

        return sample_a, sample_b

    def _generate_local_pair(self, seed_a, seed_b, rng):
        """
        Generate a neighbour pair by applying SMALL perturbations to
        non-sensitive features of a known discriminatory seed pair.
        The sensitive features remain unchanged (preserving the discrimination axis).
        """
        new_a = seed_a.copy()
        new_b = seed_b.copy()

        for col in self.non_sensitive_columns:
            ci = self.col_indices[col]
            min_val, max_val = self.feature_bounds[col]
            feature_range = max_val - min_val
            if feature_range == 0:
                continue
            # Smaller perturbation radius than global search
            perturbation = rng.uniform(
                -self.local_radius * feature_range,
                self.local_radius * feature_range,
            )
            new_a[ci] = np.clip(new_a[ci] + perturbation, min_val, max_val)
            new_b[ci] = np.clip(new_b[ci] + perturbation, min_val, max_val)

        return new_a, new_b

    def run(self, budget, seed=42):
        """
        Execute Two-Phase Directed Search within a given budget.
        
        Phase 1: Use global_fraction of the budget to randomly explore.
        Phase 2: Use the remaining budget to do local search around
                 discovered discriminatory seeds.
        
        Args:
            budget: Maximum number of unique input pairs to generate (S).
            seed: Random seed for this run.
        
        Returns:
            dict with 'idi_ratio', 'disc_count', 'total_generated',
            and 'disc_pairs'.
        """
        rng = np.random.default_rng(seed)
        disc_count = 0
        disc_pairs = []
        seen = set()

        global_budget = int(budget * self.global_fraction)
        remaining_budget = budget - global_budget

        # -- Phase 1: Global Search --
        seed_queue = deque()  # seeds for local search

        for _ in range(global_budget):
            sample_a, sample_b = self._generate_global_pair(rng)
            pair_key = (tuple(np.round(sample_a, 6)), tuple(np.round(sample_b, 6)))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            if self._is_discriminatory(sample_a, sample_b):
                disc_count += 1
                disc_pairs.append((sample_a.copy(), sample_b.copy()))
                seed_queue.append((sample_a.copy(), sample_b.copy()))

        # -- Phase 2: Local Search --
        # Use BFS-style expansion: process seeds, and any newly found
        # discriminatory instances become new seeds too.
        local_used = 0

        while seed_queue and local_used < remaining_budget:
            current_seed_a, current_seed_b = seed_queue.popleft()

            for _ in range(self.local_steps):
                if local_used >= remaining_budget:
                    break

                new_a, new_b = self._generate_local_pair(
                    current_seed_a, current_seed_b, rng
                )
                pair_key = (tuple(np.round(new_a, 6)), tuple(np.round(new_b, 6)))
                if pair_key in seen:
                    local_used += 1
                    continue
                seen.add(pair_key)
                local_used += 1

                if self._is_discriminatory(new_a, new_b):
                    disc_count += 1
                    disc_pairs.append((new_a.copy(), new_b.copy()))
                    # New discriminatory instance becomes a seed too
                    seed_queue.append((new_a.copy(), new_b.copy()))

        # If local search exhausted all seeds but budget remains,
        # fall back to global search for the rest
        fallback_budget = remaining_budget - local_used
        for _ in range(fallback_budget):
            sample_a, sample_b = self._generate_global_pair(rng)
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
