"""Reconstruction attack utilities.

A simple reconstruction attacker links anonymized records to original records by
solving a global one-to-one assignment that minimizes pairwise Euclidean
distance after feature standardization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


@dataclass
class ReconstructionAttackResult:
    rmse: float
    mae: float
    median_distance: float
    p90_distance: float
    row_reconstruction_rate: float
    evaluated_rows: int
    total_rows: int
    sampled: bool


def evaluate_reconstruction_attack(
    X_reference: np.ndarray,
    X_anonymized: np.ndarray,
    *,
    row_tolerance: float = 0.10,
    max_samples: int | None = 5000,
    random_state: int = 42,
) -> ReconstructionAttackResult:
    """Evaluate a linkage-based reconstruction attack.

    Parameters
    ----------
    X_reference : np.ndarray
        Original dataset matrix with shape (n_samples, n_features).
    X_anonymized : np.ndarray
        Anonymized dataset matrix with shape (n_samples, n_features).
    row_tolerance : float, optional
        Threshold in standardized Euclidean distance to count a row as
        successfully reconstructed.
    max_samples : int | None, optional
        Maximum number of rows evaluated by the attack to keep runtime and
        memory bounded. If None, use all rows.
    random_state : int, optional
        Seed used when sampling rows.
    """
    X_ref = np.asarray(X_reference, dtype=float)
    X_anon = np.asarray(X_anonymized, dtype=float)

    if X_ref.ndim != 2 or X_anon.ndim != 2:
        raise ValueError("X_reference and X_anonymized must be 2D arrays.")
    if X_ref.shape != X_anon.shape:
        raise ValueError("X_reference and X_anonymized must have the same shape.")

    total_rows = X_ref.shape[0]
    if max_samples is not None:
        if max_samples <= 1:
            raise ValueError("max_samples must be > 1 when provided.")
        if total_rows > max_samples:
            rng = np.random.default_rng(seed=random_state)
            sampled_indices = rng.choice(total_rows, size=max_samples, replace=False)
            X_ref = X_ref[sampled_indices]
            X_anon = X_anon[sampled_indices]

    evaluated_rows = X_ref.shape[0]
    sampled = evaluated_rows < total_rows

    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)
    X_anon_scaled = scaler.transform(X_anon)

    # Compute global optimal one-to-one linkage between anonymized and original rows.
    distance_matrix = cdist(X_anon_scaled, X_ref_scaled, metric="euclidean")
    anon_idx, ref_idx = linear_sum_assignment(distance_matrix)

    X_ref_matched = X_ref_scaled[ref_idx]
    X_anon_ordered = X_anon_scaled[anon_idx]
    deltas = X_anon_ordered - X_ref_matched

    per_row_distance = np.linalg.norm(deltas, axis=1)
    rmse = float(np.sqrt(np.mean(deltas**2)))
    mae = float(np.mean(np.abs(deltas)))
    median_distance = float(np.median(per_row_distance))
    p90_distance = float(np.quantile(per_row_distance, 0.90))
    row_reconstruction_rate = float(np.mean(per_row_distance <= row_tolerance))

    return ReconstructionAttackResult(
        rmse=rmse,
        mae=mae,
        median_distance=median_distance,
        p90_distance=p90_distance,
        row_reconstruction_rate=row_reconstruction_rate,
        evaluated_rows=evaluated_rows,
        total_rows=total_rows,
        sampled=sampled,
    )
