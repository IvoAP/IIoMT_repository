from __future__ import annotations
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein_distance
except Exception:  # pragma: no cover
    _scipy_wasserstein_distance = None


@dataclass(frozen=True)
class DistributionShiftSummary:
    kl_mean: float
    kl_median: float
    wasserstein_mean: float
    wasserstein_median: float


def _safe_normalize(hist: np.ndarray, epsilon: float) -> np.ndarray:
    hist = np.asarray(hist, dtype=float)
    hist = np.maximum(hist, 0.0)
    hist = hist + epsilon
    total = hist.sum()
    if total <= 0:
        return np.full_like(hist, 1.0 / len(hist))
    return hist / total


def kl_divergence(p: np.ndarray, q: np.ndarray, *, epsilon: float = 1e-12) -> float:
    """Compute discrete KL divergence KL(P || Q) with epsilon smoothing."""
    p_n = _safe_normalize(p, epsilon)
    q_n = _safe_normalize(q, epsilon)
    return float(np.sum(p_n * np.log(p_n / q_n)))


def wasserstein_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """Compute 1D Wasserstein distance between two sample sets."""
    u = np.asarray(u_values, dtype=float)
    v = np.asarray(v_values, dtype=float)
    if _scipy_wasserstein_distance is not None:
        return float(_scipy_wasserstein_distance(u, v))

    # NumPy fallback: approximate via quantile matching on sorted samples.
    # For equal weights in 1D, W1 can be approximated by mean absolute diff
    # between sorted samples after aligning lengths via interpolation.
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    n = max(len(u_sorted), len(v_sorted))
    if n == 0:
        return 0.0

    grid = np.linspace(0.0, 1.0, n, endpoint=True)
    u_q = np.interp(grid, np.linspace(0.0, 1.0, len(u_sorted), endpoint=True), u_sorted)
    v_q = np.interp(grid, np.linspace(0.0, 1.0, len(v_sorted), endpoint=True), v_sorted)
    return float(np.mean(np.abs(u_q - v_q)))


def _compute_common_edges(
    x_ref: np.ndarray,
    x_test: np.ndarray,
    *,
    bins: int,
) -> np.ndarray:
    min_val = float(np.nanmin([np.nanmin(x_ref), np.nanmin(x_test)]))
    max_val = float(np.nanmax([np.nanmax(x_ref), np.nanmax(x_test)]))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
        # Degenerate feature: a single-value (or invalid) distribution.
        return np.array([0.0, 1.0], dtype=float)
    return np.linspace(min_val, max_val, bins + 1, dtype=float)


def per_feature_shift_metrics(
    X_reference: np.ndarray,
    X_candidate: np.ndarray,
    *,
    bins: int = 30,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature KL divergence and Wasserstein distance.

    Returns
    -------
    kl_values : np.ndarray of shape (n_features,)
    wass_values : np.ndarray of shape (n_features,)
    """
    X_ref = np.asarray(X_reference)
    X_test = np.asarray(X_candidate)
    if X_ref.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_reference and X_candidate must be 2D arrays.")
    if X_ref.shape[1] != X_test.shape[1]:
        raise ValueError("X_reference and X_candidate must have the same number of features.")

    n_features = X_ref.shape[1]
    kl_values = np.zeros(n_features, dtype=float)
    wass_values = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        ref_col = X_ref[:, j]
        test_col = X_test[:, j]

        edges = _compute_common_edges(ref_col, test_col, bins=bins)
        hist_ref, _ = np.histogram(ref_col, bins=edges, density=False)
        hist_test, _ = np.histogram(test_col, bins=edges, density=False)

        kl_values[j] = kl_divergence(hist_ref, hist_test, epsilon=epsilon)
        wass_values[j] = wasserstein_1d(ref_col, test_col)

    return kl_values, wass_values


def summarize_shift_metrics(
    kl_values: np.ndarray,
    wass_values: np.ndarray,
) -> DistributionShiftSummary:
    kl = np.asarray(kl_values, dtype=float)
    wass = np.asarray(wass_values, dtype=float)

    kl = kl[np.isfinite(kl)]
    wass = wass[np.isfinite(wass)]

    if kl.size == 0:
        kl_mean = kl_median = float("nan")
    else:
        kl_mean = float(np.mean(kl))
        kl_median = float(np.median(kl))

    if wass.size == 0:
        w_mean = w_median = float("nan")
    else:
        w_mean = float(np.mean(wass))
        w_median = float(np.median(wass))

    return DistributionShiftSummary(
        kl_mean=kl_mean,
        kl_median=kl_median,
        wasserstein_mean=w_mean,
        wasserstein_median=w_median,
    )
