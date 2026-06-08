"""
evaluation/overlap.py

Bhattacharyya coefficient (BC) between two archetypes' per-feature distributions.

Uses a Gaussian approximation per feature (mean/std estimated from samples),
then averages BC across features within a modality.

BC near 1.0 = heavy overlap; BC near 0.0 = well-separated.

Designed-overlap target: BC in [0.4, 0.6] for named modality pairs.
"""

from __future__ import annotations

import math
import numpy as np


def bhattacharyya_coeff_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """
    Closed-form BC between two univariate Gaussians.

    BC = exp(-1/8 * (mu1-mu2)^2 / ((sigma1^2+sigma2^2)/2)) * sqrt(2*sigma1*sigma2 / (sigma1^2+sigma2^2))
    """
    sigma1 = max(sigma1, 1e-8)
    sigma2 = max(sigma2, 1e-8)
    var_mean = (sigma1**2 + sigma2**2) / 2.0
    term1 = math.exp(-0.125 * (mu1 - mu2) ** 2 / var_mean)
    term2 = math.sqrt(2.0 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
    return term1 * term2


def archetype_bc(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """
    Mean Bhattacharyya coefficient between two archetypes across features.

    Parameters
    ----------
    samples_a, samples_b : np.ndarray of shape (n_samples, n_features)
        Feature vectors for each archetype. Each column is one feature.

    Returns
    -------
    float
        Mean BC across all features (Gaussian approximation per feature).
    """
    if samples_a.ndim == 1:
        samples_a = samples_a.reshape(-1, 1)
    if samples_b.ndim == 1:
        samples_b = samples_b.reshape(-1, 1)

    n_features = samples_a.shape[1]
    bc_values = []
    for i in range(n_features):
        mu1, sigma1 = float(np.mean(samples_a[:, i])), float(np.std(samples_a[:, i]))
        mu2, sigma2 = float(np.mean(samples_b[:, i])), float(np.std(samples_b[:, i]))
        bc_values.append(bhattacharyya_coeff_gaussian(mu1, sigma1, mu2, sigma2))
    return float(np.mean(bc_values))
