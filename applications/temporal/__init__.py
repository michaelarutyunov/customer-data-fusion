"""
H1 Temporal Dynamics — Regime shift detection from CDT embedding trajectories.

Components:
- generate_monthly_embeddings: Re-encode frozen fusion model per month
- extract_features: Compute L2 distance statistics across trajectories
- train_drift_detector: Two-stage drift detector (binary + month estimation)
"""

__all__ = []
