"""
Shared probe evaluation utilities for encoder assessment.

Every encoder is evaluated the same way: freeze the trained encoder,
generate embeddings for all participants, then train a logistic regression
probe on the train split and evaluate on the val split.

This module also provides cosine similarity statistics for intra/inter
persona separation analysis (used by text encoder probe).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def probe_logistic_regression(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    max_iter: int = 1000,
    random_state: int = 42,
    standardize: bool = True,
) -> dict:
    """Train a logistic regression probe on frozen embeddings.

    Parameters
    ----------
    embeddings:
        (n_samples, embedding_dim) — frozen encoder outputs.
    labels:
        (n_samples,) — integer class labels.
    train_idx:
        Indices for training split.
    val_idx:
        Indices for validation split.
    max_iter:
        Maximum iterations for LogisticRegression solver.
    random_state:
        Random seed for reproducibility.
    standardize:
        Whether to StandardScaler the embeddings before fitting.

    Returns
    -------
    dict with keys:
        - train_accuracy: float
        - val_accuracy: float
        - confusion_matrix: np.ndarray (n_classes, n_classes) on val set
        - per_class_accuracy: dict[str, float]
        - classifier: fitted LogisticRegression
    """
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_val = embeddings[val_idx]
    y_val = labels[val_idx]

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    cm = confusion_matrix(y_val, val_preds)

    # Per-class accuracy
    per_class: dict[str, float] = {}
    unique_classes = sorted(set(y_val))
    for cls_idx in unique_classes:
        mask = y_val == cls_idx
        if mask.any():
            per_class[str(cls_idx)] = accuracy_score(y_val[mask], val_preds[mask])

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class,
        "classifier": clf,
    }


def compute_cosine_similarity_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[list[str]] = None,
) -> dict:
    """Compute intra and inter-persona cosine similarity statistics.

    Parameters
    ----------
    embeddings:
        (n_samples, embedding_dim) — L2-normalised or raw embeddings.
        Will be L2-normalised if not already.
    labels:
        (n_samples,) — class labels (integer or string).
    label_names:
        Optional mapping from class index to name.

    Returns
    -------
    dict with keys:
        - intra_mean: mean cosine sim within same class
        - intra_std: std of intra-class cosine sims
        - inter_mean: mean cosine sim across different classes
        - inter_std: std of inter-class cosine sims
        - intra_per_class: dict mapping class name → mean intra similarity
        - pairwise_matrix: (n_classes, n_classes) mean cosine sim between classes
        - class_names: list of class name strings
    """
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings_norm = embeddings / norms

    # Cosine similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T  # (N, N)

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)

    if label_names is None:
        class_names = [str(lbl) for lbl in unique_labels]
    else:
        class_names = label_names

    # Per-class masks
    class_masks = {lbl: (np.array(labels) == lbl) for lbl in unique_labels}

    # Intra-class similarities (excluding self-similarity)
    intra_values: list[float] = []
    intra_per_class: dict[str, float] = {}

    for lbl in unique_labels:
        mask = class_masks[lbl]
        indices = np.where(mask)[0]
        if len(indices) < 2:
            intra_per_class[str(lbl)] = float("nan")
            continue
        # Upper triangle of the sub-matrix for this class
        sub_sim = sim_matrix[np.ix_(indices, indices)]
        # Exclude diagonal (self-similarity = 1.0)
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        intra_vals = sub_sim[triu_idx]
        intra_values.extend(intra_vals.tolist())
        intra_per_class[str(lbl)] = float(np.mean(intra_vals))

    intra_mean = float(np.mean(intra_values)) if intra_values else float("nan")
    intra_std = float(np.std(intra_values)) if intra_values else float("nan")

    # Inter-class similarities
    inter_values: list[float] = []
    pairwise_matrix = np.zeros((n_classes, n_classes))

    for i, lbl_i in enumerate(unique_labels):
        for j, lbl_j in enumerate(unique_labels):
            if i >= j:
                continue
            mask_i = class_masks[lbl_i]
            mask_j = class_masks[lbl_j]
            sub_sim = sim_matrix[np.ix_(mask_i, mask_j)]
            mean_sim = float(np.mean(sub_sim))
            pairwise_matrix[i, j] = mean_sim
            pairwise_matrix[j, i] = mean_sim
            inter_values.extend(sub_sim.flatten().tolist())

    inter_mean = float(np.mean(inter_values)) if inter_values else float("nan")
    inter_std = float(np.std(inter_values)) if inter_values else float("nan")

    return {
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "intra_per_class": intra_per_class,
        "pairwise_matrix": pairwise_matrix,
        "class_names": class_names,
    }


def mean_pool_per_participant(
    embeddings: np.ndarray,
    participant_ids: list[str],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mean-pool multiple embeddings per participant into one.

    Parameters
    ----------
    embeddings:
        (n_samples, embedding_dim)
    participant_ids:
        (n_samples,) — participant ID for each embedding.
    labels:
        (n_samples,) — class label for each embedding.

    Returns
    -------
    pooled_embeddings: (n_participants, embedding_dim)
    pooled_labels: (n_participants,)
    unique_participants: list of participant ID strings
    """
    unique_pids = sorted(set(participant_ids))
    pooled_embs = np.zeros((len(unique_pids), embeddings.shape[1]), dtype=np.float32)
    pooled_labels = np.zeros(len(unique_pids), dtype=labels.dtype)

    for i, pid in enumerate(unique_pids):
        mask = np.array([p == pid for p in participant_ids])
        pooled_embs[i] = embeddings[mask].mean(axis=0)
        # All embeddings for a participant should have the same label
        pooled_labels[i] = labels[mask][0]

    return pooled_embs, pooled_labels, unique_pids


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = np.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
    if den < 1e-12:
        return 0.0
    return float(num / den)
