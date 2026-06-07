"""
evaluation/strategy_recovery.py

Strategy recovery evaluation for the fusion meta-learner.

Loads the trained fusion meta-learner and cached embeddings, then computes:
1. Overall strategy recovery accuracy on val split
2. Per-class accuracy for each persona archetype
3. Confusion matrix
4. Comparison table with single-modality encoder probe accuracies

Acceptance criterion: fused accuracy >85% overall (SPEC's only hard gate).

Note: text and psychographic encoders both achieve 100% individual probe accuracy
because they are near-sufficient statistics for PersonaConfig. Fusion not exceeding
them is expected and reportable, not a failure. See fusion/SPEC.md for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix

from schemas import CHECKPOINT_PATHS, PERSONA_LABELS
from fusion.meta_learner import LateFusionMetaLearner


# Single-modality encoder probe accuracies (from Phase 2a probe runs)
# These are the baseline results fusion is compared against
SINGLE_MODALITY_BASELINES = {
    "trace": 0.9502,  # 95.02% — trace encoder probe
    "transaction": 0.6259,  # 62.59% — transaction encoder probe
    "text": 1.0,  # 100% — text encoder probe
    "psychographic": 1.0,  # 100% — psychographic encoder probe
}


def load_cached_embeddings(
    cache_path: Optional[Path] = None,
) -> dict:
    """Load cached embeddings from fusion training.

    Parameters
    ----------
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt

    Returns
    -------
    dict
        Cached embeddings with keys "trace", "transaction", "text",
        "psychographic", "labels", "participant_ids".
    """
    if cache_path is None:
        cache_path = Path("models/fusion_embeddings_cache.pt")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}. "
            "Run fusion training first (uv run python -m fusion.train)."
        )

    # Load cache — weights_only=False is required because cache contains
    # a dict with tensors + a list of participant IDs (not just tensors)
    # Safe because cache is created by our own code in fusion/train.py
    embeddings = torch.load(cache_path, weights_only=False)
    return embeddings


def split_by_participant(
    participant_ids: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split participant IDs into train/val sets.

    Uses the same seed and ratio as encoder probes to ensure identical val sets.

    Parameters
    ----------
    participant_ids : list[str]
        All participant IDs.
    train_ratio : float
        Fraction of participants assigned to train set.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    tuple[list[str], list[str]]
        (train_ids, val_ids) — participant ID lists for train/val splits.
    """
    rng = np.random.default_rng(seed=seed)
    shuffled_ids = rng.permutation(participant_ids)
    split_idx = int(train_ratio * len(shuffled_ids))
    train_ids = shuffled_ids[:split_idx].tolist()
    val_ids = shuffled_ids[split_idx:].tolist()
    return train_ids, val_ids


def evaluate_fusion(
    model: LateFusionMetaLearner,
    embeddings: dict,
    device: str = "cpu",
) -> dict:
    """Evaluate fusion meta-learner on val split.

    Parameters
    ----------
    model : LateFusionMetaLearner
        Trained fusion meta-learner.
    embeddings : dict
        Cached embeddings with keys for each modality plus labels/participant_ids.
    device : str
        Target device ("cpu" or "cuda").

    Returns
    -------
    dict
        Evaluation metrics with keys:
        - val_accuracy: float — overall strategy recovery accuracy
        - train_accuracy: float — train set accuracy (for overfitting check)
        - per_class_accuracy: dict[str, float] — accuracy per persona archetype
        - confusion_matrix: np.ndarray — (7, 7) confusion matrix on val set
        - comparison_table: dict — fusion vs single-modality baseline comparison
        - n_val_participants: int — number of participants in val set
        - n_train_participants: int — number of participants in train set
    """
    model.eval()

    # Split participants
    participant_ids = embeddings["participant_ids"]
    train_ids, val_ids = split_by_participant(participant_ids)

    # Create index mappings
    participant_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
    train_indices = torch.tensor([participant_to_idx[pid] for pid in train_ids])  # type: ignore[reportPrivateImportUsage]
    val_indices = torch.tensor([participant_to_idx[pid] for pid in val_ids])  # type: ignore[reportPrivateImportUsage]

    # Extract embeddings and labels
    train_embs = {
        mod: embeddings[mod][train_indices]
        for mod in ["trace", "transaction", "text", "psychographic"]
    }
    train_labels = embeddings["labels"][train_indices]

    val_embs = {
        mod: embeddings[mod][val_indices]
        for mod in ["trace", "transaction", "text", "psychographic"]
    }
    val_labels = embeddings["labels"][val_indices]

    # Build fusion input: concatenate L2-normalised embeddings
    def build_fusion_input(embs_dict, indices):
        # Stack modality embeddings: [N, 4, 128]
        emb_list = [
            embs_dict[mod][indices]
            for mod in ["trace", "transaction", "text", "psychographic"]
        ]
        stacked = torch.stack(emb_list, dim=1)  # [N, 4, 128]

        # L2-normalise each modality
        norm_embs = [F.normalize(stacked[:, i], p=2, dim=-1) for i in range(4)]
        fusion_input = torch.cat(norm_embs, dim=-1)  # type: ignore[reportPrivateImportUsage]
        return fusion_input  # [N, 512]

    train_input = build_fusion_input(train_embs, torch.arange(len(train_ids)))  # type: ignore[reportPrivateImportUsage]
    val_input = build_fusion_input(val_embs, torch.arange(len(val_ids)))  # type: ignore[reportPrivateImportUsage]

    # Run inference
    with torch.no_grad():
        train_input = train_input.to(device)
        val_input = val_input.to(device)

        train_logits = model(train_input)
        val_logits = model(val_input)

        train_preds = train_logits.argmax(dim=-1).cpu().numpy()
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()

    train_labels_np = train_labels.cpu().numpy()
    val_labels_np = val_labels.cpu().numpy()

    # Compute metrics
    train_acc = accuracy_score(train_labels_np, train_preds)
    val_acc = accuracy_score(val_labels_np, val_preds)

    cm = confusion_matrix(val_labels_np, val_preds)

    # Per-class accuracy
    per_class_acc: dict[str, float] = {}
    for cls_idx in range(len(PERSONA_LABELS)):
        mask = val_labels_np == cls_idx
        if mask.any():
            per_class_acc[str(cls_idx)] = accuracy_score(
                val_labels_np[mask], val_preds[mask]
            )
        else:
            per_class_acc[str(cls_idx)] = float("nan")

    # Build comparison table
    comparison_table = {
        "fusion": val_acc,
        **SINGLE_MODALITY_BASELINES,
    }

    return {
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "comparison_table": comparison_table,
        "n_val_participants": len(val_ids),
        "n_train_participants": len(train_ids),
    }


def run_strategy_recovery(
    *,
    cache_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    """Run full strategy recovery evaluation.

    Parameters
    ----------
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt
    checkpoint_path : Path | None
        Path to fusion checkpoint. Default: schemas.CHECKPOINT_PATHS["fusion"]
    device : str
        Target device ("cpu" or "cuda").

    Returns
    -------
    dict
        Evaluation metrics from evaluate_fusion().
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS["fusion"]

    # Load cached embeddings
    embeddings = load_cached_embeddings(cache_path)

    # Load trained fusion model
    model = LateFusionMetaLearner(phase="2")  # Phase 2 is the default
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {checkpoint_path}. "
            "Run fusion training first (uv run python -m fusion.train)."
        )

    # Evaluate
    results = evaluate_fusion(model, embeddings, device=device)
    return results


def format_results(results: dict) -> str:
    """Format evaluation results for terminal output.

    Parameters
    ----------
    results : dict
        Results from run_strategy_recovery().

    Returns
    -------
    str
        Formatted results string.
    """
    lines = [
        "=" * 70,
        "Fusion Strategy Recovery Results",
        "=" * 70,
        f"Overall Accuracy (val): {results['val_accuracy']:.2%}",
        f"Train Accuracy: {results['train_accuracy']:.2%}",
        f"Val Participants: {results['n_val_participants']}",
        f"Train Participants: {results['n_train_participants']}",
        "",
        "Per-class accuracy (val):",
    ]

    for cls_name, acc in sorted(results["per_class_accuracy"].items()):
        cls_idx = int(cls_name)
        label = PERSONA_LABELS[cls_idx]
        lines.append(f"  {label:20s}: {acc:.2%}")

    lines.extend(
        [
            "",
            "Comparison with single-modality encoders:",
            "  " + "-" * 50,
        ]
    )

    comparison = results["comparison_table"]
    lines.append(f"  {'Modality':20s}: {'Accuracy':>10s}")
    lines.append("  " + "-" * 50)

    # Order: fusion first, then single modalities
    for modality in ["fusion", "trace", "transaction", "text", "psychographic"]:
        acc = comparison.get(modality, float("nan"))
        if modality == "fusion":
            lines.append(f"  {modality:20s}: {acc:>9.2%} ← fused meta-learner")
        else:
            lines.append(f"  {modality:20s}: {acc:>9.2%}")

    lines.extend(
        [
            "",
            "Confusion matrix (val):",
            str(results["confusion_matrix"]),
            "",
        ]
    )

    # Check acceptance criterion
    val_acc = results["val_accuracy"]
    passed = val_acc > 0.85
    lines.extend(
        [
            "=" * 70,
        ]
    )

    if passed:
        lines.append(f"✓ PASS: Strategy recovery {val_acc:.2%} > 85%")
    else:
        lines.append(f"✗ FAIL: Strategy recovery {val_acc:.2%} ≤ 85%")

    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = run_strategy_recovery(device=device)
    print(format_results(results))

    # Exit code based on acceptance criterion
    sys.exit(0 if results["val_accuracy"] > 0.85 else 1)
