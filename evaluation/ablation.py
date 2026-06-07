"""
evaluation/ablation.py

Leave-one-out modality ablation for fusion meta-learner.

For each of the 4 modalities, zero out its 128-dim slice of the 512-dim fusion
input and re-evaluate accuracy on the val split. Reports delta from full-modality
baseline.

Output: modality importance ranking by accuracy delta when removed; flags any
modality with delta <5% as a diagnostic finding (low delta is expected for
correlated modalities like text/psychographic, not a failure — see SPEC).

All runs logged to MLflow. The only hard gate remains the full-modality
accuracy >85% from fusion/train.py (evaluated in strategy_recovery.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from schemas import CHECKPOINT_PATHS
from evaluation.strategy_recovery import (
    load_cached_embeddings,
    split_by_participant,
)
from fusion.meta_learner import LateFusionMetaLearner


MODALITIES = ["trace", "transaction", "text", "psychographic"]
LOW_DELTA_THRESHOLD = 0.05  # 5% — flag modalities with smaller accuracy drops


def build_fusion_input(
    embeddings_dict: dict,
    indices: torch.Tensor,
    ablate_modality: Optional[str] = None,
) -> torch.Tensor:
    """Build fusion input with optional modality ablation.

    Parameters
    ----------
    embeddings_dict : dict
        Dictionary mapping modality names to embedding tensors.
    indices : torch.Tensor
        Participant indices to extract.
    ablate_modality : str | None
        If specified, zero out this modality's 128-dim slice.

    Returns
    -------
    torch.Tensor
        Concatenated L2-normalised embeddings, shape [N, 512].
    """
    # Stack modality embeddings: [N, 4, 128]
    emb_list = [embeddings_dict[mod][indices] for mod in MODALITIES]
    stacked = torch.stack(emb_list, dim=1)  # [N, 4, 128]

    # Zero out ablated modality
    if ablate_modality is not None:
        mod_idx = MODALITIES.index(ablate_modality)
        stacked[:, mod_idx, :] = 0.0

    # L2-normalise each modality
    norm_embs = [F.normalize(stacked[:, i], p=2, dim=-1) for i in range(4)]
    fusion_input = torch.cat(norm_embs, dim=-1)  # type: ignore[reportPrivateImportUsage]
    return fusion_input  # [N, 512]


def evaluate_ablation(
    model: LateFusionMetaLearner,
    embeddings: dict,
    device: str = "cpu",
) -> dict:
    """Evaluate fusion with each modality ablated (leave-one-out).

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
        Ablation results with keys:
        - full_modality_accuracy: float — baseline with all 4 modalities
        - ablation_results: dict[str, dict] — results per ablated modality
        - importance_ranking: list[tuple[str, float]] — ranked by accuracy delta
        - low_delta_findings: list[str] — modalities with delta <5%
        - n_val_participants: int — number of participants in val set
    """
    model.eval()

    # Split participants
    participant_ids = embeddings["participant_ids"]
    _, val_ids = split_by_participant(participant_ids)

    # Create index mappings
    participant_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
    val_indices = torch.tensor([participant_to_idx[pid] for pid in val_ids])  # type: ignore[reportPrivateImportUsage]

    # Extract embeddings and labels
    val_embs = {mod: embeddings[mod][val_indices] for mod in MODALITIES}
    val_labels = embeddings["labels"][val_indices]

    # Build full-modality input (baseline)
    val_input_full = build_fusion_input(
        val_embs,
        torch.arange(len(val_ids)),  # type: ignore[reportPrivateImportUsage]
    )

    # Evaluate full-modality baseline
    with torch.no_grad():
        val_input_full = val_input_full.to(device)
        logits_full = model(val_input_full)
        preds_full = logits_full.argmax(dim=-1).cpu().numpy()

    val_labels_np = val_labels.cpu().numpy()
    full_acc = accuracy_score(val_labels_np, preds_full)

    # Ablation results
    ablation_results: dict[str, dict] = {}

    for modality in MODALITIES:
        # Build input with this modality zeroed out
        val_input_ablated = build_fusion_input(
            val_embs,
            torch.arange(len(val_ids)),  # type: ignore[reportPrivateImportUsage]
            ablate_modality=modality,
        )

        # Evaluate
        with torch.no_grad():
            val_input_ablated = val_input_ablated.to(device)
            logits_ablated = model(val_input_ablated)
            preds_ablated = logits_ablated.argmax(dim=-1).cpu().numpy()

        ablated_acc = accuracy_score(val_labels_np, preds_ablated)
        delta = full_acc - ablated_acc

        ablation_results[modality] = {
            "accuracy": ablated_acc,
            "delta": delta,
            "delta_pct": delta * 100,
        }

    # Build importance ranking (largest delta = most important)
    importance_ranking = sorted(
        [(mod, ablation_results[mod]["delta"]) for mod in MODALITIES],
        key=lambda x: x[1],
        reverse=True,
    )

    # Flag low-delta findings (<5% accuracy drop)
    low_delta_findings = [
        mod for mod, delta in importance_ranking if delta < LOW_DELTA_THRESHOLD
    ]

    return {
        "full_modality_accuracy": full_acc,
        "ablation_results": ablation_results,
        "importance_ranking": importance_ranking,
        "low_delta_findings": low_delta_findings,
        "n_val_participants": len(val_ids),
    }


def run_ablation(
    *,
    cache_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
    log_mlflow: bool = True,
) -> dict:
    """Run full ablation evaluation.

    Parameters
    ----------
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt
    checkpoint_path : Path | None
        Path to fusion checkpoint. Default: schemas.CHECKPOINT_PATHS["fusion"]
    device : str
        Target device ("cpu" or "cuda").
    log_mlflow : bool
        Whether to log results to MLflow.

    Returns
    -------
    dict
        Ablation results from evaluate_ablation().
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
    results = evaluate_ablation(model, embeddings, device=device)

    # Log to MLflow
    if log_mlflow:
        with mlflow.start_run(run_name="fusion_ablation_v1"):
            mlflow.set_tag("modality", "fusion")
            mlflow.set_tag("task", "ablation")
            mlflow.log_metric(
                "full_modality_accuracy", results["full_modality_accuracy"]
            )
            mlflow.log_metric("n_val_participants", results["n_val_participants"])

            for modality, result in results["ablation_results"].items():
                mlflow.log_metric(f"ablated_{modality}_accuracy", result["accuracy"])
                mlflow.log_metric(f"ablated_{modality}_delta", result["delta"])
                mlflow.log_metric(f"ablated_{modality}_delta_pct", result["delta_pct"])

            # Log findings
            if results["low_delta_findings"]:
                mlflow.set_tag(
                    "low_delta_findings",
                    ",".join(results["low_delta_findings"]),
                )

    return results


def format_results(results: dict) -> str:
    """Format ablation results for terminal output.

    Parameters
    ----------
    results : dict
        Results from run_ablation().

    Returns
    -------
    str
        Formatted results string.
    """
    lines = [
        "=" * 70,
        "Fusion Modality Ablation Results",
        "=" * 70,
        f"Full-modality accuracy (val): {results['full_modality_accuracy']:.2%}",
        f"Val participants: {results['n_val_participants']}",
        "",
        "Leave-one-out ablation (zero out each modality's 128-dim slice):",
        "",
    ]

    # Ablation table
    lines.append(
        f"{'Modality':20s} {'Ablated Acc':>12s} {'Delta':>10s} {'% Drop':>10s}"
    )
    lines.append(" " * 20 + " " * 12 + "-" * 10 + "-" * 11)

    for modality in MODALITIES:
        result = results["ablation_results"][modality]
        delta_flag = " ⚠️" if result["delta"] < LOW_DELTA_THRESHOLD else ""
        lines.append(
            f"{modality:20s} {result['accuracy']:>11.2%} {result['delta']:>9.2%} {result['delta_pct']:>9.1f}%{delta_flag}"
        )

    lines.extend(
        [
            "",
            "Modality importance ranking (by accuracy delta when removed):",
        ]
    )

    for i, (modality, delta) in enumerate(results["importance_ranking"], 1):
        low_flag = " ⚠️ <5% drop" if delta < LOW_DELTA_THRESHOLD else ""
        lines.append(f"  {i}. {modality:20s}: {delta:>7.2%} accuracy drop{low_flag}")

    # Findings
    if results["low_delta_findings"]:
        lines.extend(
            [
                "",
                "⚠️  Low-delta findings (<5% accuracy drop):",
            ]
        )
        for modality in results["low_delta_findings"]:
            lines.append(
                f"    - {modality}: {results['ablation_results'][modality]['delta']:.2%}"
            )
        lines.append(
            "    Note: Low delta is expected for correlated modalities "
            "(e.g., text/psychographic both encode PersonaConfig)."
        )
        lines.append("    This is a diagnostic finding, not a failure.")

    lines.extend(
        [
            "",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = run_ablation(device=device)
    print(format_results(results))
