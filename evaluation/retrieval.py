"""
evaluation/retrieval.py

Cross-modal nearest-neighbour retrieval evaluation.

Two evaluations:
  (A) CDT-vs-single (primary): fused CDT embedding as query → nearest neighbour
      in each single-modality embedding space. Tests whether fusion learns a
      shared participant-level representation.
  (B) Single-vs-single (baseline): each modality pair. Tests encoder alignment.

For each test: recall@1, recall@10, and per-archetype recall@1.
Within-archetype recall@1 chance baseline ≈ 1/143 for balanced archetypes.

Depends on:
  - models/fusion_embeddings_cache.pt (written by fusion/train.py)
  - models/fusion_meta_learner.pt (written by fusion/train.py)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

from schemas import CHECKPOINT_PATHS, PERSONA_LABELS
from fusion.meta_learner import LateFusionMetaLearner

CACHE_PATH = Path("models/fusion_embeddings_cache.pt")
FUSION_CHECKPOINT = CHECKPOINT_PATHS.get(
    "fusion", Path("models/fusion_meta_learner.pt")
)

MODALITIES = ["trace", "transaction", "text", "psychographic"]


def _load_cache(cache_path: Path) -> dict:
    # weights_only=False is required because cache contains
    # a dict with tensors + a list of participant IDs (not just tensors)
    # Safe because cache is created by our own code in fusion/train.py
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def _load_fusion_model(checkpoint_path: Path) -> LateFusionMetaLearner:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Infer n_modalities from fc1 input dim (input_dim = n_modalities * 128).
    n_modalities = state["fc1.weight"].shape[1] // 128
    model = LateFusionMetaLearner(n_modalities=n_modalities)
    model.load_state_dict(state)
    model.eval()
    return model


def _recall_at_k(
    query_embs: torch.Tensor,
    key_embs: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """Fraction of queries whose true match appears in top-k results."""
    # Cosine similarity: [N, N]
    q = F.normalize(query_embs, dim=-1)
    k_ = F.normalize(key_embs, dim=-1)
    sim = q @ k_.T  # [N, N]

    # Exclude self-match when query and key spaces are the same modality
    same_space = query_embs.shape == key_embs.shape and torch.allclose(  # type: ignore[reportPrivateImportUsage]
        query_embs, key_embs
    )
    if same_space:
        sim.fill_diagonal_(-float("inf"))

    topk_indices = sim.topk(k, dim=-1).indices  # [N, k]
    correct = (topk_indices == torch.arange(len(labels)).unsqueeze(1)).any(dim=-1)  # type: ignore[reportPrivateImportUsage]
    return correct.float().mean().item()


def _per_archetype_recall_at_1(
    query_embs: torch.Tensor,
    key_embs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """recall@1 restricted to participants sharing the same archetype label."""
    q = F.normalize(query_embs, dim=-1)
    k = F.normalize(key_embs, dim=-1)
    sim = q @ k.T  # [N, N]

    same_space = query_embs.shape == key_embs.shape and torch.allclose(  # type: ignore[reportPrivateImportUsage]
        query_embs, key_embs
    )

    result: dict[str, float] = {}
    for arch_idx, arch_name in enumerate(PERSONA_LABELS):
        mask = (labels == arch_idx).nonzero(as_tuple=True)[0]
        if len(mask) < 2:
            result[arch_name] = float("nan")
            continue

        sub_sim = sim[mask][:, mask]  # [n_arch, n_arch]
        if same_space:
            sub_sim.fill_diagonal_(-float("inf"))

        best = sub_sim.argmax(dim=-1)  # index within archetype subset
        # Correct if best match is the participant itself (index in mask)
        correct = (best == torch.arange(len(mask))).float().mean().item()  # type: ignore[reportPrivateImportUsage]
        result[arch_name] = correct

    return result


def _cache_modalities(cache: dict) -> list[str]:
    return [m for m in cache if m not in ("labels", "participant_ids")]


def _cdt_embeddings(
    cache: dict[str, torch.Tensor],
    model: LateFusionMetaLearner,
) -> torch.Tensor:
    """Compute CDT embeddings from cached single-modality embeddings."""
    embs = [F.normalize(cache[m], dim=-1) for m in _cache_modalities(cache)]
    fusion_input = torch.cat(embs, dim=-1)  # [N, n_modalities * 128]
    with torch.no_grad():
        cdt = model.embed(fusion_input)  # [N, 128]
    return cdt


def evaluate(
    cache_path: Path = CACHE_PATH,
    checkpoint_path: Path = FUSION_CHECKPOINT,
    log_mlflow: bool = True,
) -> dict:
    """
    Run cross-modal retrieval evaluation.

    Returns
    -------
    dict with keys:
      'cdt_vs_single': {modality: {recall_at_1, recall_at_10, per_archetype_recall_at_1}}
      'single_vs_single': {pair_name: {recall_at_1, recall_at_10, per_archetype_recall_at_1}}
    """
    cache = _load_cache(cache_path)
    model = _load_fusion_model(checkpoint_path)
    labels = cache["labels"]

    cdt_embs = _cdt_embeddings(cache, model)

    # ── (A) CDT-vs-single ────────────────────────────────────────────────────
    cdt_vs_single: dict[str, dict] = {}
    for modality in _cache_modalities(cache):
        key_embs = cache[modality]
        r1 = _recall_at_k(cdt_embs, key_embs, labels, k=1)
        r10 = _recall_at_k(cdt_embs, key_embs, labels, k=10)
        per_arch = _per_archetype_recall_at_1(cdt_embs, key_embs, labels)
        cdt_vs_single[modality] = {
            "recall_at_1": r1,
            "recall_at_10": r10,
            "per_archetype_recall_at_1": per_arch,
        }

    # ── (B) Single-vs-single ─────────────────────────────────────────────────
    single_vs_single: dict[str, dict] = {}
    for m1, m2 in combinations(MODALITIES, 2):
        pair_name = f"{m1}_vs_{m2}"
        embs1 = cache[m1]
        embs2 = cache[m2]
        r1 = _recall_at_k(embs1, embs2, labels, k=1)
        r10 = _recall_at_k(embs1, embs2, labels, k=10)
        per_arch = _per_archetype_recall_at_1(embs1, embs2, labels)
        single_vs_single[pair_name] = {
            "recall_at_1": r1,
            "recall_at_10": r10,
            "per_archetype_recall_at_1": per_arch,
        }

    result = {
        "cdt_vs_single": cdt_vs_single,
        "single_vs_single": single_vs_single,
    }

    if log_mlflow:
        _log_to_mlflow(result)

    return result


def _log_to_mlflow(result: dict) -> None:
    with mlflow.start_run(run_name="retrieval_evaluation"):
        mlflow.set_tag("stage", "retrieval")
        chance_baseline = 1.0 / 143  # ≈ 1/143 for balanced within-archetype

        for modality, metrics in result["cdt_vs_single"].items():
            mlflow.log_metric(f"cdt_vs_{modality}_recall_at_1", metrics["recall_at_1"])
            mlflow.log_metric(
                f"cdt_vs_{modality}_recall_at_10", metrics["recall_at_10"]
            )

        for pair_name, metrics in result["single_vs_single"].items():
            mlflow.log_metric(f"{pair_name}_recall_at_1", metrics["recall_at_1"])
            mlflow.log_metric(f"{pair_name}_recall_at_10", metrics["recall_at_10"])

        mlflow.log_metric("within_archetype_chance_baseline", chance_baseline)


if __name__ == "__main__":
    result = evaluate()
    print("\nCDT-vs-single recall@1:")
    for modality, m in result["cdt_vs_single"].items():
        print(
            f"  {modality:15s}: {m['recall_at_1']:.3f}  (recall@10: {m['recall_at_10']:.3f})"
        )
    print("\nSingle-vs-single recall@1:")
    for pair, m in result["single_vs_single"].items():
        print(
            f"  {pair:30s}: {m['recall_at_1']:.3f}  (recall@10: {m['recall_at_10']:.3f})"
        )
    print(f"\nWithin-archetype chance baseline: {1 / 143:.4f}")
