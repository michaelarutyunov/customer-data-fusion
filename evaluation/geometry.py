"""
evaluation/geometry.py

Geometry evaluation for fused CDT embeddings via UMAP projection.

Computes:
1. UMAP 2D projection of fused CDT embeddings [N, 128]
2. Mean intra-persona cosine distance
3. Mean inter-persona cosine distance
4. Silhouette score per persona

Saves enriched JSON output to data/synthetic/umap_fused.json with:
- participant_id, umap_x, umap_y (for visualization)
- archetype label (for between-persona coloring)
- 7 PersonaConfig float params (for within-persona coloring)

The within-persona coloring tests whether the CDT embedding preserves
individual variation or collapses all members of an archetype to a single point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from umap import UMAP

from schemas import CHECKPOINT_PATHS, PARTICIPANT_CONFIG_PATH, PERSONA_LABELS
from evaluation.strategy_recovery import load_cached_embeddings
from fusion.meta_learner import LateFusionMetaLearner


def load_participant_configs(
    config_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load participant configs with continuous latent variables.

    Parameters
    ----------
    config_path : Path | None
        Path to participant_configs.jsonl. Default: schemas.PARTICIPANT_CONFIG_PATH

    Returns
    -------
    dict[str, dict]
        Dictionary mapping participant_id to config dict with 7 float params.
    """
    if config_path is None:
        config_path = PARTICIPANT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Participant configs not found: {config_path}. "
            "Run generator pipeline with participant config output (bead c33)."
        )

    configs: dict[str, dict] = {}
    for line in config_path.read_text().strip().splitlines():
        record = json.loads(line)
        participant_id = record["participant_id"]
        configs[participant_id] = record

    return configs


def extract_cdt_embeddings(
    model: LateFusionMetaLearner,
    embeddings: dict,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Extract CDT embeddings for all participants using fusion model.

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
    tuple[torch.Tensor, list[str], list[int]]
        (cdt_embeddings, participant_ids, labels)
        - cdt_embeddings: [N, 128] CDT embeddings
        - participant_ids: list of N participant ID strings
        - labels: list of N integer persona labels
    """
    model.eval()

    # Get all participants
    participant_ids = embeddings["participant_ids"]
    n_participants = len(participant_ids)

    # Create index mappings
    indices = torch.arange(n_participants)  # type: ignore[reportPrivateImportUsage]

    # Extract embeddings for all modalities
    all_embs = {
        mod: embeddings[mod][indices]
        for mod in ["trace", "transaction", "text", "psychographic"]
    }

    # Build fusion input: concatenate L2-normalised embeddings
    emb_list = [
        all_embs[mod][indices]
        for mod in ["trace", "transaction", "text", "psychographic"]
    ]
    stacked = torch.stack(emb_list, dim=1)  # [N, 4, 128]

    # L2-normalise each modality
    norm_embs = [F.normalize(stacked[:, i], p=2, dim=-1) for i in range(4)]
    fusion_input = torch.cat(norm_embs, dim=-1)  # type: ignore[reportPrivateImportUsage]

    # Extract CDT embeddings
    with torch.no_grad():
        fusion_input = fusion_input.to(device)
        cdt_embeddings = model.embed(fusion_input)  # [N, 128]

    labels = embeddings["labels"].tolist()

    return cdt_embeddings, participant_ids, labels


def compute_geometric_metrics(
    cdt_embeddings: np.ndarray,
    labels: list[int],
) -> dict:
    """Compute geometric metrics on CDT embeddings.

    Parameters
    ----------
    cdt_embeddings : np.ndarray
        [N, 128] CDT embeddings.
    labels : list[int]
        Persona labels for each participant.

    Returns
    -------
    dict
        Geometric metrics with keys:
        - mean_intra_cosine_distance: float
        - mean_inter_cosine_distance: float
        - silhouette_scores: dict[str, float] — silhouette per persona
        - mean_silhouette: float — overall silhouette score
    """
    # L2-normalise embeddings for cosine distance
    norms = np.linalg.norm(cdt_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings_norm = cdt_embeddings / norms

    # Cosine similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T  # [N, N]

    unique_labels = sorted(set(labels))

    # Per-class masks
    class_masks = {lbl: np.array(labels) == lbl for lbl in unique_labels}

    # Intra-class cosine distances
    intra_distances: list[float] = []
    for lbl in unique_labels:
        mask = class_masks[lbl]
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        # Upper triangle of the sub-matrix for this class
        sub_sim = sim_matrix[np.ix_(indices, indices)]
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        intra_sims = sub_sim[triu_idx]
        # Convert similarity to distance: distance = 1 - similarity
        intra_distances.extend((1 - intra_sims).tolist())

    mean_intra_distance = (
        float(np.mean(intra_distances)) if intra_distances else float("nan")
    )

    # Inter-class cosine distances
    inter_distances: list[float] = []
    for i, lbl_i in enumerate(unique_labels):
        for j, lbl_j in enumerate(unique_labels):
            if i >= j:
                continue
            mask_i = class_masks[lbl_i]
            mask_j = class_masks[lbl_j]
            sub_sim = sim_matrix[np.ix_(mask_i, mask_j)]
            inter_distances.extend((1 - sub_sim.flatten()).tolist())

    mean_inter_distance = (
        float(np.mean(inter_distances)) if inter_distances else float("nan")
    )

    # Silhouette scores
    # Overall silhouette score
    mean_silhouette = silhouette_score(cdt_embeddings, labels)

    # Per-class silhouette scores
    silhouette_scores: dict[str, float] = {}
    for lbl in unique_labels:
        mask = np.array(labels) == lbl
        if mask.sum() < 2:
            silhouette_scores[str(lbl)] = float("nan")
            continue

        # Silhouette score for this class (comparing to all other classes)
        try:
            class_silhouette = silhouette_score(
                cdt_embeddings, labels, sample_mask=mask
            )
            silhouette_scores[str(lbl)] = float(class_silhouette)
        except ValueError:
            # Not enough samples or other issue
            silhouette_scores[str(lbl)] = float("nan")

    return {
        "mean_intra_cosine_distance": mean_intra_distance,
        "mean_inter_cosine_distance": mean_inter_distance,
        "silhouette_scores": silhouette_scores,
        "mean_silhouette": mean_silhouette,
    }


def run_geometry(
    *,
    cache_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    device: str = "cpu",
    output_path: Optional[Path] = None,
) -> dict:
    """Run geometry evaluation on fused CDT embeddings.

    Parameters
    ----------
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt
    checkpoint_path : Path | None
        Path to fusion checkpoint. Default: schemas.CHECKPOINT_PATHS["fusion"]
    config_path : Path | None
        Path to participant_configs.jsonl. Default: schemas.PARTICIPANT_CONFIG_PATH
    umap_n_neighbors : int
        UMAP n_neighbors parameter (local vs global structure).
    umap_min_dist : float
        UMAP min_dist parameter (cluster tightness).
    device : str
        Target device ("cpu" or "cuda").
    output_path : Path | None
        Path to save UMAP JSON output. Default: data/synthetic/umap_fused.json

    Returns
    -------
    dict
        Geometry results with keys:
        - cdt_embeddings: np.ndarray — [N, 128] CDT embeddings
        - umap_coordinates: np.ndarray — [N, 2] UMAP projection
        - geometric_metrics: dict — intra/inter distances, silhouette scores
        - n_participants: int — total number of participants
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS["fusion"]
    if output_path is None:
        output_path = Path("data/synthetic/umap_fused.json")

    # Load data
    embeddings = load_cached_embeddings(cache_path)
    participant_configs = load_participant_configs(config_path)

    # Load fusion model
    model = LateFusionMetaLearner(phase="2")
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {checkpoint_path}. "
            "Run fusion training first (uv run python -m fusion.train)."
        )

    # Extract CDT embeddings
    cdt_embeddings, participant_ids, labels = extract_cdt_embeddings(
        model, embeddings, device=device
    )
    cdt_embeddings_np = cdt_embeddings.cpu().numpy()

    # Compute geometric metrics
    geometric_metrics = compute_geometric_metrics(cdt_embeddings_np, labels)

    # UMAP projection
    umap_reducer = UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    umap_coordinates = umap_reducer.fit_transform(cdt_embeddings_np)

    # Save enriched JSON output for notebook visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    umap_data: list[dict] = []

    for i, participant_id in enumerate(participant_ids):
        label = labels[i]
        archetype = PERSONA_LABELS[label]
        config = participant_configs.get(participant_id, {})

        umap_x = float(umap_coordinates[i, 0])  # type: ignore[index]
        umap_y = float(umap_coordinates[i, 1])  # type: ignore[index]

        record = {
            "participant_id": participant_id,
            "umap_x": umap_x,
            "umap_y": umap_y,
            "archetype": archetype,
            "price_sensitivity": config.get("price_sensitivity", float("nan")),
            "brand_loyalty": config.get("brand_loyalty", float("nan")),
            "inspection_depth": config.get("inspection_depth", float("nan")),
            "maximiser_score": config.get("maximiser_score", float("nan")),
            "involvement_score": config.get("involvement_score", float("nan")),
            "risk_tolerance": config.get("risk_tolerance", float("nan")),
            "p_strategy_lapse": config.get("p_strategy_lapse", float("nan")),
        }
        umap_data.append(record)

    with output_path.open("w") as f:
        for record in umap_data:
            f.write(json.dumps(record) + "\n")

    return {
        "cdt_embeddings": cdt_embeddings_np,
        "umap_coordinates": umap_coordinates,
        "geometric_metrics": geometric_metrics,
        "n_participants": len(participant_ids),
        "umap_output_path": str(output_path),
    }


def format_results(results: dict) -> str:
    """Format geometry results for terminal output.

    Parameters
    ----------
    results : dict
        Results from run_geometry().

    Returns
    -------
    str
        Formatted results string.
    """
    metrics = results["geometric_metrics"]

    lines = [
        "=" * 70,
        "Fusion CDT Embedding Geometry Results",
        "=" * 70,
        f"Total participants: {results['n_participants']}",
        f"UMAP output: {results['umap_output_path']}",
        "",
        "Geometric metrics (CDT embeddings, 128-dim):",
        f"  Mean intra-persona cosine distance: {metrics['mean_intra_cosine_distance']:.4f}",
        f"  Mean inter-persona cosine distance: {metrics['mean_inter_cosine_distance']:.4f}",
        f"  Overall silhouette score: {metrics['mean_silhouette']:.4f}",
        "",
        "Per-persona silhouette scores:",
    ]

    for cls_name, sil in sorted(metrics["silhouette_scores"].items()):
        cls_idx = int(cls_name)
        archetype = PERSONA_LABELS[cls_idx]
        if not np.isnan(sil):
            lines.append(f"  {archetype:20s}: {sil:.4f}")
        else:
            lines.append(f"  {archetype:20s}: N/A (insufficient samples)")

    lines.extend(
        [
            "",
            "Interpretation:",
            "  - Intra-persona distance: Lower = tighter archetype clusters",
            "  - Inter-persona distance: Higher = better between-archetype separation",
            "  - Silhouette score: Range [-1, 1]. Higher = better-defined clusters",
            "    - >0.5: strong structure",
            "    - 0.2-0.5: moderate structure",
            "    - <0.2: weak structure",
            "",
            "UMAP coordinates saved with PersonaConfig params for within-persona coloring.",
            "  Within-archetype gradient (e.g., by price_sensitivity) indicates",
            "  preserved individual variation. Collapse to single point indicates",
            "  archetype-only encoding (loss of participant-level structure).",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = run_geometry(device=device)
    print(format_results(results))
