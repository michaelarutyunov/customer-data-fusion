"""
evaluation/config_probe.py

Linear regression probe for PersonaConfig continuous latent parameters.

For each of 7 PersonaConfig float parameters, trains a Ridge regression on
each of 5 embedding sets (fused CDT + 4 single-modality). Reports R² on the
val split. Tests whether the fused CDT embedding recovers latent behavioural
parameters better than any single modality.

Interpretation:
  - Fused R² > all single-modality R² for a parameter → fusion adds information
  - High single-modality R² (e.g. psychographic for price_sensitivity) → that
    modality encodes the parameter directly; fusion should match or exceed it
  - Negative R² is valid: Ridge with poor fit produces negative R² on val split

Depends on:
  - models/fusion_embeddings_cache.pt (written by fusion/train.py)
  - models/fusion_meta_learner.pt (written by fusion/train.py)
  - data/synthetic/participant_configs.jsonl (written by generator/pipeline.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from schemas import CHECKPOINT_PATHS, PARTICIPANT_CONFIG_PATH
from fusion.meta_learner import LateFusionMetaLearner

CACHE_PATH = Path("models/fusion_embeddings_cache.pt")
FUSION_CHECKPOINT = CHECKPOINT_PATHS.get(
    "fusion", Path("models/fusion_meta_learner.pt")
)

CONFIG_PARAMS = [
    "price_sensitivity",
    "brand_loyalty",
    "inspection_depth",
    "maximiser_score",
    "involvement_score",
    "risk_tolerance",
    "p_strategy_lapse",
]

MODALITIES = ["fused", "trace", "transaction", "text", "psychographic"]


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


def _load_participant_configs(path: Path) -> dict[str, dict]:
    """Load participant_configs.jsonl keyed by participant_id."""
    configs: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            configs[record["participant_id"]] = record
    return configs


def _cache_modalities(cache: dict) -> list[str]:
    """Modality tensor keys in the cache (excludes 'labels'/'participant_ids')."""
    return [m for m in cache if m not in ("labels", "participant_ids")]


def _cdt_embeddings(
    cache: dict,
    model: LateFusionMetaLearner,
) -> torch.Tensor:
    embs = [F.normalize(cache[m], dim=-1) for m in _cache_modalities(cache)]
    fusion_input = torch.cat(embs, dim=-1)  # [N, n_modalities * 128]
    with torch.no_grad():
        return model.embed(fusion_input)


def probe(
    cache_path: Path = CACHE_PATH,
    checkpoint_path: Path = FUSION_CHECKPOINT,
    config_path: Path = PARTICIPANT_CONFIG_PATH,
    log_mlflow: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run PersonaConfig regression probes.

    Returns
    -------
    dict[param_name, dict[modality_name, r2_val]]
    Seven outer keys × five inner keys. All values are floats.
    """
    cache = _load_cache(cache_path)
    model = _load_fusion_model(checkpoint_path)
    participant_ids: list[str] = cache["participant_ids"]

    participant_configs = _load_participant_configs(config_path)

    # Build target matrix aligned to cache ordering
    targets: dict[str, list[float]] = {p: [] for p in CONFIG_PARAMS}
    for pid in participant_ids:
        cfg = participant_configs[pid]
        for param in CONFIG_PARAMS:
            targets[param].append(cfg[param])

    target_arrays = {p: np.array(v, dtype=np.float32) for p, v in targets.items()}

    # Build embedding arrays per modality
    cdt_embs = _cdt_embeddings(cache, model).numpy()
    embedding_arrays: dict[str, np.ndarray] = {"fused": cdt_embs}
    for m in _cache_modalities(cache):
        embedding_arrays[m] = cache[m].numpy()

    # Train/val split: same participant_ids ordering used in fusion training
    # Replicate split_participants(seed=42) logic: first 80% train, last 20% val
    n = len(participant_ids)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:]

    results: dict[str, dict[str, float]] = {}

    for param in CONFIG_PARAMS:
        y = target_arrays[param]
        y_train, y_val = y[train_idx], y[val_idx]
        results[param] = {}

        for modality in MODALITIES:
            X = embedding_arrays[modality]
            X_train, X_val = X[train_idx], X[val_idx]

            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_val)
            r2 = float(r2_score(y_val, y_pred))
            results[param][modality] = r2

    if log_mlflow:
        _log_to_mlflow(results)

    return results


def _log_to_mlflow(results: dict[str, dict[str, float]]) -> None:
    with mlflow.start_run(run_name="config_probe"):
        mlflow.set_tag("stage", "config_probe")
        for param, modality_scores in results.items():
            for modality, r2 in modality_scores.items():
                mlflow.log_metric(f"{param}__{modality}_r2", r2)


if __name__ == "__main__":
    results = probe()
    header = f"{'Parameter':<25}" + "".join(f"{m:>14}" for m in MODALITIES)
    print(header)
    print("-" * len(header))
    for param, scores in results.items():
        row = f"{param:<25}" + "".join(f"{scores[m]:>14.3f}" for m in MODALITIES)
        print(row)
