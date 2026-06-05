"""
Psychographic encoder probe — strategy recovery + raw features comparison.

Evaluates the psychographic encoder via frozen + logistic regression probe.
Additionally compares MLP embeddings against raw 22-dim features (no MLP).

Acceptance (szm.13): strategy recovery >75%; MLP outperforms raw features baseline.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

from evaluation.probe import probe_logistic_regression

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
MODEL_DIR = Path("models")
PSYCHO_MODEL_PATH = MODEL_DIR / "psychographic_encoder.pt"

PERSONA_LABELS = [
    "price_lex",
    "compensatory",
    "satisficer",
    "brand_affect",
    "quality_lex",
    "adaptive",
    "low_involve",
]
PERSONA_TO_IDX = {p: i for i, p in enumerate(PERSONA_LABELS)}


def load_psychographic_data(
    path: Path = DATA_DIR / "psychographics.jsonl",
) -> tuple[list[dict], dict[str, str]]:
    """Load psychographic records and return (records, participant_to_persona)."""
    records: list[dict] = []
    pid_to_persona: dict[str, str] = {}
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        records.append(r)
        pid_to_persona[r["participant_id"]] = r.get("persona_id", "unknown")
    return records, pid_to_persona


def split_participants(
    records: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split participants into train/val sets."""
    participant_ids = sorted(set(r["participant_id"] for r in records))
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)
    split_idx = int(train_ratio * len(participant_ids))
    return set(participant_ids[:split_idx]), set(participant_ids[split_idx:])


def generate_psychographic_embeddings(
    encoder,
    records: list[dict],
    relevant_pids: set[str],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate embeddings and raw features for relevant participants.

    Returns (embeddings, raw_features, labels).
    """
    from encoders.psychographic.features import to_feature_vector
    from schemas.psychographic import PsychographicVector

    # Filter to relevant participants
    filtered = [r for r in records if r["participant_id"] in relevant_pids]

    embeddings_list: list[np.ndarray] = []
    raw_features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    encoder.eval()
    with torch.no_grad():
        for r in filtered:
            psych = PsychographicVector(
                participant_id=r["participant_id"],
                persona_id=r.get("persona_id", "unknown"),
                involvement_score=r.get("involvement_score", 0.5),
                maximiser_score=r.get("maximiser_score", 0.5),
                risk_tolerance=r.get("risk_tolerance", 0.5),
                price_consciousness=r.get("price_consciousness", 0.5),
                brand_sensitivity=r.get("brand_sensitivity", 0.5),
                openness_to_new=r.get("openness_to_new", 0.5),
                decision_style_dominant=r.get("decision_style_dominant", "analytical"),
                age_band=r.get("age_band", "25-34"),
                household_type=r.get("household_type", "single"),
                employment_status=r.get("employment_status", "full_time"),
                category=r.get("category", "electronics"),
                purchase_frequency_band=r.get("purchase_frequency_band", "monthly"),
                years_buying_category=r.get("years_buying_category"),
            )
            raw_vec = to_feature_vector(psych).to(device)  # (22,)
            raw_features_list.append(raw_vec.cpu().numpy())

            # Forward through encoder
            emb = encoder(raw_vec.unsqueeze(0)).cpu().numpy().squeeze(0)
            embeddings_list.append(emb)

            persona = r.get("persona_id", "unknown")
            labels_list.append(PERSONA_TO_IDX.get(persona, 0))

    embeddings = np.stack(embeddings_list)
    raw_features = np.stack(raw_features_list)
    labels = np.array(labels_list)
    return embeddings, raw_features, labels


def probe(
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run the psychographic encoder probe evaluation."""
    from encoders.psychographic.model import PsychographicEncoder

    records, pid_to_persona = load_psychographic_data()
    train_pids, val_pids = split_participants(
        records, train_ratio=train_ratio, seed=seed
    )
    logger.info(
        "Train participants: %d, Val participants: %d", len(train_pids), len(val_pids)
    )

    # Build or load encoder
    if PSYCHO_MODEL_PATH.exists():
        logger.info("Loading psychographic encoder from %s", PSYCHO_MODEL_PATH)
        encoder = PsychographicEncoder().to(device)
        state = torch.load(PSYCHO_MODEL_PATH, map_location=device, weights_only=True)
        encoder.load_state_dict(state, strict=False)
    else:
        logger.info("No checkpoint found — training psychographic encoder from scratch")
        from encoders.psychographic.train import train as train_psycho

        encoder = train_psycho(
            n_epochs=40,
            batch_size=128,
            device=device,
            log_mlflow=False,
        )

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Generate embeddings
    embeddings, raw_features, labels = generate_psychographic_embeddings(
        encoder, records, train_pids | val_pids, device
    )

    # Build train/val index masks
    all_pids = [
        r["participant_id"]
        for r in records
        if r["participant_id"] in (train_pids | val_pids)
    ]
    train_mask = np.array([pid in train_pids for pid in all_pids])
    val_mask = np.array([pid in val_pids for pid in all_pids])
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # MLP embedding probe
    result = probe_logistic_regression(embeddings, labels, train_idx, val_idx)

    # Raw features baseline probe
    raw_result = probe_logistic_regression(raw_features, labels, train_idx, val_idx)

    # Log to MLflow
    with mlflow.start_run(run_name="psychographic_probe_v1"):
        mlflow.set_tag("modality", "psychographic")
        mlflow.set_tag("task", "probe")
        mlflow.log_metric("strategy_recovery_acc", result["val_accuracy"])
        mlflow.log_metric(
            "raw_features_strategy_recovery_acc", raw_result["val_accuracy"]
        )

    result["raw_features_val_accuracy"] = raw_result["val_accuracy"]
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    result = probe(device=device)

    print("\n" + "=" * 60)
    print("Psychographic Encoder Probe Results")
    print("=" * 60)
    print(f"MLP Embedding Strategy Recovery: {result['val_accuracy']:.2%}")
    print(f"Raw Features (22-dim) Baseline:  {result['raw_features_val_accuracy']:.2%}")
    print(
        f"MLP > Raw: {'Yes' if result['val_accuracy'] > result['raw_features_val_accuracy'] else 'No'}"
    )
    print()

    val_acc = result["val_accuracy"]
    mlp_better = result["val_accuracy"] > result["raw_features_val_accuracy"]

    if val_acc > 0.75 and mlp_better:
        print("✓ PASS: Strategy recovery >75% and MLP outperforms raw features")
        sys.exit(0)
    else:
        if val_acc <= 0.75:
            print(f"✗ FAIL: Strategy recovery {val_acc:.2%} ≤ 75%")
        if not mlp_better:
            print("✗ FAIL: MLP does not outperform raw features baseline")
        sys.exit(1)
