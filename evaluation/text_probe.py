"""
Text encoder probe — strategy recovery + intra/cross-persona cosine similarity.

Evaluates the text encoder via frozen + logistic regression probe.
Additionally computes intra-persona and cross-persona cosine similarity stats.

Acceptance (szm.12): strategy recovery >70%; intra-persona cosine sim >0.6;
                      cross-persona cosine sim <0.4.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

from evaluation.probe import (
    compute_cosine_similarity_stats,
    probe_logistic_regression,
)
from schemas import PERSONA_LABELS, PERSONA_TO_IDX, CHECKPOINT_PATHS

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
TEXT_MODEL_PATH = CHECKPOINT_PATHS["text"]


def load_narrative_data(
    path: Path = DATA_DIR / "narratives.jsonl",
) -> tuple[list[dict], dict[str, str]]:
    """Load narrative records and return (records, participant_to_persona)."""
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


def generate_text_embeddings(
    encoder,
    records: list[dict],
    relevant_pids: set[str],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Generate embeddings for narratives.

    Returns (embeddings, labels, persona_ids, participant_ids).
    """
    from schemas.text import PersonaNarrative

    # Filter to relevant participants and convert to PersonaNarrative
    filtered: list[PersonaNarrative] = []
    for r in records:
        if r["participant_id"] in relevant_pids:
            filtered.append(
                PersonaNarrative(
                    participant_id=r["participant_id"],
                    persona_id=r.get("persona_id", "unknown"),
                    category=r.get("category", "electronics"),
                    text=r["text"],
                    word_count=r.get("word_count", 0),
                    model_id=r.get("model_id", "unknown"),
                    prompt_version=r.get("prompt_version", "v1"),
                    embedding=r.get("embedding"),
                    embedding_model_id=r.get("embedding_model_id"),
                )
            )

    texts = [n.text for n in filtered]
    with torch.no_grad():
        sentence_embs = encoder.encode_texts(texts).to(device)
        projected = encoder(sentence_embs)

    embeddings = projected.cpu().numpy()
    labels = np.array([PERSONA_TO_IDX.get(n.persona_id, 0) for n in filtered])
    persona_ids = [n.persona_id for n in filtered]
    participant_ids = [n.participant_id for n in filtered]

    return embeddings, labels, persona_ids, participant_ids


def probe(
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run the text encoder probe evaluation."""
    from encoders.text.embed import TextEncoder, train as train_text

    records, pid_to_persona = load_narrative_data()
    train_pids, val_pids = split_participants(
        records, train_ratio=train_ratio, seed=seed
    )
    logger.info(
        "Train participants: %d, Val participants: %d", len(train_pids), len(val_pids)
    )

    # Build or load encoder
    if TEXT_MODEL_PATH.exists():
        logger.info("Loading text encoder from %s", TEXT_MODEL_PATH)
        encoder = TextEncoder().to(device)
        state = torch.load(TEXT_MODEL_PATH, map_location=device, weights_only=True)
        encoder.load_state_dict(state, strict=False)
    else:
        logger.info("No checkpoint found — training text encoder from scratch")
        from schemas.text import PersonaNarrative

        narratives = [
            PersonaNarrative(
                participant_id=r["participant_id"],
                persona_id=r.get("persona_id", "unknown"),
                category=r.get("category", "electronics"),
                text=r["text"],
                word_count=r.get("word_count", 0),
                model_id=r.get("model_id", "unknown"),
                prompt_version=r.get("prompt_version", "v1"),
                embedding=r.get("embedding"),
                embedding_model_id=r.get("embedding_model_id"),
            )
            for r in records
        ]
        encoder = train_text(
            narratives=narratives,
            n_epochs=20,
            batch_size=64,
            device=device,
            log_mlflow=False,
        )

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Generate embeddings for all data
    all_embs, all_labels, all_persona_ids, all_pids = generate_text_embeddings(
        encoder, records, train_pids | val_pids, device
    )

    # Build train/val index masks
    train_mask = np.array([pid in train_pids for pid in all_pids])
    val_mask = np.array([pid in val_pids for pid in all_pids])
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # Logistic regression probe
    result = probe_logistic_regression(all_embs, all_labels, train_idx, val_idx)

    # Cosine similarity stats
    cos_sim_stats = compute_cosine_similarity_stats(
        all_embs,
        np.array([PERSONA_TO_IDX.get(p, 0) for p in all_persona_ids]),
        label_names=PERSONA_LABELS,
    )

    # Log to MLflow
    with mlflow.start_run(run_name="text_probe_v1"):
        mlflow.set_tag("modality", "text")
        mlflow.set_tag("task", "probe")
        mlflow.log_metric("strategy_recovery_acc", result["val_accuracy"])
        mlflow.log_metric("intra_persona_cosine_sim", cos_sim_stats["intra_mean"])
        mlflow.log_metric("inter_persona_cosine_sim", cos_sim_stats["inter_mean"])

    result["cosine_sim_stats"] = cos_sim_stats
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    result = probe(device=device)

    cos = result["cosine_sim_stats"]
    print("\n" + "=" * 60)
    print("Text Encoder Probe Results")
    print("=" * 60)
    print(f"Strategy Recovery Accuracy: {result['val_accuracy']:.2%}")
    print(f"Intra-persona cosine similarity: {cos['intra_mean']:.4f} (target >0.6)")
    print(f"Inter-persona cosine similarity: {cos['inter_mean']:.4f} (target <0.4)")
    print()
    print("Per-class intra-persona cosine sim:")
    for cls_name, sim in cos["intra_per_class"].items():
        print(f"  {cls_name:20s}: {sim:.4f}")
    print()
    print("Pairwise cosine similarity matrix:")
    print("         " + " ".join(f"{n[:4]:>6s}" for n in cos["class_names"]))
    for i, row_name in enumerate(cos["class_names"]):
        row_str = " ".join(
            f"{cos['pairwise_matrix'][i, j]:6.3f}"
            for j in range(len(cos["class_names"]))
        )
        print(f"  {row_name[:4]:>6s} {row_str}")
    print()

    val_acc = result["val_accuracy"]
    intra = cos["intra_mean"]
    inter = cos["inter_mean"]

    passed = True
    if val_acc <= 0.70:
        print(f"✗ FAIL: Strategy recovery {val_acc:.2%} ≤ 70%")
        passed = False
    else:
        print(f"✓ PASS: Strategy recovery {val_acc:.2%} > 70%")

    if intra <= 0.6:
        print(f"✗ FAIL: Intra-persona cosine sim {intra:.4f} ≤ 0.6")
        passed = False
    else:
        print(f"✓ PASS: Intra-persona cosine sim {intra:.4f} > 0.6")

    if inter >= 0.4:
        print(f"✗ FAIL: Inter-persona cosine sim {inter:.4f} ≥ 0.4")
        passed = False
    else:
        print(f"✓ PASS: Inter-persona cosine sim {inter:.4f} < 0.4")

    # Check for overly uniform generation
    high_sim_pairs = []
    for i in range(len(cos["class_names"])):
        for j in range(i + 1, len(cos["class_names"])):
            if cos["pairwise_matrix"][i, j] > 0.8:
                high_sim_pairs.append((cos["class_names"][i], cos["class_names"][j]))
    if high_sim_pairs:
        print(
            "⚠ WARNING: Pairs with cosine sim >0.8 (LLM generation may be too uniform):"
        )
        for a, b in high_sim_pairs:
            print(
                f"  {a} ↔ {b}: {cos['pairwise_matrix'][cos['class_names'].index(a), cos['class_names'].index(b)]:.3f}"
            )

    sys.exit(0 if passed else 1)
