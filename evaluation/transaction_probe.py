"""
Transaction encoder probe — strategy recovery + price sensitivity correlation.

Evaluates the transaction encoder via frozen + logistic regression probe.
Additionally computes Pearson r between mean(price_paid_normalised) per
participant and psychographic.price_consciousness.

Acceptance (szm.11): strategy recovery >60%; Pearson r >0.7.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch

from evaluation.probe import pearson_r, probe_logistic_regression
from schemas import PERSONA_TO_IDX, CHECKPOINT_PATHS

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
TX_MODEL_PATH = CHECKPOINT_PATHS["transaction"]


def load_transaction_data(
    tx_path: Path = DATA_DIR / "transactions.jsonl",
) -> tuple[list[dict], dict[str, str]]:
    """Load transaction records and return (records, participant_to_persona).

    Each record dict has keys: participant_id, persona_id, brand_tier,
    price_paid_normalised, days_before_session, etc.
    """
    records: list[dict] = []
    pid_to_persona: dict[str, str] = {}
    for line in tx_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        records.append(r)
        pid = r.get("participant_id", r.get("persona_id", ""))
        pid_to_persona[pid] = r.get("persona_id", "unknown")
    return records, pid_to_persona


def split_participants(
    records: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split participants into train/val sets."""
    participant_ids = sorted(
        set(r.get("participant_id", r.get("persona_id", "")) for r in records)
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)
    split_idx = int(train_ratio * len(participant_ids))
    return set(participant_ids[:split_idx]), set(participant_ids[split_idx:])


def generate_transaction_embeddings(
    encoder,
    records: list[dict],
    pid_to_persona: dict[str, str],
    relevant_pids: set[str],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate per-participant embeddings for the given participant IDs.

    Returns (embeddings, labels, participant_ids).
    """
    from encoders.transaction.features import (
        sort_transactions_most_recent_first,
    )
    from schemas.transaction import TransactionRecord

    vocab = encoder.vocab

    # Group records by participant, sort most-recent-first, convert to TransactionRecord
    by_participant: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pid = r.get("participant_id", r.get("persona_id", ""))
        if pid in relevant_pids:
            by_participant[pid].append(r)

    participant_ids = sorted(by_participant.keys())
    embeddings_list: list[np.ndarray] = []
    labels_list: list[int] = []

    encoder.eval()
    with torch.no_grad():
        for pid in participant_ids:
            recs = by_participant[pid]
            # Convert to TransactionRecord dataclass instances
            tx_records = [
                TransactionRecord(
                    participant_id=r.get("participant_id", r.get("persona_id", "")),
                    persona_id=r.get("persona_id", "unknown"),
                    transaction_id=r.get("transaction_id", ""),
                    days_before_session=r.get("days_before_session", 0),
                    category=r.get("category", ""),
                    product_id=r.get("product_id", ""),
                    brand_tier=r.get("brand_tier", "value"),
                    price_paid_normalised=r.get("price_paid_normalised", 0.0),
                    quantity=r.get("quantity", 1),
                    channel=r.get("channel", "online"),
                    purchase_type=r.get("purchase_type", "planned"),
                    on_promotion=r.get("on_promotion", False),
                    loyalty_card=r.get("loyalty_card"),
                )
                for r in recs
            ]
            # Sort most-recent-first
            sorted_records = sort_transactions_most_recent_first(tx_records)
            # Tokenise
            token_seq = vocab.encode_sequence(sorted_records)
            token_seq_b = token_seq.unsqueeze(0).to(device)  # (1, T, 20)
            lengths = torch.tensor([len(sorted_records)], device=device)
            emb = encoder(token_seq_b, lengths)
            embeddings_list.append(emb.cpu().numpy().squeeze(0))

            persona = pid_to_persona.get(pid, recs[0].get("persona_id", "unknown"))
            labels_list.append(PERSONA_TO_IDX.get(persona, 0))

    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list)
    return embeddings, labels, participant_ids


def compute_price_consciousness_correlation(
    participant_ids: list[str],
    records: list[dict],
) -> float:
    """Compute Pearson r between mean(price_paid_normalised) and price_consciousness."""
    from schemas.psychographic import PsychographicVector

    # Load psychographic data
    psycho_path = DATA_DIR / "psychographics.jsonl"
    psycho_records: dict[str, float] = {}
    for line in psycho_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        p = PsychographicVector(**json.loads(line))
        psycho_records[p.participant_id] = p.price_consciousness

    # Compute mean price_paid_normalised per participant
    by_participant: dict[str, list[float]] = defaultdict(list)
    for r in records:
        pid = r.get("participant_id", r.get("persona_id", ""))
        if "price_paid_normalised" in r and pid in psycho_records:
            by_participant[pid].append(r["price_paid_normalised"])

    mean_prices: list[float] = []
    price_consciousness_vals: list[float] = []
    for pid in participant_ids:
        if pid in by_participant and pid in psycho_records:
            mean_prices.append(np.mean(by_participant[pid]))
            price_consciousness_vals.append(psycho_records[pid])

    if len(mean_prices) < 10:
        return 0.0

    return pearson_r(np.array(mean_prices), np.array(price_consciousness_vals))


def probe(
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run the transaction encoder probe evaluation."""
    from encoders.transaction.model import TransactionEncoder

    # Load data
    records, pid_to_persona = load_transaction_data()
    train_pids, val_pids = split_participants(
        records, train_ratio=train_ratio, seed=seed
    )
    logger.info(
        "Train participants: %d, Val participants: %d", len(train_pids), len(val_pids)
    )

    # Build or load encoder
    if TX_MODEL_PATH.exists():
        logger.info("Loading transaction encoder from %s", TX_MODEL_PATH)
        encoder = TransactionEncoder().to(device)
        state = torch.load(TX_MODEL_PATH, map_location=device, weights_only=True)
        encoder.load_state_dict(state, strict=False)
    else:
        logger.info("No checkpoint found — training transaction encoder from scratch")
        from encoders.transaction.train import train as train_tx

        # Convert records to TransactionRecord dataclass instances
        from schemas.transaction import TransactionRecord

        tx_records = [
            TransactionRecord(
                participant_id=r.get("participant_id", r.get("persona_id", "")),
                persona_id=r.get("persona_id", "unknown"),
                transaction_id=r.get("transaction_id", ""),
                days_before_session=r.get("days_before_session", 0),
                category=r.get("category", ""),
                product_id=r.get("product_id", ""),
                brand_tier=r.get("brand_tier", "value"),
                price_paid_normalised=r.get("price_paid_normalised", 0.0),
                quantity=r.get("quantity", 1),
                channel=r.get("channel", "online"),
                purchase_type=r.get("purchase_type", "planned"),
                on_promotion=r.get("on_promotion", False),
                loyalty_card=r.get("loyalty_card"),
            )
            for r in records
        ]
        encoder = train_tx(records=tx_records, device=device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Generate embeddings
    train_embs, train_labels, train_pids_list = generate_transaction_embeddings(
        encoder, records, pid_to_persona, train_pids, device
    )
    val_embs, val_labels, val_pids_list = generate_transaction_embeddings(
        encoder, records, pid_to_persona, val_pids, device
    )

    # Combine for probe
    all_embs = np.concatenate([train_embs, val_embs])
    all_labels = np.concatenate([train_labels, val_labels])
    train_idx = np.arange(len(train_embs))
    val_idx = np.arange(len(train_embs), len(all_embs))

    result = probe_logistic_regression(all_embs, all_labels, train_idx, val_idx)

    # Price sensitivity correlation
    r = compute_price_consciousness_correlation(
        train_pids_list + val_pids_list, records
    )

    # Log to MLflow
    with mlflow.start_run(run_name="transaction_probe_v1"):
        mlflow.set_tag("modality", "transaction")
        mlflow.set_tag("task", "probe")
        mlflow.log_metric("strategy_recovery_acc", result["val_accuracy"])
        mlflow.log_metric("pearson_r_price_consciousness", r)

    result["pearson_r"] = r
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    result = probe(device=device)

    print("\n" + "=" * 60)
    print("Transaction Encoder Probe Results")
    print("=" * 60)
    print(f"Strategy Recovery Accuracy: {result['val_accuracy']:.2%}")
    print(
        f"Pearson r (price paid vs price consciousness): {result.get('pearson_r', 0):.4f}"
    )
    print()

    val_acc = result["val_accuracy"]
    pearson = result.get("pearson_r", 0)

    if val_acc > 0.60 and pearson > 0.7:
        print("✓ PASS: Strategy recovery >60% and Pearson r >0.7")
        sys.exit(0)
    else:
        if val_acc <= 0.60:
            print(f"✗ FAIL: Strategy recovery {val_acc:.2%} ≤ 60%")
        if pearson <= 0.7:
            print(f"✗ FAIL: Pearson r {pearson:.4f} ≤ 0.7")
        sys.exit(1)
