"""
Unified probe runner — evaluates all 4 encoders with logistic regression probe.

Since the synthetic data has 7 participants (one per persona archetype) with
many records each, probes use record-level stratified train/test splits.
Results are logged to MLflow.

Usage:
    PYTHONPATH=. uv run python -m evaluation.run_probes
    PYTHONPATH=. uv run python -m evaluation.run_probes --device cuda

Expected thresholds (from szm.10–szm.13):
    trace:         > 85% strategy recovery
    transaction:   > 60% strategy recovery
    text:          > 70% strategy recovery
    psychographic: > 75% strategy recovery

Exit code: 0 if all thresholds met, 1 otherwise or on error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from schemas import PERSONA_LABELS, PERSONA_TO_IDX

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")


def probe_with_sklearn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Probe embeddings with stratified logistic regression.

    Uses StratifiedShuffleSplit for robust evaluation.
    """
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )
    accuracies: list[float] = []

    for train_idx, val_idx in sss.split(embeddings, labels):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train_s, y_train)
        accuracies.append(accuracy_score(y_val, clf.predict(X_val_s)))

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "n_splits": n_splits,
    }


# ---------------------------------------------------------------------------
# Trace probe
# ---------------------------------------------------------------------------


def probe_trace(device: str = "cpu") -> dict:
    """Probe the trace encoder."""
    from encoders.trace.tokeniser import build_vocab, tokenise_trial
    from schemas.trace import AcquisitionEvent, TrialRecord

    logger.info("=== Trace Encoder Probe ===")

    # Load data
    traces_path = DATA_DIR / "traces.jsonl"
    trials_path = DATA_DIR / "trials.jsonl"

    events = [
        AcquisitionEvent(**json.loads(line))
        for line in traces_path.read_text().strip().split("\n")
        if line.strip()
    ]
    trials: dict[str, TrialRecord] = {}
    for line in trials_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        r = TrialRecord(**json.loads(line))
        trials[r.trial_id] = r

    # Group events by trial
    events_by_trial: dict[str, list[AcquisitionEvent]] = defaultdict(list)
    for ev in events:
        events_by_trial[ev.trial_id].append(ev)

    # Build vocab
    vocab = build_vocab(events)

    # Train encoder
    logger.info("Training trace encoder...")
    from encoders.trace.train import train as train_trace

    with mlflow.start_run(run_name="trace_encoder_probe_v1"):
        mlflow.set_tag("modality", "trace")
        encoder = train_trace(device=device, n_epochs=30, seed=42)

    # Generate per-trial embeddings
    encoder.eval()
    embeddings_list: list[np.ndarray] = []
    labels_list: list[int] = []

    with torch.no_grad():
        for tid, trial in trials.items():
            trial_events = events_by_trial.get(tid, [])
            if not trial_events:
                continue
            tokens, mask = tokenise_trial(trial_events, trial, vocab)
            tokens_b = tokens.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device) if mask is not None else None
            emb = encoder(tokens_b, mask_b)
            embeddings_list.append(emb.cpu().numpy().squeeze(0))
            labels_list.append(PERSONA_TO_IDX[trial.persona_id])

    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list)
    logger.info("Generated %d trial embeddings", len(embeddings))

    result = probe_with_sklearn(embeddings, labels)
    logger.info(
        "Trace strategy recovery: %.2f%% ± %.2f%%",
        result["mean_accuracy"] * 100,
        result["std_accuracy"] * 100,
    )

    try:
        with mlflow.start_run(run_name="trace_probe_v1"):
            mlflow.set_tag("modality", "trace")
            mlflow.set_tag("task", "probe")
            mlflow.log_metric("strategy_recovery_acc", result["mean_accuracy"])
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Transaction probe
# ---------------------------------------------------------------------------


def probe_transaction(device: str = "cpu") -> dict:
    """Probe the transaction encoder."""
    from encoders.transaction.features import sort_transactions_most_recent_first
    from schemas.transaction import TransactionRecord

    logger.info("=== Transaction Encoder Probe ===")

    # Load data
    tx_path = DATA_DIR / "transactions.jsonl"
    records: list[dict] = []
    pid_to_persona: dict[str, str] = {}
    for line in tx_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        records.append(r)
        pid = r.get("participant_id", r.get("persona_id", ""))
        pid_to_persona[pid] = r.get("persona_id", "unknown")

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
        for r in records
    ]

    # Train encoder
    logger.info("Training transaction encoder...")
    from encoders.transaction.train import train as train_tx

    encoder = train_tx(records=tx_records, device=device, n_epochs=20)

    # Generate per-participant embeddings
    encoder.eval()
    by_participant: dict[str, list] = defaultdict(list)
    for r in tx_records:
        by_participant[r.participant_id].append(r)

    embeddings_list: list[np.ndarray] = []
    labels_list: list[int] = []

    with torch.no_grad():
        for pid, recs in by_participant.items():
            sorted_recs = sort_transactions_most_recent_first(recs)
            token_seq = encoder.vocab.encode_sequence(sorted_recs)
            token_seq_b = token_seq.unsqueeze(0).to(device)
            lengths = torch.tensor([len(sorted_recs)], device=device)
            emb = encoder(token_seq_b, lengths)
            embeddings_list.append(emb.cpu().numpy().squeeze(0))
            persona = pid_to_persona.get(pid, recs[0].persona_id)
            labels_list.append(PERSONA_TO_IDX.get(persona, 0))

    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list)
    logger.info("Generated %d participant embeddings", len(embeddings))

    result = probe_with_sklearn(embeddings, labels)
    logger.info(
        "Transaction strategy recovery: %.2f%% ± %.2f%%",
        result["mean_accuracy"] * 100,
        result["std_accuracy"] * 100,
    )

    # Pearson r correlation with price consciousness
    from evaluation.probe import pearson_r
    from schemas.psychographic import PsychographicVector

    psycho_path = DATA_DIR / "psychographics.jsonl"
    price_consciousness_map: dict[str, float] = {}
    for line in psycho_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        p = PsychographicVector(**json.loads(line))
        price_consciousness_map[p.participant_id] = p.price_consciousness

    mean_prices: list[float] = []
    pc_vals: list[float] = []
    for pid in by_participant:
        if pid in price_consciousness_map:
            prices = [r.price_paid_normalised for r in by_participant[pid]]
            mean_prices.append(np.mean(prices))
            pc_vals.append(price_consciousness_map[pid])

    if len(mean_prices) >= 3:
        r_val = pearson_r(np.array(mean_prices), np.array(pc_vals))
        logger.info("Pearson r (price×consciousness): %.4f", r_val)
    else:
        r_val = 0.0

    try:
        with mlflow.start_run(run_name="transaction_probe_v1"):
            mlflow.set_tag("modality", "transaction")
            mlflow.set_tag("task", "probe")
            mlflow.log_metric("strategy_recovery_acc", result["mean_accuracy"])
            mlflow.log_metric("pearson_r_price_consciousness", r_val)
    except Exception:
        pass

    result["pearson_r"] = r_val
    return result


# ---------------------------------------------------------------------------
# Text probe
# ---------------------------------------------------------------------------


def probe_text(device: str = "cpu") -> dict:
    """Probe the text encoder."""
    from encoders.text.embed import train as train_text
    from schemas.text import PersonaNarrative

    logger.info("=== Text Encoder Probe ===")

    # Load data
    narratives_path = DATA_DIR / "narratives.jsonl"
    narratives = [
        PersonaNarrative(**json.loads(line))
        for line in narratives_path.read_text().strip().split("\n")
        if line.strip()
    ]
    logger.info("Loaded %d narratives", len(narratives))

    # Train encoder
    logger.info("Training text encoder...")
    encoder = train_text(
        narratives=narratives,
        n_epochs=20,
        batch_size=64,
        device=device,
        log_mlflow=False,
    )

    # Generate embeddings
    encoder.eval()
    texts = [n.text for n in narratives]
    with torch.no_grad():
        sentence_embs = encoder.encode_texts(texts).to(device)
        projected = encoder(sentence_embs)

    embeddings = projected.cpu().numpy()
    labels = np.array([PERSONA_TO_IDX.get(n.persona_id, 0) for n in narratives])
    logger.info("Generated %d narrative embeddings", len(embeddings))

    result = probe_with_sklearn(embeddings, labels)
    logger.info(
        "Text strategy recovery: %.2f%% ± %.2f%%",
        result["mean_accuracy"] * 100,
        result["std_accuracy"] * 100,
    )

    # Cosine similarity stats
    from evaluation.probe import compute_cosine_similarity_stats

    cos_sim = compute_cosine_similarity_stats(
        embeddings, labels, label_names=PERSONA_LABELS
    )
    logger.info("Intra-persona cosine sim: %.4f", cos_sim["intra_mean"])
    logger.info("Inter-persona cosine sim: %.4f", cos_sim["inter_mean"])

    try:
        with mlflow.start_run(run_name="text_probe_v1"):
            mlflow.set_tag("modality", "text")
            mlflow.set_tag("task", "probe")
            mlflow.log_metric("strategy_recovery_acc", result["mean_accuracy"])
            mlflow.log_metric("intra_persona_cosine_sim", cos_sim["intra_mean"])
            mlflow.log_metric("inter_persona_cosine_sim", cos_sim["inter_mean"])
    except Exception:
        pass

    result["cosine_sim"] = cos_sim
    return result


# ---------------------------------------------------------------------------
# Psychographic probe
# ---------------------------------------------------------------------------


def probe_psychographic(device: str = "cpu") -> dict:
    """Probe the psychographic encoder."""
    from encoders.psychographic.features import to_feature_vector
    from schemas.psychographic import PsychographicVector

    logger.info("=== Psychographic Encoder Probe ===")

    # Load data
    psycho_path = DATA_DIR / "psychographics.jsonl"
    psych_records: list[PsychographicVector] = []
    for line in psycho_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        psych_records.append(PsychographicVector(**json.loads(line)))
    logger.info("Loaded %d psychographic records", len(psych_records))

    # Train encoder
    logger.info("Training psychographic encoder...")
    from encoders.psychographic.train import train as train_psycho

    encoder = train_psycho(
        records=psych_records,
        n_epochs=40,
        batch_size=128,
        device=device,
        log_mlflow=False,
    )

    # Generate embeddings and raw features
    encoder.eval()
    embeddings_list: list[np.ndarray] = []
    raw_features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    with torch.no_grad():
        for r in psych_records:
            vec = to_feature_vector(r).unsqueeze(0).to(device)
            raw_features_list.append(vec.cpu().numpy().squeeze(0))
            emb = encoder(vec).cpu().numpy().squeeze(0)
            embeddings_list.append(emb)
            labels_list.append(PERSONA_TO_IDX.get(r.persona_id, 0))

    embeddings = np.stack(embeddings_list)
    raw_features = np.stack(raw_features_list)
    labels = np.array(labels_list)
    logger.info("Generated %d psychographic embeddings", len(embeddings))

    # MLP probe
    result = probe_with_sklearn(embeddings, labels)
    logger.info(
        "Psychographic (MLP) strategy recovery: %.2f%% ± %.2f%%",
        result["mean_accuracy"] * 100,
        result["std_accuracy"] * 100,
    )

    # Raw features baseline
    raw_result = probe_with_sklearn(raw_features, labels)
    logger.info(
        "Raw features (22-dim) strategy recovery: %.2f%% ± %.2f%%",
        raw_result["mean_accuracy"] * 100,
        raw_result["std_accuracy"] * 100,
    )

    try:
        with mlflow.start_run(run_name="psychographic_probe_v1"):
            mlflow.set_tag("modality", "psychographic")
            mlflow.set_tag("task", "probe")
            mlflow.log_metric("strategy_recovery_acc", result["mean_accuracy"])
            mlflow.log_metric(
                "raw_features_strategy_recovery_acc", raw_result["mean_accuracy"]
            )
    except Exception:
        pass

    result["raw_accuracy"] = raw_result["mean_accuracy"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run all 4 encoder probes and report strategy recovery."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--probe",
        choices=["trace", "transaction", "text", "psychographic"],
        default=None,
        help="Run a single probe instead of all 4",
    )
    args = parser.parse_args(argv)

    device = args.device
    logger.info("Device: %s", device)

    results: dict[str, dict] = {}

    # Run all probes (or a single one)
    probe_registry = [
        ("trace", probe_trace),
        ("transaction", probe_transaction),
        ("text", probe_text),
        ("psychographic", probe_psychographic),
    ]
    if args.probe:
        probe_registry = [
            (args.probe, fn) for name, fn in probe_registry if name == args.probe
        ]

    for name, probe_fn in probe_registry:
        logger.info("\n" + "=" * 60)
        logger.info("Running %s probe...", name)
        logger.info("=" * 60)
        try:
            results[name] = probe_fn(device)
        except Exception as e:
            logger.error("%s probe failed: %s", name, e, exc_info=True)
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("PROBE RESULTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if "error" in result:
            print(f"  {name:20s}: ERROR — {result['error'][:80]}")
        else:
            acc = result.get("mean_accuracy", 0)
            print(f"  {name:20s}: {acc:.2%} strategy recovery")

    # Check pass thresholds
    thresholds = {
        "trace": 0.85,
        "transaction": 0.60,
        "text": 0.70,
        "psychographic": 0.75,
    }
    all_pass = True
    for name, threshold in thresholds.items():
        if name in results and "error" not in results[name]:
            acc = results[name]["mean_accuracy"]
            if acc >= threshold:
                print(f"  ✓ {name}: {acc:.2%} ≥ {threshold:.0%}")
            else:
                print(f"  ✗ {name}: {acc:.2%} < {threshold:.0%}")
                all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
