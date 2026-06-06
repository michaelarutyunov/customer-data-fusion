"""
Trace encoder probe — strategy recovery evaluation.

Loads (or trains) the trace encoder, freezes it, generates per-participant
embeddings via mean-pooling over trials, and evaluates via logistic regression
probe.

Acceptance (szm.10): strategy recovery >85%; all archetypes except adaptive >80%.
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

from evaluation.probe import probe_logistic_regression, mean_pool_per_participant
from schemas import PERSONA_LABELS, PERSONA_TO_IDX
from schemas.trace import AcquisitionEvent, TrialRecord

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
MODEL_DIR = Path("models")
TRACE_MODEL_PATH = MODEL_DIR / "trace_encoder.pt"


def load_and_split_data(
    traces_path: Path = DATA_DIR / "traces.jsonl",
    trials_path: Path = DATA_DIR / "trials.jsonl",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[
    dict[str, list[AcquisitionEvent]],
    dict[str, TrialRecord],
    dict[str, TrialRecord],
]:
    """Load trace events and trials, split train/val by participant_id."""
    # Load events
    events = [
        AcquisitionEvent(**json.loads(line))
        for line in traces_path.read_text().strip().split("\n")
        if line.strip()
    ]
    # Load trials
    trials: dict[str, TrialRecord] = {}
    for line in trials_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        r = TrialRecord(**json.loads(line))
        trials[r.trial_id] = r

    # Split by participant
    participant_ids = sorted(set(r.participant_id for r in trials.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)
    split_idx = int(train_ratio * len(participant_ids))
    train_pids = set(participant_ids[:split_idx])
    val_pids = set(participant_ids[split_idx:])

    train_trials = {
        tid: r for tid, r in trials.items() if r.participant_id in train_pids
    }
    val_trials = {tid: r for tid, r in trials.items() if r.participant_id in val_pids}

    # Group events by trial_id
    events_by_trial: dict[str, list[AcquisitionEvent]] = defaultdict(list)
    for ev in events:
        events_by_trial[ev.trial_id].append(ev)

    return events_by_trial, train_trials, val_trials


def generate_trace_embeddings(
    encoder,
    events_by_trial: dict[str, list[AcquisitionEvent]],
    trials: dict[str, TrialRecord],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Generate embeddings for all trials in the given trial set.

    Returns (embeddings, labels, persona_ids, participant_ids).
    """
    from encoders.trace.tokeniser import build_vocab, tokenise_trial

    # Build vocab from all events
    all_events = [ev for evs in events_by_trial.values() for ev in evs]
    vocab = build_vocab(all_events)

    embeddings_list: list[np.ndarray] = []
    labels_list: list[int] = []
    persona_ids_list: list[str] = []
    participant_ids_list: list[str] = []

    encoder.eval()
    with torch.no_grad():
        for tid, trial in trials.items():
            trial_events = events_by_trial.get(tid, [])
            if not trial_events:
                continue
            tokens, mask = tokenise_trial(trial_events, trial, vocab)
            # Add batch dimension
            tokens_b = tokens.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device) if mask is not None else None
            emb = encoder(tokens_b, mask_b)
            embeddings_list.append(emb.cpu().numpy().squeeze(0))
            labels_list.append(PERSONA_TO_IDX[trial.persona_id])
            persona_ids_list.append(trial.persona_id)
            participant_ids_list.append(trial.participant_id)

    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list)
    return embeddings, labels, persona_ids_list, participant_ids_list


def probe(
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run the trace encoder probe evaluation.

    Returns a dict with probe metrics.
    """
    from encoders.trace.model import TraceEncoder

    # Load data
    events_by_trial, train_trials, val_trials = load_and_split_data(
        train_ratio=train_ratio, seed=seed
    )

    # Build encoder (determine vocab sizes from data)
    # We need n_attributes and n_alternatives — peek at tokeniser
    from encoders.trace.tokeniser import build_vocab

    all_events = [ev for evs in events_by_trial.values() for ev in evs]
    vocab = build_vocab(all_events)
    n_attributes = len(vocab.get("attribute", {})) + 1
    n_alternatives = len(vocab.get("alternative", {})) + 1

    # Train encoder if no checkpoint exists
    if TRACE_MODEL_PATH.exists():
        logger.info("Loading trace encoder from %s", TRACE_MODEL_PATH)
        encoder = TraceEncoder(
            n_attributes=n_attributes,
            n_alternatives=n_alternatives,
            n_classes=len(PERSONA_LABELS),
        ).to(device)
        state = torch.load(TRACE_MODEL_PATH, map_location=device, weights_only=True)
        # Load backbone only (classifier might be missing from saved state)
        encoder.load_state_dict(state, strict=False)
    else:
        logger.info("No checkpoint found — training trace encoder from scratch")
        from encoders.trace.train import train as train_trace

        encoder = train_trace(
            device=device,
            n_epochs=50,  # full training per SPEC
            seed=seed,
        )

    encoder.eval()
    # Freeze
    for p in encoder.parameters():
        p.requires_grad = False

    # Generate embeddings for train and val
    train_embs, train_labels, _, train_pids = generate_trace_embeddings(
        encoder, events_by_trial, train_trials, device
    )
    val_embs, val_labels, _, val_pids = generate_trace_embeddings(
        encoder, events_by_trial, val_trials, device
    )

    # Mean-pool per participant
    train_embs_pooled, train_labels_pooled, _ = mean_pool_per_participant(
        train_embs, train_pids, train_labels
    )
    val_embs_pooled, val_labels_pooled, _ = mean_pool_per_participant(
        val_embs, val_pids, val_labels
    )

    # Combine for probe
    all_embs = np.concatenate([train_embs_pooled, val_embs_pooled])
    all_labels = np.concatenate([train_labels_pooled, val_labels_pooled])
    train_idx = np.arange(len(train_embs_pooled))
    val_idx = np.arange(len(train_embs_pooled), len(all_embs))

    result = probe_logistic_regression(all_embs, all_labels, train_idx, val_idx)

    # Log to MLflow
    with mlflow.start_run(run_name="trace_probe_v1"):
        mlflow.set_tag("modality", "trace")
        mlflow.set_tag("task", "probe")
        mlflow.log_metric("strategy_recovery_acc", result["val_accuracy"])
        for cls_name, acc in result["per_class_accuracy"].items():
            cls_idx = int(cls_name)
            mlflow.log_metric(f"per_class_acc_{PERSONA_LABELS[cls_idx]}", acc)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    result = probe(device=device)

    print("\n" + "=" * 60)
    print("Trace Encoder Probe Results")
    print("=" * 60)
    print(f"Strategy Recovery Accuracy: {result['val_accuracy']:.2%}")
    print(f"  Train accuracy: {result['train_accuracy']:.2%}")
    print()
    print("Per-class accuracy:")
    for cls_name, acc in sorted(result["per_class_accuracy"].items()):
        cls_idx = int(cls_name)
        label = PERSONA_LABELS[cls_idx]
        passed = "✓" if acc > 0.80 or label == "adaptive" else "✗"
        print(f"  {label:20s}: {acc:.2%} {passed}")
    print()
    print("Confusion matrix (val):")
    print(result["confusion_matrix"])
    print()

    # Check pass threshold
    val_acc = result["val_accuracy"]
    all_above_80 = all(
        acc > 0.80
        for cls_name, acc in result["per_class_accuracy"].items()
        if PERSONA_LABELS[int(cls_name)] != "adaptive"
    )

    if val_acc > 0.85 and all_above_80:
        print("✓ PASS: Strategy recovery >85% and all non-adaptive archetypes >80%")
        sys.exit(0)
    else:
        if val_acc <= 0.85:
            print(f"✗ FAIL: Strategy recovery {val_acc:.2%} ≤ 85%")
        if not all_above_80:
            print("✗ FAIL: Some non-adaptive archetypes ≤ 80%")
        sys.exit(1)
