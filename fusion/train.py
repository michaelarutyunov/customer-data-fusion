"""
fusion/train.py

Training script for the late fusion meta-learner.

Loads all four frozen modality encoders, generates cached embeddings,
and trains the LateFusionMetaLearner on persona classification.

Usage:
    uv run python -m fusion.train
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import mlflow

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM, PERSONA_TO_IDX
from fusion.meta_learner import LateFusionMetaLearner


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------


def load_encoders(device: str = "cpu") -> dict[str, nn.Module]:
    """Load all four encoder checkpoints (frozen).

    Parameters
    ----------
    device : str
        Target device for models ("cpu" or "cuda").

    Returns
    -------
    dict[str, nn.Module]
        Dictionary mapping modality names to frozen encoder models.
    """
    from encoders.trace.model import TraceEncoder
    from encoders.transaction.model import TransactionEncoder
    from encoders.text.embed import TextEncoder
    from encoders.psychographic.model import PsychographicEncoder

    encoders = {}

    # Trace encoder
    trace_encoder = TraceEncoder(n_classes=7).to(device)
    trace_state = torch.load(
        CHECKPOINT_PATHS["trace"], map_location=device, weights_only=True
    )
    trace_encoder.load_state_dict(trace_state, strict=False)
    for param in trace_encoder.parameters():
        param.requires_grad = False
    encoders["trace"] = trace_encoder

    # Transaction encoder
    tx_encoder = TransactionEncoder().to(device)
    tx_state = torch.load(
        CHECKPOINT_PATHS["transaction"], map_location=device, weights_only=True
    )
    tx_encoder.load_state_dict(tx_state, strict=False)
    for param in tx_encoder.parameters():
        param.requires_grad = False
    encoders["transaction"] = tx_encoder

    # Text encoder
    text_encoder = TextEncoder(n_classes=7).to(device)
    # Text encoder checkpoint contains only trainable parameters
    text_state = torch.load(
        CHECKPOINT_PATHS["text"], map_location=device, weights_only=True
    )
    text_encoder.load_state_dict(text_state, strict=False)
    for param in text_encoder.parameters():
        param.requires_grad = False
    encoders["text"] = text_encoder

    # Psychographic encoder
    psycho_encoder = PsychographicEncoder(n_classes=7).to(device)
    psycho_state = torch.load(
        CHECKPOINT_PATHS["psychographic"], map_location=device, weights_only=True
    )
    psycho_encoder.load_state_dict(psycho_state)
    for param in psycho_encoder.parameters():
        param.requires_grad = False
    encoders["psychographic"] = psycho_encoder

    return encoders


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_psychographics(
    path: Path = Path("data/synthetic/psychographics.jsonl"),
) -> list[dict]:
    """Load psychographic records to get participant IDs and labels.

    Parameters
    ----------
    path : Path
        Path to psychographics.jsonl.

    Returns
    -------
    list[dict]
        List of psychographic records with participant_id and persona_id.
    """
    records = []
    for line in path.read_text().strip().splitlines():
        record = json.loads(line)
        records.append(record)
    return records


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


def load_modality_data(
    modality: str,
    participant_ids: list[str],
) -> dict[str, dict]:
    """Load modality data and index by participant_id.

    Parameters
    ----------
    modality : str
        Modality name ("traces", "trials", "transactions", "narratives").
    participant_ids : list[str]
        Canonical participant ID ordering.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping participant_id to modality data tensor.
    """
    data_path = Path(f"data/synthetic/{modality}.jsonl")
    participant_to_data = {}

    for line in data_path.read_text().strip().splitlines():
        record = json.loads(line)
        participant_id = record.get("participant_id")
        if participant_id in participant_ids:
            participant_to_data[participant_id] = record

    return participant_to_data


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embeddings(
    encoders: dict[str, nn.Module],
    psychographics: list[dict],
    modality_data: dict[str, dict[str, dict]],
    device: str = "cpu",
) -> dict:
    """Generate embeddings for all participants using frozen encoders."""
    from collections import defaultdict

    from encoders.psychographic.features import to_feature_vector
    from encoders.trace.tokeniser import build_vocab, tokenise_trial
    from encoders.transaction.features import sort_transactions_most_recent_first
    from encoders.transaction.model import TransactionEncoder as TxEncoder
    from encoders.text.embed import TextEncoder as TxtEncoder
    from schemas.psychographic import PsychographicVector
    from schemas.trace import AcquisitionEvent, TrialRecord
    from schemas.transaction import TransactionRecord

    n_participants = len(psychographics)
    embeddings: dict = {
        "trace": torch.zeros(n_participants, EMBEDDING_DIM, device=device),  # type: ignore[reportPrivateImportUsage]
        "transaction": torch.zeros(n_participants, EMBEDDING_DIM, device=device),  # type: ignore[reportPrivateImportUsage]
        "text": torch.zeros(n_participants, EMBEDDING_DIM, device=device),  # type: ignore[reportPrivateImportUsage]
        "psychographic": torch.zeros(n_participants, EMBEDDING_DIM, device=device),  # type: ignore[reportPrivateImportUsage]
        "labels": torch.zeros(n_participants, dtype=torch.long, device=device),  # type: ignore[reportPrivateImportUsage]
        "participant_ids": [],
    }

    # Build pid → index map
    pid_to_idx = {p["participant_id"]: i for i, p in enumerate(psychographics)}

    # ── Load trace events grouped by participant ──────────────────────────────
    events_by_pid: dict[str, list] = defaultdict(list)
    trials_by_pid: dict[str, list] = defaultdict(list)

    traces_path = Path("data/synthetic/traces.jsonl")
    for line in traces_path.read_text().strip().splitlines():
        r = json.loads(line)
        pid = r.get("participant_id", "")
        if pid in pid_to_idx:
            events_by_pid[pid].append(AcquisitionEvent(**r))

    trials_path = Path("data/synthetic/trials.jsonl")
    for line in trials_path.read_text().strip().splitlines():
        r = json.loads(line)
        pid = r.get("participant_id", "")
        if pid in pid_to_idx:
            trials_by_pid[pid].append(TrialRecord(**r))

    # Build vocab from all events
    all_events = [ev for evs in events_by_pid.values() for ev in evs]
    vocab = build_vocab(all_events)

    # ── Load transaction records grouped by participant ────────────────────────
    tx_by_pid: dict[str, list] = defaultdict(list)
    tx_path = Path("data/synthetic/transactions.jsonl")
    for line in tx_path.read_text().strip().splitlines():
        r = json.loads(line)
        pid = r.get("participant_id", "")
        if pid in pid_to_idx:
            tx_by_pid[pid].append(r)

    # ── Encode per participant ─────────────────────────────────────────────────
    for i, psycho in enumerate(psychographics):
        pid = psycho["participant_id"]
        embeddings["participant_ids"].append(pid)
        embeddings["labels"][i] = PERSONA_TO_IDX[psycho["persona_id"]]

        with torch.no_grad():
            # Trace: mean-pool embeddings across all trials for this participant
            trial_embs = []
            for trial in trials_by_pid.get(pid, []):
                trial_events = events_by_pid.get(pid, [])
                # Filter events for this specific trial
                tid_events = [e for e in trial_events if e.trial_id == trial.trial_id]
                if not tid_events:
                    continue
                tokens, mask = tokenise_trial(tid_events, trial, vocab)
                tokens_b = tokens.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device) if mask is not None else None
                emb = encoders["trace"](tokens_b, mask_b)
                trial_embs.append(emb.squeeze(0))
            if trial_embs:
                embeddings["trace"][i] = torch.stack(trial_embs).mean(0)

            # Transaction: encode sequence, get participant embedding
            raw_txs = tx_by_pid.get(pid, [])
            if raw_txs:
                tx_records = [
                    TransactionRecord(
                        participant_id=r.get("participant_id", ""),
                        persona_id=r.get("persona_id", ""),
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
                    for r in raw_txs
                ]
                sorted_tx = sort_transactions_most_recent_first(tx_records)
                tx_enc = encoders["transaction"]
                assert isinstance(tx_enc, TxEncoder)
                token_seq = tx_enc.vocab.encode_sequence(sorted_tx)
                token_seq_b = token_seq.unsqueeze(0).to(device)
                lengths = torch.tensor([len(sorted_tx)], device=device)  # type: ignore[reportPrivateImportUsage]
                embeddings["transaction"][i] = tx_enc(token_seq_b, lengths).squeeze(0)

            # Text: sentence-transformer encode then project
            narrative = modality_data["narratives"].get(pid)
            if narrative:
                text = narrative.get("text", "")
                if text:
                    txt_enc = encoders["text"]
                    assert isinstance(txt_enc, TxtEncoder)
                    sent_emb = txt_enc.encode_texts([text]).to(device)
                    embeddings["text"][i] = txt_enc(sent_emb).squeeze(0)

            # Psychographic: feature vector → MLP
            psycho_vec = PsychographicVector(
                **{
                    k: v
                    for k, v in psycho.items()
                    if k in PsychographicVector.__dataclass_fields__
                }
            )
            raw_vec = to_feature_vector(psycho_vec).to(device)
            embeddings["psychographic"][i] = encoders["psychographic"](
                raw_vec.unsqueeze(0)
            ).squeeze(0)

        if (i + 1) % 100 == 0:
            print(f"  Embedded {i + 1}/{n_participants} participants")

    return embeddings


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def build_cache(
    encoders: dict[str, nn.Module],
    cache_path: Path,
    device: str = "cpu",
) -> dict:
    """Build or load embedding cache.

    Parameters
    ----------
    encoders : dict[str, nn.Module]
        Frozen encoder models.
    cache_path : Path
        Path to cache file.
    device : str
        Target device for computation.

    Returns
    -------
    dict
        Cached embeddings with keys "trace", "transaction", "text",
        "psychographic", "labels", "participant_ids".
    """
    # Check if cache is valid (newer than all encoder checkpoints)
    cache_valid = cache_path.exists()
    if cache_valid:
        cache_mtime = cache_path.stat().st_mtime
        for modality, checkpoint_path in CHECKPOINT_PATHS.items():
            if modality == "fusion":
                continue
            if checkpoint_path.stat().st_mtime > cache_mtime:
                cache_valid = False
                break

    if cache_valid:
        print(f"Loading cached embeddings from {cache_path}")
        # Note: weights_only=False required for cache (dict with tensors + list)
        # Safe because cache is created by our own code in same script
        return torch.load(cache_path)

    # Build cache
    print("Generating embeddings from encoder checkpoints...")

    # Load psychographics to get participant ordering
    psychographics = load_psychographics()
    participant_ids = [p["participant_id"] for p in psychographics]

    # Narratives loaded here; traces and transactions loaded inside generate_embeddings
    modality_data = {
        "narratives": load_modality_data("narratives", participant_ids),
    }

    # Generate embeddings
    embeddings = generate_embeddings(encoders, psychographics, modality_data, device)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    print(f"Saved embeddings cache to {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def nt_xent_fusion(
    emb_v1: torch.Tensor,
    emb_v2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent (SimCLR-style) for a matched pair of fused embedding views.

    emb_v1[i] and emb_v2[i] are two modality-dropout augmented views of the same
    participant i. All other cross-participant pairs are negatives.

    Parameters
    ----------
    emb_v1, emb_v2 : Tensor, shape (B, D)
        L2-normalised CDT embeddings from two dropout-augmented forward passes.
    temperature : float

    Returns
    -------
    Scalar loss tensor.
    """
    B = emb_v1.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb_v1.device, requires_grad=True)

    embs = F.normalize(torch.cat([emb_v1, emb_v2], dim=0), dim=1)  # (2B, D)
    sim = torch.mm(embs, embs.t()) / temperature  # (2B, 2B)

    labels = torch.cat(
        [
            torch.arange(B, 2 * B, device=emb_v1.device),
            torch.arange(0, B, device=emb_v1.device),
        ]
    )
    mask = torch.eye(2 * B, dtype=torch.bool, device=emb_v1.device)
    sim = sim.masked_fill(mask, float("-inf"))
    return F.cross_entropy(sim, labels)


def _apply_modality_dropout(
    batch_embs: torch.Tensor, p_dropout: float, device: str
) -> torch.Tensor:
    """Apply independent per-modality dropout to a [B, M, 128] embedding batch."""
    B, M, _ = batch_embs.shape
    masks = [torch.rand(B, 1, device=device) >= p_dropout for _ in range(M)]
    result = batch_embs.clone()
    for i in range(M):
        result[:, i] = result[:, i] * masks[i].float()
    return result


def train(
    *,
    cache_path: Optional[Path] = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    p_dropout: float = 0.2,
    device: str = "cpu",
    log_mlflow: bool = True,
    phase: str = "2",
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
) -> LateFusionMetaLearner:
    """Train the fusion meta-learner with NT-Xent + CE multi-task objective.

    Two modality-dropout augmented views of each participant's fused embedding
    are used as NT-Xent positive pairs. Other participants in the batch are
    negatives. A CE auxiliary head retains archetype separability (Tier 1 gate).

    Parameters
    ----------
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    p_dropout : float
        Per-modality dropout probability for augmentation views (default 0.2).
    device : str
        Target device ("cpu" or "cuda").
    log_mlflow : bool
        Whether to log to MLflow.
    phase : str
        Meta-learner phase ("1" or "2").
    lambda_contrastive : float
        Weight for NT-Xent loss. Total = CE + lambda * NT-Xent.
    nt_xent_temperature : float
        NT-Xent temperature (default 0.07).

    Returns
    -------
    LateFusionMetaLearner
        Trained meta-learner model.
    """
    if cache_path is None:
        cache_path = Path("models/fusion_embeddings_cache.pt")

    # Load encoders
    print("Loading encoder checkpoints...")
    encoders = load_encoders(device)
    print("Encoders loaded and frozen.")

    # Build or load embedding cache
    embeddings = build_cache(encoders, cache_path, device)

    # Split participants
    participant_ids: list[str] = embeddings["participant_ids"]  # type: ignore[assignment]
    train_ids, val_ids = split_by_participant(participant_ids)

    # Create train/val indices
    participant_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
    train_indices = torch.tensor([participant_to_idx[pid] for pid in train_ids])  # type: ignore[reportPrivateImportUsage]
    val_indices = torch.tensor([participant_to_idx[pid] for pid in val_ids])  # type: ignore[reportPrivateImportUsage]

    # Extract embeddings and labels.
    # Modalities derived from the embedding cache (supports 4 or 6 modalities).
    _MODALITIES = [k for k in embeddings if k != "labels"]
    n_modalities = len(_MODALITIES)
    train_embs = {mod: embeddings[mod][train_indices] for mod in _MODALITIES}
    train_labels = embeddings["labels"][train_indices]

    val_embs = {mod: embeddings[mod][val_indices] for mod in _MODALITIES}
    val_labels = embeddings["labels"][val_indices]

    def make_dataset(embs_dict, labels):
        all_embs = torch.stack(
            [embs_dict[mod] for mod in _MODALITIES], dim=1
        )  # [N, M, 128]
        return TensorDataset(all_embs, labels)

    train_ds = make_dataset(train_embs, train_labels)
    val_ds = make_dataset(val_embs, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model with the correct modality count
    model = LateFusionMetaLearner(phase=phase, n_modalities=n_modalities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        model.train()
        epoch_ce = 0.0
        epoch_nt = 0.0
        n_batches = 0

        for batch_embs, batch_labels in train_loader:
            batch_embs = batch_embs.to(device)
            batch_labels = batch_labels.to(device)

            # View 1: modality-dropout augmented fusion input
            v1 = _apply_modality_dropout(batch_embs, p_dropout, device)
            norm_v1 = [F.normalize(v1[:, i], p=2, dim=-1) for i in range(n_modalities)]
            fusion_v1 = torch.cat(norm_v1, dim=-1)  # [B, M*128]

            # View 2: independent modality-dropout augmented fusion input
            v2 = _apply_modality_dropout(batch_embs, p_dropout, device)
            norm_v2 = [F.normalize(v2[:, i], p=2, dim=-1) for i in range(n_modalities)]
            fusion_v2 = torch.cat(norm_v2, dim=-1)

            # CE loss on view 1 (archetype auxiliary head)
            logits, emb_v1 = model.forward_with_embedding(fusion_v1)
            ce_loss = criterion(logits, batch_labels)

            # NT-Xent: same participant across two dropout views
            _, emb_v2 = model.forward_with_embedding(fusion_v2)
            nt_loss = nt_xent_fusion(emb_v1, emb_v2, nt_xent_temperature)

            loss = ce_loss + lambda_contrastive * nt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_nt += nt_loss.item()
            n_batches += 1

        avg_ce = epoch_ce / n_batches
        avg_nt = epoch_nt / n_batches

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_embs, batch_labels in val_loader:
                batch_embs = batch_embs.to(device)
                batch_labels = batch_labels.to(device)

                norm_embs = [
                    F.normalize(batch_embs[:, i], p=2, dim=-1)
                    for i in range(n_modalities)
                ]
                fusion_input = torch.cat(norm_embs, dim=-1)

                logits, _ = model.forward_with_embedding(fusion_input)
                val_loss_sum += criterion(logits, batch_labels).item()

                val_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
                val_total += batch_labels.shape[0]

        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"ce={avg_ce:.4f}  nt={avg_nt:.4f}  "
            f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if log_mlflow:
            mlflow.log_metrics(
                {
                    "train_ce_loss": avg_ce,
                    "train_nt_loss": avg_nt,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = CHECKPOINT_PATHS["fusion"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(
                    f"  → Early stopping at epoch {epoch + 1} (patience={max_patience})"
                )
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    train()
