"""
Text encoder — frozen sentence-transformer + trainable linear projection.

Architecture:
    PersonaNarrative.text (string)
        -> sentence-transformers: all-MiniLM-L6-v2 (frozen, 384-dim output)
        -> Linear projection (384 -> EMBEDDING_DIM)
        -> LayerNorm(EMBEDDING_DIM)
        -> e_text: [EMBEDDING_DIM=128]

Only the projection layer is trained. Sentence-transformer weights are frozen
and verified before every training run.

Training objective: CE (archetype classification) + NT-Xent (individual identity).
Two augmented views per sample are created by adding independent Gaussian noise
(std=noise_std) to the pre-computed frozen sentence embeddings before the projection.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

import mlflow

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM, PERSONA_LABELS, PERSONA_TO_IDX
from schemas.text import PersonaNarrative

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SENTENCE_DIM = 384  # all-MiniLM-L6-v2 output dimension

DATA_DIR = Path("data/synthetic")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Frozen sentence-transformer + trainable linear projection.

    The sentence-transformer produces 384-dim embeddings. A linear
    projection maps these to ``EMBEDDING_DIM`` (128), followed by
    LayerNorm. Only the projection layer has trainable parameters.

    Parameters
    ----------
    n_classes
        Number of persona classes for the classification head (training only).
    """

    def __init__(self, n_classes: int = 7) -> None:
        super().__init__()
        self.sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        # Freeze ALL sentence-transformer parameters
        for param in self.sentence_model.parameters():
            param.requires_grad = False

        # Trainable projection layer
        self.projection = nn.Linear(SENTENCE_DIM, EMBEDDING_DIM)
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)

        # Classification head — used for supervised training only
        self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode raw text strings to sentence-transformer embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``(len(texts), SENTENCE_DIM)`` with ``dtype=float32``.
        """
        embeddings = self.sentence_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        # sentence-transformers may return float16 on GPU; ensure float32
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project sentence embeddings to ``EMBEDDING_DIM``.

        Parameters
        ----------
        x
            Sentence-transformer embeddings, shape ``(batch_size, SENTENCE_DIM)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, EMBEDDING_DIM)``, dtype ``float32``.
        """
        projected = self.projection(x)
        return self.layer_norm(projected)

    def forward_with_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(embedding, classification_logits)``.

        ``embedding`` has shape ``(batch_size, EMBEDDING_DIM)``.
        ``logits`` has shape ``(batch_size, n_classes)``.
        """
        embedding = self.forward(x)
        logits = self.classifier(embedding)
        return embedding, logits

    def assert_frozen(self) -> None:
        """Assert that all sentence-transformer parameters are frozen.

        Raises ``AssertionError`` if any parameter has ``requires_grad=True``.
        """
        n_trainable = sum(p.requires_grad for p in self.sentence_model.parameters())
        assert n_trainable == 0, (
            f"Sentence-transformer has {n_trainable} trainable parameters — "
            "all should be frozen"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_narratives(
    path: Path = DATA_DIR / "narratives.jsonl",
) -> list[PersonaNarrative]:
    """Load persona narratives from JSONL.

    Raises ``FileNotFoundError`` with a clear message if the file does not
    exist or is empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Narratives file not found: {path}. "
            "Run the generator pipeline first (uv run python -m generator.pipeline)."
        )
    lines = path.read_text().strip().splitlines()
    if not lines:
        raise ValueError(
            f"Narratives file is empty: {path}. "
            "Text encoder cannot run without narrative data."
        )
    return [PersonaNarrative(**json.loads(line)) for line in lines]


def split_by_participant(
    narratives: list[PersonaNarrative],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[PersonaNarrative], list[PersonaNarrative]]:
    """Split narratives into train/val sets by participant_id."""
    participant_ids = sorted(set(n.participant_id for n in narratives))
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(participant_ids)
    split_idx = int(train_ratio * len(participant_ids))
    train_ids = set(participant_ids[:split_idx])
    val_ids = set(participant_ids[split_idx:])

    train_records = [n for n in narratives if n.participant_id in train_ids]
    val_records = [n for n in narratives if n.participant_id in val_ids]
    return train_records, val_records


def narratives_to_tensors(
    encoder: TextEncoder,
    narratives: list[PersonaNarrative],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode narratives and return (features, labels) tensors.

    Parameters
    ----------
    encoder
        TextEncoder with a loaded sentence-transformer.
    narratives
        List of PersonaNarrative records.
    device
        Target device for tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        features: ``(len(narratives), SENTENCE_DIM)``
        labels: ``(len(narratives),)`` with dtype ``long``.
    """
    texts = [n.text for n in narratives]
    with torch.no_grad():
        features = encoder.encode_texts(texts).to(device)
    labels = torch.tensor(
        [PERSONA_TO_IDX[n.persona_id] for n in narratives],
        dtype=torch.long,
        device=device,
    )
    return features, labels


# ---------------------------------------------------------------------------
# Embedding persistence
# ---------------------------------------------------------------------------


def save_embeddings(
    narratives: list[PersonaNarrative],
    embeddings: list[list[float]],
    path: Path,
) -> None:
    """Write embeddings back to narratives JSONL.

    Uses ``dataclasses.replace()`` to create updated copies — never mutates
    the frozen dataclass directly.
    """
    with path.open("w") as f:
        for narrative, emb in zip(narratives, embeddings):
            updated = replace(
                narrative,
                embedding=emb,
                embedding_model_id=SENTENCE_TRANSFORMER_MODEL,
            )
            f.write(json.dumps(asdict(updated)) + "\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def embed_narratives(
    encoder: TextEncoder,
    narratives: list[PersonaNarrative],
) -> list[PersonaNarrative]:
    """Run inference on narratives, returning updated copies with embeddings.

    Skips narratives that already have an embedding (``embedding is not None``).
    """
    to_embed = [n for n in narratives if n.embedding is None]
    if not to_embed:
        return narratives

    texts = [n.text for n in to_embed]
    with torch.no_grad():
        sentence_embs = encoder.encode_texts(texts)
        projected = encoder(sentence_embs)

    # Build updated list
    emb_iter = iter(projected.tolist())
    result: list[PersonaNarrative] = []
    for n in narratives:
        if n.embedding is None:
            emb_vec = next(emb_iter)
            result.append(
                replace(
                    n,
                    embedding=emb_vec,
                    embedding_model_id=SENTENCE_TRANSFORMER_MODEL,
                )
            )
        else:
            result.append(n)
    return result


# ---------------------------------------------------------------------------
# NT-Xent for augmented view pairs (SimCLR-style)
# ---------------------------------------------------------------------------


def nt_xent_views(
    emb_v1: torch.Tensor,
    emb_v2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent for a matched pair of augmented views.

    emb_v1[i] and emb_v2[i] are the positive pair for sample i.
    All other cross-sample pairs in the batch are negatives.
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


def _compute_text_similarity_delta(
    model: TextEncoder,
    val_features: torch.Tensor,
    val_records: list[PersonaNarrative],
    noise_std: float,
    n_samples: int = 300,
) -> float:
    """Same-participant view similarity minus same-archetype cross-participant baseline."""
    model.eval()
    by_persona: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(val_records):
        by_persona[r.persona_id].append(i)

    rng = torch.Generator()
    rng.manual_seed(0)
    same_sims: list[float] = []
    cross_sims: list[float] = []

    with torch.no_grad():
        for idx in range(min(n_samples, len(val_records))):
            feat = val_features[idx].unsqueeze(0)
            n1 = torch.randn_like(feat, generator=rng) * noise_std
            n2 = torch.randn_like(feat, generator=rng) * noise_std
            e1 = F.normalize(model(feat + n1), dim=1)
            e2 = F.normalize(model(feat + n2), dim=1)
            same_sims.append((e1 * e2).sum().item())

            pid = val_records[idx].persona_id
            same_arch = [j for j in by_persona[pid] if j != idx]
            if not same_arch:
                continue
            j = same_arch[torch.randint(len(same_arch), (1,), generator=rng).item()]  # type: ignore[arg-type]
            feat_j = val_features[j].unsqueeze(0)
            ej = F.normalize(model(feat_j), dim=1)
            cross_sims.append((e1 * ej).sum().item())

    if not same_sims or not cross_sims:
        return 0.0
    return float(np.mean(same_sims) - np.mean(cross_sims))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    narratives: Optional[list[PersonaNarrative]] = None,
    *,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
    log_mlflow: bool = True,
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
    noise_std: float = 0.01,
) -> TextEncoder:
    """Train the text encoder projection layer with CE + NT-Xent multi-task objective.

    The sentence-transformer is fully frozen. Only the linear projection
    (384 -> EMBEDDING_DIM) and classification head are trained.

    Sentence embeddings are pre-computed once (frozen backbone). At each training
    step, two augmented views are created by adding independent Gaussian noise
    (std=noise_std) to the pre-computed embeddings before the projection head.
    NT-Xent treats position-matched view pairs as positives (SimCLR-style).

    Parameters
    ----------
    narratives
        Pre-loaded narratives. If ``None``, loads from ``data/synthetic/``.
    n_epochs
        Number of training epochs.
    batch_size
        Mini-batch size.
    lr
        Learning rate for the projection layer.
    weight_decay
        AdamW weight decay.
    train_ratio
        Fraction of participants assigned to train set.
    seed
        Random seed for participant splitting.
    device
        ``"cpu"`` or ``"cuda"``.
    log_mlflow
        Whether to log the run to MLflow.
    lambda_contrastive
        Weight for NT-Xent loss (total = CE + lambda * NT-Xent).
    nt_xent_temperature
        Temperature for NT-Xent (default 0.07).
    noise_std
        Std of Gaussian noise added to sentence embeddings for augmentation.
        all-MiniLM-L6-v2 outputs are unit-normalised so 0.01 ≈ 1% noise.
        Increase to 0.05 if similarity_delta criterion is not met.

    Returns
    -------
    TextEncoder
        Trained encoder (with classification head still attached).
    """
    if narratives is None:
        narratives = load_narratives()

    # Initialise model and verify frozen sentence-transformer
    n_classes = len(PERSONA_LABELS)
    model = TextEncoder(n_classes=n_classes).to(device)
    model.assert_frozen()

    # Split by participant
    train_records, val_records = split_by_participant(
        narratives, train_ratio=train_ratio, seed=seed
    )

    # Pre-compute sentence embeddings (frozen backbone — do this once)
    train_features, train_labels = narratives_to_tensors(
        model, train_records, device=device
    )
    val_features, val_labels = narratives_to_tensors(model, val_records, device=device)

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Optimiser only over projection + classifier params (sentence model is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_trainable_state: dict[str, torch.Tensor] = {}
    patience, patience_counter = 10, 0

    def _evaluate() -> tuple[float, float]:
        """Returns (val_loss, val_acc)."""
        model.eval()
        with torch.no_grad():
            _, logits = model.forward_with_logits(val_features)
            loss = criterion(logits, val_labels).item()
            preds = logits.argmax(dim=1)
            acc = (preds == val_labels).float().mean().item()
        return loss, acc

    def _train_loop() -> None:
        nonlocal best_val_loss, best_trainable_state, patience_counter

        for epoch in range(n_epochs):
            model.projection.train()
            model.classifier.train()
            epoch_ce = 0.0
            epoch_nt = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # CE loss on original (noiseless) sentence embeddings
                _, logits = model.forward_with_logits(batch_x)
                ce_loss = criterion(logits, batch_y)

                # NT-Xent: two Gaussian-noise views of the sentence embedding
                noise1 = torch.randn_like(batch_x) * noise_std
                noise2 = torch.randn_like(batch_x) * noise_std
                emb_v1 = model(batch_x + noise1)
                emb_v2 = model(batch_x + noise2)
                nt_loss = nt_xent_views(emb_v1, emb_v2, nt_xent_temperature)

                loss = ce_loss + lambda_contrastive * nt_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_ce += ce_loss.item()
                epoch_nt += nt_loss.item()
                n_batches += 1

            avg_ce = epoch_ce / max(n_batches, 1)
            avg_nt = epoch_nt / max(n_batches, 1)
            val_loss, val_acc = _evaluate()

            if log_mlflow:
                mlflow.log_metrics(
                    {
                        "train_ce_loss": avg_ce,
                        "train_nt_loss": avg_nt,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )

            print(
                f"Epoch {epoch + 1}/{n_epochs}  "
                f"ce={avg_ce:.4f}  nt={avg_nt:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_trainable_state = {
                    name: param.clone()
                    for name, param in model.state_dict().items()
                    if not name.startswith("sentence_model.")
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    if log_mlflow:
        with mlflow.start_run(run_name="text_encoder_v2_contrastive"):
            mlflow.set_tag("modality", "text")
            mlflow.log_params(
                {
                    "lr": lr,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "train_ratio": train_ratio,
                    "embedding_dim": EMBEDDING_DIM,
                    "sentence_model": SENTENCE_TRANSFORMER_MODEL,
                    "lambda_contrastive": lambda_contrastive,
                    "nt_xent_temperature": nt_xent_temperature,
                    "noise_std": noise_std,
                    "objective": "ce+nt_xent",
                }
            )
            _train_loop()
            final_val_loss, final_val_acc = _evaluate()
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("final_val_acc", final_val_acc)

            sim_delta = _compute_text_similarity_delta(
                model, val_features, val_records, noise_std
            )
            mlflow.log_metric("similarity_delta_within_vs_cross_archetype", sim_delta)
            print(f"Similarity delta: {sim_delta:.4f}")
            print(f"Final val_acc: {final_val_acc:.4f}")
    else:
        _train_loop()

    # Save checkpoint (only trainable parameters, not frozen sentence-transformer)
    checkpoint_path = CHECKPOINT_PATHS["text"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_trainable_state:
        torch.save(best_trainable_state, checkpoint_path)
    else:
        trainable_state_dict = {
            name: param
            for name, param in model.state_dict().items()
            if not name.startswith("sentence_model.")
        }
        torch.save(trainable_state_dict, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    return model


if __name__ == "__main__":
    trained_model = train()
    print("Text encoder training complete.")
    print(f"Encoder output dim: {EMBEDDING_DIM}")
