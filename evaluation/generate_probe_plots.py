"""
Generate probe evaluation plots for the 02_encoder_probing notebook.

Produces:
- UMAP 2D projections colored by persona (text + psychographic encoders)
- Strategy recovery bar chart
- Confusion matrices
- Cosine similarity heatmap (text encoder)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from schemas import PERSONA_LABELS, PERSONA_TO_IDX

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
OUTPUT_DIR = Path("notebooks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = plt.cm.tab10(np.linspace(0, 1, 7))


def generate_text_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Generate text encoder embeddings for all narratives."""
    from encoders.text.embed import train as train_text
    from schemas.text import PersonaNarrative

    path = DATA_DIR / "narratives.jsonl"
    narratives = [
        PersonaNarrative(**json.loads(line))
        for line in path.read_text().strip().split("\n")
        if line.strip()
    ]

    logger.info("Training text encoder for plot generation...")
    encoder = train_text(
        narratives=narratives,
        n_epochs=20,
        batch_size=64,
        device="cpu",
        log_mlflow=False,
    )
    encoder.eval()

    texts = [n.text for n in narratives]
    with torch.no_grad():
        sentence_embs = encoder.encode_texts(texts)
        projected = encoder(sentence_embs)

    embeddings = projected.cpu().numpy()
    labels = np.array([PERSONA_TO_IDX.get(n.persona_id, 0) for n in narratives])
    return embeddings, labels


def generate_psychographic_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Generate psychographic encoder embeddings."""
    from encoders.psychographic.features import to_feature_vector
    from encoders.psychographic.train import train as train_psycho
    from schemas.psychographic import PsychographicVector

    path = DATA_DIR / "psychographics.jsonl"
    records = [
        PsychographicVector(**json.loads(line))
        for line in path.read_text().strip().split("\n")
        if line.strip()
    ]

    logger.info("Training psychographic encoder for plot generation...")
    encoder = train_psycho(
        records=records, n_epochs=40, batch_size=128, device="cpu", log_mlflow=False
    )
    encoder.eval()

    embs, lbls = [], []
    with torch.no_grad():
        for r in records:
            vec = to_feature_vector(r).unsqueeze(0)
            emb = encoder(vec).cpu().numpy().squeeze(0)
            embs.append(emb)
            lbls.append(PERSONA_TO_IDX.get(r.persona_id, 0))

    return np.stack(embs), np.array(lbls)


def generate_trace_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Generate trace encoder embeddings from saved checkpoint or fresh training."""
    from collections import defaultdict

    from encoders.trace.model import TraceEncoder
    from encoders.trace.tokeniser import build_vocab, tokenise_trial
    from schemas.trace import AcquisitionEvent, TrialRecord

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

    events_by_trial: dict[str, list[AcquisitionEvent]] = defaultdict(list)
    for ev in events:
        events_by_trial[ev.trial_id].append(ev)

    vocab = build_vocab(events)
    n_attributes = max(len(vocab.get("attribute", {})), 1) + 1
    n_alternatives = max(len(vocab.get("alternative", {})), 1) + 1

    model_path = Path("models/trace_encoder.pt")
    if model_path.exists():
        logger.info("Loading trace encoder from checkpoint")
        encoder = TraceEncoder(
            n_attributes=n_attributes,
            n_alternatives=n_alternatives,
            n_classes=7,
        )
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=False)
    else:
        logger.info("Training trace encoder from scratch...")
        from encoders.trace.train import train as train_trace
        import mlflow

        with mlflow.start_run(run_name="trace_plot_gen"):
            mlflow.set_tag("modality", "trace")
            encoder = train_trace(device="cpu", n_epochs=30, seed=42)

    encoder.eval()

    embs, lbls = [], []
    with torch.no_grad():
        for tid, trial in trials.items():
            trial_events = events_by_trial.get(tid, [])
            if not trial_events:
                continue
            tokens, mask = tokenise_trial(trial_events, trial, vocab)
            tokens_b = tokens.unsqueeze(0)
            mask_b = mask.unsqueeze(0) if mask is not None else None
            emb = encoder(tokens_b, mask_b)
            embs.append(emb.cpu().numpy().squeeze(0))
            lbls.append(PERSONA_TO_IDX.get(trial.persona_id, 0))

    return np.stack(embs), np.array(lbls)


def plot_umap(
    embeddings: np.ndarray, labels: np.ndarray, title: str, filename: str
) -> None:
    """Generate and save a UMAP 2D projection plot."""
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        proj = reducer.fit_transform(embeddings)
    except ImportError:
        logger.warning("umap not installed, skipping UMAP plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(PERSONA_LABELS):
        mask = labels == i
        ax.scatter(
            proj[mask, 0], proj[mask, 1], c=[COLORS[i]], label=name, s=8, alpha=0.6
        )

    ax.set_title(f"UMAP Projection — {title}", fontsize=14)
    ax.legend(markerscale=3, fontsize=9, loc="lower left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    logger.info("Saved UMAP plot: %s", filename)


def plot_strategy_recovery(results: dict[str, float], filename: str) -> None:
    """Bar chart comparing strategy recovery across encoders."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    values = [results[n] for n in names]
    bars = ax.bar(names, values, color=[COLORS[i] for i in range(len(names))])

    # Add threshold lines
    thresholds = {
        "trace": 0.85,
        "transaction": 0.60,
        "text": 0.70,
        "psychographic": 0.75,
    }
    for i, name in enumerate(names):
        if name in thresholds:
            ax.axhline(
                y=thresholds[name],
                xmin=i / len(names),
                xmax=(i + 1) / len(names),
                color="red",
                linestyle="--",
                linewidth=1.5,
            )

    ax.set_ylabel("Strategy Recovery Accuracy")
    ax.set_title("Strategy Recovery by Encoder Modality")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart: %s", filename)


def plot_confusion_matrix(
    embeddings: np.ndarray, labels: np.ndarray, title: str, filename: str
) -> None:
    """Train logistic regression and plot confusion matrix."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(embeddings, labels))

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_val_s)
    acc = accuracy_score(y_val, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_val,
        y_pred,
        display_labels=PERSONA_LABELS,
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"{title} (Accuracy: {acc:.1%})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s (accuracy: %.2f%%)", filename, acc * 100)


def main() -> None:
    logger.info("Generating probe evaluation plots...")

    # Text embeddings
    text_embs, text_labels = generate_text_embeddings()
    plot_umap(
        text_embs,
        text_labels,
        "Text Encoder (sentence-transformer + projection)",
        "text_umap.png",
    )
    plot_confusion_matrix(
        text_embs, text_labels, "Text Encoder Confusion Matrix", "text_confusion.png"
    )

    # Psychographic embeddings
    psycho_embs, psycho_labels = generate_psychographic_embeddings()
    plot_umap(
        psycho_embs, psycho_labels, "Psychographic Encoder (MLP)", "psycho_umap.png"
    )
    plot_confusion_matrix(
        psycho_embs,
        psycho_labels,
        "Psychographic Encoder Confusion Matrix",
        "psycho_confusion.png",
    )

    # Trace embeddings (from checkpoint if available)
    try:
        trace_embs, trace_labels = generate_trace_embeddings()
        plot_umap(
            trace_embs,
            trace_labels,
            "Trace Encoder (Transformer Contrastive)",
            "trace_umap.png",
        )
        plot_confusion_matrix(
            trace_embs,
            trace_labels,
            "Trace Encoder Confusion Matrix",
            "trace_confusion.png",
        )
    except Exception as e:
        logger.warning("Trace encoder plots skipped: %s", e)

    # Strategy recovery summary
    results = {
        "text": 0.99,
        "psychographic": 1.00,
        "trace": 0.3775,
        "transaction": None,  # training bug
    }
    valid_results = {k: v for k, v in results.items() if v is not None}
    plot_strategy_recovery(valid_results, "strategy_recovery_bars.png")

    # Cosine similarity heatmap for text encoder
    from evaluation.probe import compute_cosine_similarity_stats

    cos_sim = compute_cosine_similarity_stats(
        text_embs, text_labels, label_names=PERSONA_LABELS
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cos_sim["pairwise_matrix"], cmap="RdYlBu_r", vmin=-0.2, vmax=0.8)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(PERSONA_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(PERSONA_LABELS)
    ax.set_title("Text Encoder: Inter-Persona Cosine Similarity")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "text_cosine_sim_heatmap.png", dpi=150)
    plt.close(fig)

    logger.info("All plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
