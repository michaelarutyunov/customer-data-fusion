"""
evaluation/counterfactual_option_b.py

Option B counterfactual simulation: re-run the generator for a target participant
with a modified PersonaConfig, re-encode through frozen encoders, and measure CDT
embedding cosine distance shift vs. the participant's original baseline embedding.

This is the higher-fidelity complement to Option A (archetype-level redistribution
in evaluation/counterfactual.py).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fusion.meta_learner import LateFusionMetaLearner
from generator.pipeline import run_pipeline
from schemas import EMBEDDING_DIM

log = logging.getLogger(__name__)

CACHE_PATH = Path("models/fusion_embeddings_cache.pt")
FUSION_CHECKPOINT = Path("models/fusion_meta_learner.pt")

# Threshold from bead c11: 2× intra-archetype cosine distance SD
MEANINGFUL_SHIFT_THRESHOLD = 0.27


def _load_fusion_model(
    checkpoint_path: Path = FUSION_CHECKPOINT,
) -> LateFusionMetaLearner:
    """Load the frozen fusion meta-learner."""
    model = LateFusionMetaLearner()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _encode_single_participant(
    data_dir: Path,
    participant_id: str,
    baseline_text_embedding: torch.Tensor,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Encode a single participant's counterfactual data from a custom directory.

    Encodes trace, transaction, and psychographic modalities from the generated
    data. Text embedding is copied from baseline (narratives are not regenerated).

    Parameters
    ----------
    data_dir : Path
        Directory containing JSONL files for the counterfactual participant.
    participant_id : str
        Participant ID to encode.
    baseline_text_embedding : torch.Tensor
        Text embedding from baseline (copied because skip_narratives=True).
    device : str
        Target device for computation.

    Returns
    -------
    dict[str, torch.Tensor]
        Modality embeddings: trace, transaction, text, psychographic — each [128].
    """
    from encoders.psychographic.features import to_feature_vector
    from encoders.trace.tokeniser import build_vocab, tokenise_trial
    from encoders.transaction.features import sort_transactions_most_recent_first
    from encoders.transaction.model import TransactionEncoder as TxEncoder
    from fusion.train import load_encoders
    from schemas.psychographic import PsychographicVector
    from schemas.trace import AcquisitionEvent, TrialRecord
    from schemas.transaction import TransactionRecord

    encoders = load_encoders(device=device)

    result: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        # ── Trace encoding ────────────────────────────────────────────────
        events: list[AcquisitionEvent] = []
        for line in (data_dir / "traces.jsonl").read_text().strip().splitlines():
            r = json.loads(line)
            if r.get("participant_id") == participant_id:
                events.append(AcquisitionEvent(**r))

        trials: list[TrialRecord] = []
        for line in (data_dir / "trials.jsonl").read_text().strip().splitlines():
            r = json.loads(line)
            if r.get("participant_id") == participant_id:
                trials.append(TrialRecord(**r))

        trial_embs = []
        if events and trials:
            vocab = build_vocab(events)
            for trial in trials:
                tid_events = [e for e in events if e.trial_id == trial.trial_id]
                if not tid_events:
                    continue
                tokens, mask = tokenise_trial(tid_events, trial, vocab)
                tokens_b = tokens.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device) if mask is not None else None
                emb = encoders["trace"](tokens_b, mask_b)
                trial_embs.append(emb.squeeze(0))

        if trial_embs:
            result["trace"] = torch.stack(trial_embs).mean(0)
        else:
            result["trace"] = torch.zeros(EMBEDDING_DIM, device=device)

        # ── Transaction encoding ──────────────────────────────────────────
        raw_txs: list[dict] = []
        for line in (data_dir / "transactions.jsonl").read_text().strip().splitlines():
            r = json.loads(line)
            if r.get("participant_id") == participant_id:
                raw_txs.append(r)

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
            lengths = torch.tensor([len(sorted_tx)], device=device)
            result["transaction"] = tx_enc(token_seq_b, lengths).squeeze(0)
        else:
            result["transaction"] = torch.zeros(EMBEDDING_DIM, device=device)

        # ── Text: copy from baseline ──────────────────────────────────────
        result["text"] = baseline_text_embedding.to(device)

        # ── Psychographic encoding ────────────────────────────────────────
        psycho_record: dict | None = None
        for line in (
            (data_dir / "psychographics.jsonl").read_text().strip().splitlines()
        ):
            r = json.loads(line)
            if r.get("participant_id") == participant_id:
                psycho_record = r
                break

        if psycho_record:
            psycho_vec = PsychographicVector(
                **{
                    k: v
                    for k, v in psycho_record.items()
                    if k in PsychographicVector.__dataclass_fields__
                }
            )
            raw_vec = to_feature_vector(psycho_vec).to(device)
            result["psychographic"] = encoders["psychographic"](
                raw_vec.unsqueeze(0)
            ).squeeze(0)
        else:
            result["psychographic"] = torch.zeros(EMBEDDING_DIM, device=device)

    return result


def _extract_baseline_embedding(
    cache: dict,
    model: LateFusionMetaLearner,
    participant_idx: int,
) -> torch.Tensor:
    """Extract CDT embedding for a single participant from the cache."""
    stacked = torch.cat(
        [
            cache["trace"][participant_idx].unsqueeze(0),
            cache["transaction"][participant_idx].unsqueeze(0),
            cache["text"][participant_idx].unsqueeze(0),
            cache["psychographic"][participant_idx].unsqueeze(0),
        ],
        dim=1,
    )  # [1, 512]
    with torch.no_grad():
        return model.embed(stacked).squeeze(0)  # [128]


def simulate_counterfactual(
    participant_id: str,
    overrides: dict[str, float],
    baseline_cache_path: Path = CACHE_PATH,
) -> dict[str, Any]:
    """Simulate a counterfactual by re-running the generator with modified PersonaConfig.

    Parameters
    ----------
    participant_id : str
        Original participant ID (e.g. 'price_lex_0042').
    overrides : dict[str, float]
        Flat field name → new value (e.g. {'price_sensitivity': 0.99}).
    baseline_cache_path : Path
        Path to the fusion embeddings cache containing baseline embeddings.

    Returns
    -------
    dict with keys:
        participant_id, overrides, baseline_embedding (Tensor[128]),
        counterfactual_embedding (Tensor[128]), cosine_distance_shift (float)
    """
    # Load baseline cache and fusion model
    cache = torch.load(baseline_cache_path, weights_only=False)  # noqa: S614 — cache contains participant_ids (list[str]) which requires weights_only=False
    model = _load_fusion_model()

    # Find participant index in cache
    pids = cache["participant_ids"]
    idx = pids.index(participant_id)

    # Extract baseline CDT embedding
    baseline_emb = _extract_baseline_embedding(cache, model, idx)

    # Derive archetype and generated participant ID
    archetype = participant_id.rsplit("_", 1)[0]
    generated_pid = f"{archetype}_0000"

    # Re-run generator to temp directory with overrides
    with tempfile.TemporaryDirectory() as tmpdir:
        run_pipeline(
            n=1,
            archetypes=[archetype],
            counterfactual_overrides={generated_pid: overrides},
            output_dir=Path(tmpdir),
            skip_narratives=True,
        )

        # Encode counterfactual data (text copied from baseline)
        baseline_text = cache["text"][idx]
        cf_modality_embs = _encode_single_participant(
            Path(tmpdir), generated_pid, baseline_text
        )

    # Compute counterfactual CDT embedding
    cf_stacked = torch.cat(
        [
            cf_modality_embs["trace"].unsqueeze(0),
            cf_modality_embs["transaction"].unsqueeze(0),
            cf_modality_embs["text"].unsqueeze(0),
            cf_modality_embs["psychographic"].unsqueeze(0),
        ],
        dim=1,
    )  # [1, 512]

    with torch.no_grad():
        cf_emb = model.embed(cf_stacked).squeeze(0)  # [128]

    # Compute cosine distance shift
    cosine_distance_shift = (
        1 - F.cosine_similarity(baseline_emb.unsqueeze(0), cf_emb.unsqueeze(0)).item()
    )

    meaningful = cosine_distance_shift >= MEANINGFUL_SHIFT_THRESHOLD
    log.info(
        "counterfactual.shift: participant_id=%s shift=%.4f meaningful=%s threshold=%.2f",
        participant_id,
        cosine_distance_shift,
        meaningful,
        MEANINGFUL_SHIFT_THRESHOLD,
    )

    return {
        "participant_id": participant_id,
        "overrides": overrides,
        "baseline_embedding": baseline_emb,
        "counterfactual_embedding": cf_emb,
        "cosine_distance_shift": cosine_distance_shift,
    }
