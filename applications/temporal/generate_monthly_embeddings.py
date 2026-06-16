"""
Generate monthly CDT embeddings using the frozen fusion meta-learner.

Reads month-partitioned modality files, passes them through the frozen fusion
model, and writes one embedding per participant-month to disk.

Output: applications/_cache/cdt_embeddings_monthly.parquet
Columns: [participant_id, month, cdt] where cdt is a 128-dim float array.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
from tqdm import tqdm

from fusion.meta_learner import LateFusionMetaLearner
from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM
from schemas.transaction import Channel, PurchaseType

log = structlog.get_logger(__name__)

# Allowlist of valid modalities (prevents path traversal)
_VALID_MODALITIES = frozenset(
    {"transactions", "clickstream", "campaigns", "traces", "psychographics"}
)


def _load_monthly_modality(
    modality: str, months: list[int], data_dir: Path
) -> list[dict]:
    """Load month-partitioned files for a modality.

    Parameters
    ----------
    modality : str
        Modality name (must be in _VALID_MODALITIES)
    months : list of int
        Months to load (1-indexed)
    data_dir : Path
        Directory containing month-partitioned JSONL files

    Returns
    -------
    list of dict
        Records from the specified months

    Raises
    ------
    ValueError
        If modality is not in the allowlist
    """
    if modality not in _VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Must be one of {sorted(_VALID_MODALITIES)}"
        )

    records: list[dict] = []
    for month in months:
        path = data_dir / f"{modality}_month_{month:02d}.jsonl"
        if not path.exists():
            log.warning("monthly_embeddings.missing_month_file", path=str(path))
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    log.info(
        "monthly_embeddings.loaded_modality",
        modality=modality,
        n_months=len(months),
        n_records=len(records),
    )
    return records


def _encode_month(
    month: int,
    data_dir: Path,
    fusion_model: LateFusionMetaLearner,
    device: torch.device,
) -> pd.DataFrame:
    """Encode a single month's data using the frozen fusion model.

    Parameters
    ----------
    month : int
        Month number (1-12)
    data_dir : Path
        Directory containing month-partitioned data files
    fusion_model : FusionMetaLearner
        Frozen fusion meta-learner (loaded once, reused across months)
    device : torch.device
        Torch device (cpu or cuda)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [participant_id, month, cdt]
    """
    log.info("monthly_embeddings.encoding_month", month=month)

    # Load all modalities for this month
    all_records: dict[str, list[dict]] = {}
    for modality in _VALID_MODALITIES:
        records = _load_monthly_modality(modality, [month], data_dir)
        all_records[modality] = records

    # Group by participant (each participant may have multiple trials/transactions)
    participant_data: dict[str, dict[str, list]] = {}
    for modality, records in all_records.items():
        for rec in records:
            pid = rec["participant_id"]
            if pid not in participant_data:
                participant_data[pid] = {m: [] for m in _VALID_MODALITIES}
            participant_data[pid][modality].append(rec)

    # Encode each participant using frozen modality encoders + fusion model
    embeddings: list[dict] = []

    # Build participant index map for this month
    participant_ids = list(participant_data.keys())
    pid_to_idx = {pid: i for i, pid in enumerate(participant_ids)}

    # Prepare per-modality data structures for encoding
    from collections import defaultdict
    from schemas.trace import AcquisitionEvent, TrialRecord
    from schemas.transaction import TransactionRecord
    from schemas.psychographic import PsychographicVector
    from schemas.clickstream import ClickstreamEvent
    from schemas.campaign import CampaignEvent

    # Group data by participant for each modality
    events_by_pid: dict[str, list] = defaultdict(list)
    trials_by_pid: dict[str, list] = defaultdict(list)
    tx_by_pid: dict[str, list] = defaultdict(list)
    click_sessions_by_pid: dict[str, list] = defaultdict(list)
    campaign_events_by_pid: dict[str, list] = defaultdict(list)
    psychographic_by_pid: dict[str, dict] = {}

    # Process traces (if available for this month)
    if "traces" in all_records:
        for rec in all_records["traces"]:
            pid = rec.get("participant_id", "")
            if pid in pid_to_idx:
                if rec.get("record_type") == "event":
                    events_by_pid[pid].append(
                        AcquisitionEvent(**{k: v for k, v in rec.items() if k in AcquisitionEvent.__dataclass_fields__})
                    )
                elif rec.get("record_type") == "trial":
                    trials_by_pid[pid].append(
                        TrialRecord(**{k: v for k, v in rec.items() if k in TrialRecord.__dataclass_fields__})
                    )

    # Process transactions
    if "transactions" in all_records:
        for rec in all_records["transactions"]:
            pid = rec.get("participant_id", "")
            if pid in pid_to_idx:
                tx_by_pid[pid].append(rec)

    # Process clickstream
    if "clickstream" in all_records:
        raw_by_pid_sid: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for rec in all_records["clickstream"]:
            pid = rec.get("participant_id", "")
            customer_id = rec.get("customer_id", "")
            if not pid or customer_id == "anonymous":
                continue
            if "event_type" not in rec:
                continue
            if pid in pid_to_idx:
                # Filter to known dataclass fields to prevent injection
                filtered_fields = {k: v for k, v in rec.items() if k in ClickstreamEvent.__dataclass_fields__}
                raw_by_pid_sid[pid][rec.get("session_id", "")].append(ClickstreamEvent(**filtered_fields))
        for pid, sessions in raw_by_pid_sid.items():
            click_sessions_by_pid[pid] = [
                sorted(s, key=lambda e: e.event_ts) for s in sessions.values()
            ]

    # Process campaigns
    if "campaigns" in all_records:
        for rec in all_records["campaigns"]:
            pid = rec.get("participant_id", "")
            if pid in pid_to_idx:
                # Filter to known dataclass fields to prevent injection
                filtered_fields = {k: v for k, v in rec.items() if k in CampaignEvent.__dataclass_fields__}
                campaign_events_by_pid[pid].append(CampaignEvent(**filtered_fields))
        for pid in campaign_events_by_pid:
            campaign_events_by_pid[pid].sort(key=lambda e: e.sent_ts)

    # Process psychographics (take first record per participant)
    if "psychographics" in all_records:
        for rec in all_records["psychographics"]:
            pid = rec.get("participant_id", "")
            if pid in pid_to_idx and pid not in psychographic_by_pid:
                psychographic_by_pid[pid] = rec

    # Build trace vocab if traces exist
    trace_vocab = None
    if events_by_pid or trials_by_pid:
        from encoders.trace.tokeniser import build_vocab
        all_events = [ev for evs in events_by_pid.values() for ev in evs]
        if all_events:
            trace_vocab = build_vocab(all_events)

    # Encode each participant
    for participant_id in tqdm(participant_ids, desc=f"Month {month}"):
        modality_embeddings = []

        with torch.no_grad():
            # Trace encoding
            if "trace" in fusion_model.available_modalities and trace_vocab is not None:
                trial_embs = []
                for trial in trials_by_pid.get(participant_id, []):
                    tid_events = [e for e in events_by_pid.get(participant_id, []) if e.trial_id == trial.trial_id]
                    if not tid_events:
                        continue
                    from encoders.trace.tokeniser import tokenise_trial
                    tokens, mask = tokenise_trial(tid_events, trial, trace_vocab)
                    tokens_b = tokens.unsqueeze(0).to(device)
                    mask_b = mask.unsqueeze(0).to(device) if mask is not None else None
                    emb = fusion_model.modality_encoders["trace"](tokens_b, mask_b)
                    trial_embs.append(emb.squeeze(0))
                if trial_embs:
                    trace_emb = torch.stack(trial_embs).mean(0)
                    modality_embeddings.append(trace_emb)
                else:
                    modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))

            # Transaction encoding
            if "transaction" in fusion_model.available_modalities:
                raw_txs = tx_by_pid.get(participant_id, [])
                if raw_txs:
                    from encoders.transaction.features import sort_transactions_most_recent_first
                    from schemas.transaction import Channel, PurchaseType

                    tx_records = []
                    for r in raw_txs:
                        tx_rec = TransactionRecord(
                            participant_id=r.get("participant_id", ""),
                            persona_id=r.get("persona_id", ""),
                            transaction_id=r.get("transaction_id", ""),
                            days_before_session=r.get("days_before_session", 0),
                            category=r.get("category", ""),
                            product_id=r.get("product_id", ""),
                            brand_tier=r.get("brand_tier", ""),
                            price_paid_normalised=r.get("price_paid_normalised", 0.0),
                            quantity=r.get("quantity", 1),
                            channel=r.get("channel", Channel.ONLINE),
                            purchase_type=r.get("purchase_type", PurchaseType.PLANNED),
                            on_promotion=r.get("on_promotion", False),
                        )
                        tx_records.append(tx_rec)

                    tx_records = sort_transactions_most_recent_first(tx_records)
                    tx_enc = fusion_model.modality_encoders["transaction"]
                    token_seq = tx_enc.vocab.encode_sequence(tx_records)
                    token_seq_b = token_seq.unsqueeze(0).to(device)
                    lengths = torch.tensor([len(tx_records)], device=device)
                    tx_emb = tx_enc(token_seq_b, lengths).squeeze(0)
                    modality_embeddings.append(tx_emb)
                else:
                    modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))

            # Text encoding
            if "text" in fusion_model.available_modalities:
                psycho = psychographic_by_pid.get(participant_id, {})
                narrative = psycho.get("narrative", "")
                if narrative:
                    from encoders.text.embed import TextEncoder
                    text_encoder = TextEncoder(device)
                    text_emb = text_encoder.encode_text(narrative)
                    modality_embeddings.append(text_emb)
                else:
                    modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))

            # Psychographic encoding
            if "psychographic" in fusion_model.available_modalities:
                psycho = psychographic_by_pid.get(participant_id, {})
                if psycho:
                    from encoders.psychographic.features import to_feature_vector
                    # Filter to known dataclass fields to prevent injection
                    filtered_psycho = {k: v for k, v in psycho.items() if k in PsychographicVector.__dataclass_fields__}
                    psych_vector = to_feature_vector(PsychographicVector(**filtered_psycho))
                    psych_emb = fusion_model.modality_encoders["psychographic"](psych_vector)
                    modality_embeddings.append(psych_emb)
                else:
                    modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))

            # Clickstream encoding
            if "clickstream" in fusion_model.available_modalities:
                sessions = click_sessions_by_pid.get(participant_id, [])
                if sessions:
                    from encoders.clickstream.features import MAX_EVENTS_PER_SESSION, MAX_SESSIONS

                    click_enc = fusion_model.modality_encoders["clickstream"]
                    sessions_sorted = sorted(sessions, key=lambda s: s[0].event_ts if s else "")
                    sessions_sorted = sessions_sorted[-MAX_SESSIONS:]
                    session_embeddings = []
                    for sess in sessions_sorted:
                        tokens = click_enc.vocab.encode_session(sess)[:MAX_EVENTS_PER_SESSION]
                        if tokens.size(0) > 0:
                            tokens_b = tokens.unsqueeze(0).to(device)
                            sess_emb = click_enc(tokens_b).squeeze(0)
                            session_embeddings.append(sess_emb)
                    if session_embeddings:
                        click_emb = torch.stack(session_embeddings).mean(0)
                        modality_embeddings.append(click_emb)
                    else:
                        modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
                else:
                    modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))

            # Campaign encoding
            if "campaign" in fusion_model.available_modalities:
                events = campaign_events_by_pid.get(participant_id, [])
                if events:
                    from encoders.campaign.features import TOKEN_DIM

                    camp_enc = fusion_model.modality_encoders["campaign"]
                    raw = camp_enc.vocab.encode_sequence(events).to(device)
                    seq_len = raw.size(0)
                    if raw.size(1) < TOKEN_DIM:
                        pad = torch.zeros(seq_len, TOKEN_DIM - raw.size(1), dtype=raw.dtype, device=device)
                        raw = torch.cat([raw, pad], dim=1)
                    raw_b = raw.unsqueeze(0)
                    camp_emb = camp_enc(raw_b).squeeze(0)
                    modality_embeddings.append(camp_emb)
                else:
                    modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                modality_embeddings.append(torch.zeros(EMBEDDING_DIM, device=device))

        # Concatenate all modality embeddings
        combined = torch.cat(modality_embeddings, dim=0)

        # Pass through fusion model to get CDT embedding
        combined_b = combined.unsqueeze(0).to(device)
        logits, cdt_embedding = fusion_model.forward_with_embedding(combined_b)
        embedding = cdt_embedding.squeeze(0).cpu()

        embeddings.append(
            {
                "participant_id": participant_id,
                "month": month,
                "cdt": embedding.numpy(),
            }
        )

    return pd.DataFrame(embeddings)


def generate_monthly_embeddings(
    n_months: int = 12,
    data_dir: Path | str = "data/synthetic",
    output_path: Path | str = "applications/_cache/cdt_embeddings_monthly.parquet",
    fusion_model_path: Path | str = "models/fusion_metalearner.pt",
    force: bool = False,
) -> None:
    """Generate CDT embeddings for all participants across all months.

    Parameters
    ----------
    n_months : int
        Number of months to encode (default: 12)
    data_dir : Path or str
        Directory containing month-partitioned data files
    output_path : Path or str
        Output parquet file path
    fusion_model_path : Path or str
        Path to frozen fusion meta-learner checkpoint
    force : bool
        Overwrite existing output file if True
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    fusion_model_path = Path(fusion_model_path)

    # SECURITY: Validate output path is within allowed directory
    output_path = output_path.resolve()
    allowed_base = Path("applications/_cache").resolve()
    if not output_path.is_relative_to(allowed_base):
        raise ValueError(f"Output path must be within {allowed_base}")

    # Check if output already exists
    if output_path.exists() and not force:
        log.info(
            "monthly_embeddings.cache_hit",
            output_path=str(output_path),
            message="Embeddings already cached. Use --force to regenerate.",
        )
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load frozen fusion model and modality encoders
    log.info("monthly_embeddings.loading_fusion_model", path=str(fusion_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not fusion_model_path.exists():
        log.warning(
            "monthly_embeddings.model_not_found",
            path=str(fusion_model_path),
            message="Fusion model checkpoint not found. Cannot generate embeddings.",
        )
        raise FileNotFoundError(f"Fusion model not found at {fusion_model_path}")

    # Load modality encoders
    from fusion.train import load_encoders
    modality_encoders = load_encoders(device=device)

    # Load fusion model
    n_modalities = len(modality_encoders)
    fusion_model = LateFusionMetaLearner(n_modalities=n_modalities)
    fusion_model.load_state_dict(
        torch.load(fusion_model_path, map_location=device, weights_only=True)
    )
    fusion_model.eval()
    fusion_model.to(device)

    # Attach modality encoders to fusion model for easy access
    fusion_model.modality_encoders = modality_encoders
    fusion_model.available_modalities = set(modality_encoders.keys())

    # Encode each month
    all_embeddings: list[pd.DataFrame] = []
    months_to_encode = list(range(1, n_months + 1))

    for month in months_to_encode:
        month_df = _encode_month(month, data_dir, fusion_model, device)
        all_embeddings.append(month_df)

    # Concatenate all months
    combined_df = pd.concat(all_embeddings, ignore_index=True)

    # Validate output
    assert "participant_id" in combined_df.columns
    assert "month" in combined_df.columns
    assert "cdt" in combined_df.columns
    embedding_lengths = combined_df["cdt"].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )
    assert (embedding_lengths == EMBEDDING_DIM).all()

    # Write to parquet
    combined_df.to_parquet(output_path, index=False)
    log.info(
        "monthly_embeddings.complete",
        output_path=str(output_path),
        n_participants=combined_df["participant_id"].nunique(),
        n_months=len(months_to_encode),
        n_rows=len(combined_df),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate monthly CDT embeddings")
    parser.add_argument(
        "--n-months", type=int, default=12, help="Number of months to encode"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/synthetic", help="Data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="applications/_cache/cdt_embeddings_monthly.parquet",
        help="Output parquet file path (must be within applications/_cache/)",
    )
    parser.add_argument(
        "--fusion-model", type=str, default="models/fusion_metalearner.pt"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    args = parser.parse_args()

    generate_monthly_embeddings(
        n_months=args.n_months,
        data_dir=args.data_dir,
        output_path=args.output,
        fusion_model_path=args.fusion_model,
        force=args.force,
    )
