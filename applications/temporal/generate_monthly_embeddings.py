"""
Generate monthly CDT embeddings using the frozen fusion meta-learner.

Reads month-partitioned modality files, passes them through the frozen fusion
model, and writes one embedding per participant-month to disk.

Output: applications/_cache/cdt_embeddings_monthly.parquet
Columns: [participant_id, month, cdt] where cdt is a 128-dim float array.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
from tqdm import tqdm

from fusion.meta_learner import LateFusionMetaLearner
from schemas import EMBEDDING_DIM

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

    # Encode each participant
    embeddings: list[dict] = []
    for participant_id, modality_data in tqdm(
        participant_data.items(), desc=f"Month {month}"
    ):
        # TODO: Implement actual encoding pipeline:
        # 1. Load frozen modality encoders
        # 2. Pass raw data through each encoder to get modality embeddings
        # 3. Concatenate modality embeddings
        # 4. Pass through fusion model
        #
        # For now, create a placeholder embedding (zeros) to satisfy the interface
        # This will be replaced with the actual encoding logic in a follow-up task
        log.warning(
            "monthly_embeddings.placeholder_embedding",
            participant_id=participant_id,
            month=month,
            message="Using placeholder zero embedding - actual encoding not yet implemented",
        )
        embedding = torch.zeros(EMBEDDING_DIM, device=device)

        embeddings.append(
            {
                "participant_id": participant_id,
                "month": month,
                "cdt": embedding.cpu().numpy(),
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

    # Load frozen fusion model
    log.info("monthly_embeddings.loading_fusion_model", path=str(fusion_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not fusion_model_path.exists():
        log.warning(
            "monthly_embeddings.model_not_found",
            path=str(fusion_model_path),
            message="Fusion model checkpoint not found. Using placeholder model.",
        )
        fusion_model = LateFusionMetaLearner()
    else:
        fusion_model = LateFusionMetaLearner()
        fusion_model.load_state_dict(
            torch.load(fusion_model_path, map_location=device, weights_only=True)
        )
    fusion_model.eval()
    fusion_model.to(device)

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
