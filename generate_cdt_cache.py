"""Generate CDT embeddings cache from fusion model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq

from fusion.meta_learner import LateFusionMetaLearner


def generate_cdt_cache(
    fusion_checkpoint: Path = Path("models/fusion_meta_learner.pt"),
    embeddings_cache: Path = Path("models/fusion_embeddings_cache.pt"),
    output_path: Path = Path("applications/_cache/cdt_embeddings.parquet"),
) -> None:
    """Generate CDT embeddings from fusion model and encoder outputs."""
    print(f"Loading fusion model from {fusion_checkpoint}")
    print(f"Loading encoder outputs from {embeddings_cache}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder outputs cache
    cache = torch.load(embeddings_cache, map_location=device, weights_only=True)

    # Load fusion model
    fusion_model = LateFusionMetaLearner(
        n_modalities=6,
        per_modality_dim=128,
        hidden_dim=256,
        embed_dim=128,
        n_classes=7,
        dropout=0.2,
    ).to(device)

    fusion_model.load_state_dict(
        torch.load(fusion_checkpoint, map_location=device, weights_only=True)
    )
    fusion_model.eval()

    # Extract data from cache
    trace_embs = cache["trace"]  # [1000, 128]
    transaction_embs = cache["transaction"]  # [1000, 128]
    text_embs = cache["text"]  # [1000, 128]
    psychographic_embs = cache["psychographic"]  # [1000, 128]
    clickstream_embs = cache["clickstream"]  # [1000, 128]
    campaign_embs = cache["campaign"]  # [1000, 128]
    participant_ids = cache["participant_ids"]  # list of 1000 strings
    labels = cache["labels"]  # [1000]

    print(f"Found {len(participant_ids)} participants")

    # Stack modality embeddings: [1000, 6, 128]
    modality_embs = torch.stack([
        trace_embs,
        transaction_embs,
        text_embs,
        psychographic_embs,
        clickstream_embs,
        campaign_embs,
    ], dim=1)  # [1000, 6, 128]

    # Normalize each modality
    norm_embs = []
    for i in range(6):
        norm_embs.append(F.normalize(modality_embs[:, i], p=2, dim=-1))

    # Concatenate: [1000, 6, 128] -> [1000, 768]
    fusion_input = torch.cat(norm_embs, dim=-1)

    # Generate CDT embeddings
    print("Generating CDT embeddings...")
    cdt_embeddings = []

    with torch.no_grad():
        for i in range(len(participant_ids)):
            participant_input = fusion_input[i:i+1]  # [1, 768]
            _, cdt_embedding = fusion_model.forward_with_embedding(participant_input)
            embedding = cdt_embedding.squeeze(0).cpu().numpy()  # [128]

            cdt_embeddings.append({
                "participant_id": participant_ids[i],
                "cdt": embedding.tolist(),
            })

    # Save to parquet
    df = pd.DataFrame(cdt_embeddings)
    print(f"Generated {len(df)} CDT embeddings")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")
    print("✅ CDT cache saved successfully")


if __name__ == "__main__":
    generate_cdt_cache()
