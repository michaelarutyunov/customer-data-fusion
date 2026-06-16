"""Tests for monthly CDT embedding generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.temporal.generate_monthly_embeddings import (
    _load_monthly_modality,
    generate_monthly_embeddings,
)


@pytest.fixture
def sample_monthly_data(tmp_path):
    """Create sample monthly modality files for testing."""
    data_dir = tmp_path / "data" / "synthetic"
    data_dir.mkdir(parents=True)

    # Create month 1-3 transaction files
    for month in [1, 2, 3]:
        tx_file = data_dir / f"transactions_month_{month:02d}.jsonl"
        tx_file.write_text(
            json.dumps(
                {
                    "participant_id": "test_participant",
                    "month": month,
                    "transaction_id": f"tx_{month}",
                    "category": "electronics",
                    "product_id": "prod_01",
                    "brand_tier": "mid",
                    "price_paid_normalised": 0.5,
                    "quantity": 1,
                }
            )
            + "\n"
        )

    # Create month 1-3 trace files
    for month in [1, 2, 3]:
        trace_file = data_dir / f"traces_month_{month:02d}.jsonl"
        trace_file.write_text(
            json.dumps(
                {
                    "participant_id": "test_participant",
                    "month": month,
                    "session_id": f"session_{month}",
                    "trial_id": f"trial_{month}",
                    "n_alternatives": 3,
                    "final_choice": "A",
                }
            )
            + "\n"
        )

    return data_dir


def test_load_monthly_modality(sample_monthly_data):
    """Test loading month-partitioned modality files."""
    records = _load_monthly_modality(
        "transactions", months=[1, 2, 3], data_dir=sample_monthly_data
    )

    assert len(records) == 3
    assert records[0]["participant_id"] == "test_participant"
    assert records[0]["month"] == 1
    assert records[1]["month"] == 2


def test_load_monthly_modality_invalid_name(sample_monthly_data):
    """Test that invalid modality names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid modality"):
        _load_monthly_modality(
            "not_a_modality", months=[1], data_dir=sample_monthly_data
        )


@pytest.mark.parametrize("n_months", [1, 3, 12])
def test_generate_monthly_embeddings_integration(
    n_months, tmp_path, sample_monthly_data
):
    """Integration test for monthly embedding generation (requires frozen fusion model)."""
    # This test requires the actual frozen fusion model
    # It will be skipped if the model doesn't exist
    fusion_path = Path("models/fusion_metalearner.pt")
    if not fusion_path.exists():
        pytest.skip(f"Frozen fusion model not found at {fusion_path}")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    output_path = cache_dir / "cdt_embeddings_monthly_test.parquet"

    # Run generation (only for months 1-3 to keep test fast)
    generate_monthly_embeddings(
        n_months=3,
        data_dir=sample_monthly_data,
        output_path=output_path,
        fusion_model_path=fusion_path,
    )

    # Verify output exists and has correct structure
    assert output_path.exists()

    df = pd.read_parquet(output_path)
    assert "participant_id" in df.columns
    assert "month" in df.columns
    assert "cdt" in df.columns

    # Verify we have embeddings for all months
    participant_rows = df[df["participant_id"] == "test_participant"]
    assert len(participant_rows) == 3
    assert set(participant_rows["month"]) == {1, 2, 3}

    # Verify embedding dimension
    embedding = participant_rows.iloc[0]["cdt"]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (128,)  # EMBEDDING_DIM from schemas
    assert np.all(np.isfinite(embedding))  # No NaN/Inf
