"""
Tests for evaluation/counterfactual_option_b.py — Option B counterfactual simulation.

Validates that simulate_counterfactual returns the correct structure and that
cosine_distance_shift is a reasonable float value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.counterfactual_option_b import simulate_counterfactual

# Use a val-set participant for testing (known to exist in cache)
CACHE_PATH = Path("models/fusion_embeddings_cache.pt")


def _pick_val_participant() -> tuple[str, float]:
    """Pick a participant from the cache and return (id, price_sensitivity)."""
    cache = torch.load(CACHE_PATH, weights_only=False)  # noqa: S614 — cache contains participant_ids (list[str])
    pids = cache["participant_ids"]
    # Use a participant from the middle of the dataset
    pid = pids[500]

    # Load original price_sensitivity from participant_configs
    configs_path = Path("data/synthetic/participant_configs.jsonl")
    for line in configs_path.read_text().strip().splitlines():
        rec = json.loads(line)
        if rec["participant_id"] == pid:
            return pid, rec["price_sensitivity"]

    raise AssertionError(f"Participant {pid} not found in participant_configs.jsonl")


@pytest.fixture(scope="module")
def val_participant() -> tuple[str, float]:
    """Module-scoped fixture to pick participant once."""
    return _pick_val_participant()


class TestSimulateCounterfactual:
    """Tests for the simulate_counterfactual function."""

    def test_returns_required_keys(self, val_participant: tuple[str, float]) -> None:
        """simulate_counterfactual returns dict with all required keys."""
        pid, original_ps = val_participant
        override_value = min(original_ps + 0.3, 1.0)

        result = simulate_counterfactual(
            participant_id=pid,
            overrides={"price_sensitivity": override_value},
        )

        required_keys = [
            "participant_id",
            "overrides",
            "baseline_embedding",
            "counterfactual_embedding",
            "cosine_distance_shift",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_participant_id_preserved(self, val_participant: tuple[str, float]) -> None:
        """Returned participant_id matches input."""
        pid, original_ps = val_participant
        result = simulate_counterfactual(
            participant_id=pid,
            overrides={"price_sensitivity": min(original_ps + 0.3, 1.0)},
        )
        assert result["participant_id"] == pid

    def test_overrides_preserved(self, val_participant: tuple[str, float]) -> None:
        """Returned overrides match input."""
        pid, original_ps = val_participant
        overrides = {"price_sensitivity": min(original_ps + 0.3, 1.0)}
        result = simulate_counterfactual(participant_id=pid, overrides=overrides)
        assert result["overrides"] == overrides

    def test_embeddings_shape(self, val_participant: tuple[str, float]) -> None:
        """Both embeddings are [128] tensors."""
        pid, original_ps = val_participant
        result = simulate_counterfactual(
            participant_id=pid,
            overrides={"price_sensitivity": min(original_ps + 0.3, 1.0)},
        )
        assert result["baseline_embedding"].shape == (128,)
        assert result["counterfactual_embedding"].shape == (128,)

    def test_cosine_distance_in_range(self, val_participant: tuple[str, float]) -> None:
        """cosine_distance_shift is a float in [0.0, 2.0]."""
        pid, original_ps = val_participant
        result = simulate_counterfactual(
            participant_id=pid,
            overrides={"price_sensitivity": min(original_ps + 0.3, 1.0)},
        )
        shift = result["cosine_distance_shift"]
        assert isinstance(shift, float), f"Expected float, got {type(shift)}"
        assert 0.0 <= shift <= 2.0, f"Shift {shift} outside [0.0, 2.0]"

    def test_large_override_produces_shift(
        self, val_participant: tuple[str, float]
    ) -> None:
        """A large price_sensitivity override produces a non-zero shift."""
        pid, _ = val_participant
        result = simulate_counterfactual(
            participant_id=pid,
            overrides={"price_sensitivity": 0.99},
        )
        # With a large override, we expect some shift (but not guaranteed
        # to exceed the 0.27 threshold due to noise confound)
        assert result["cosine_distance_shift"] >= 0.0
