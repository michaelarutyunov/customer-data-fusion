"""Unit tests for generator/pipeline.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generator.pipeline import run_pipeline, _to_json
from schemas.persona import Strategy


# ---------------------------------------------------------------------------
# _to_json helper
# ---------------------------------------------------------------------------


class TestToJson:
    def test_serialises_enum_as_value(self):
        import dataclasses

        @dataclasses.dataclass
        class Stub:
            s: Strategy

        obj = Stub(s=Strategy.LEXICOGRAPHIC)
        row = json.loads(_to_json(obj))
        assert row["s"] == "lexicographic"

    def test_produces_valid_json(self):
        import dataclasses

        @dataclasses.dataclass
        class Tiny:
            x: int

        row = _to_json(Tiny(x=42))
        assert json.loads(row) == {"x": 42}


# ---------------------------------------------------------------------------
# run_pipeline — basic output
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_produces_jsonl_files(self, tmp_path: Path):
        counts = run_pipeline(n=3, output_dir=tmp_path, skip_narratives=True)

        for stem in ["traces", "trials", "transactions", "psychographics"]:
            assert (tmp_path / f"{stem}.jsonl").exists(), f"{stem}.jsonl missing"
        assert counts["narratives"] == 0

    def test_psychographics_count_equals_n(self, tmp_path: Path):
        n = 5
        counts = run_pipeline(n=n, output_dir=tmp_path, skip_narratives=True)
        # Psychographics fielded at months 1 and 7 (snapshot modality) → 2 per participant
        assert counts["psychographics"] == 2 * n

    def test_trials_count_matches_coverage_subset(self, tmp_path: Path):
        n, n_trials = 4, 10
        # All n participants in coverage subset (trace_coverage >= n)
        counts = run_pipeline(
            n=n,
            n_trials=n_trials,
            output_dir=tmp_path,
            skip_narratives=True,
            trace_coverage=n,
        )
        # Traces fielded at months 1 and 2 → 2 fielding rounds per participant
        assert counts["trials"] == n * n_trials * 2

    def test_traces_count_positive(self, tmp_path: Path):
        counts = run_pipeline(n=2, output_dir=tmp_path, skip_narratives=True)
        assert counts["traces"] > 0

    def test_transactions_count_positive(self, tmp_path: Path):
        counts = run_pipeline(n=2, output_dir=tmp_path, skip_narratives=True)
        assert counts["transactions"] > 0

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b"
        run_pipeline(n=1, output_dir=nested, skip_narratives=True)
        assert nested.exists()

    def test_single_archetype_cycles(self, tmp_path: Path):
        run_pipeline(
            n=3, archetypes=["price_lex"], output_dir=tmp_path, skip_narratives=True
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "psychographics.jsonl").read_text().splitlines()
        ]
        assert all(r["persona_id"] == "price_lex" for r in rows)

    def test_archetype_cycling(self, tmp_path: Path):
        run_pipeline(n=7, output_dir=tmp_path, skip_narratives=True)
        rows = [
            json.loads(line)
            for line in (tmp_path / "psychographics.jsonl").read_text().splitlines()
        ]
        seen = {r["persona_id"] for r in rows}
        # 7 participants cycling all 7 archetypes → all 7 distinct persona_ids
        assert len(seen) == 7

    def test_returns_dict_with_all_modalities(self, tmp_path: Path):
        counts = run_pipeline(n=2, output_dir=tmp_path, skip_narratives=True)
        assert set(counts.keys()) == {
            "traces",
            "trials",
            "transactions",
            "psychographics",
            "narratives",
            "narrative_failures",
            "participant_configs",
            "clickstream",
            "campaigns",
        }


# ---------------------------------------------------------------------------
# Cross-modal consistency
# ---------------------------------------------------------------------------


class TestCrossModalConsistency:
    def test_persona_id_consistent_across_modalities(self, tmp_path: Path):
        run_pipeline(
            n=3, archetypes=["price_lex"], output_dir=tmp_path, skip_narratives=True
        )

        psycho_ids = {
            json.loads(line)["persona_id"]
            for line in (tmp_path / "psychographics.jsonl").read_text().splitlines()
        }
        trial_ids = {
            json.loads(line)["persona_id"]
            for line in (tmp_path / "trials.jsonl").read_text().splitlines()
        }
        tx_ids = {
            json.loads(line)["persona_id"]
            for line in (tmp_path / "transactions.jsonl").read_text().splitlines()
        }

        # All modalities reference the same persona_id
        assert psycho_ids == {"price_lex"}
        assert trial_ids == {"price_lex"}
        assert tx_ids == {"price_lex"}

    def test_trace_participant_id_matches_psychographic(self, tmp_path: Path):
        run_pipeline(
            n=2, archetypes=["compensatory"], output_dir=tmp_path, skip_narratives=True
        )

        trace_pids = {
            json.loads(line)["participant_id"]
            for line in (tmp_path / "traces.jsonl").read_text().splitlines()
        }
        psycho_pids = {
            json.loads(line)["participant_id"]
            for line in (tmp_path / "psychographics.jsonl").read_text().splitlines()
        }

        assert trace_pids == psycho_pids

    def test_clickstream_campaign_participant_id_matches_psychographic(
        self, tmp_path: Path
    ):
        run_pipeline(
            n=2, archetypes=["compensatory"], output_dir=tmp_path, skip_narratives=True
        )

        psycho_pids = {
            json.loads(line)["participant_id"]
            for line in (tmp_path / "psychographics.jsonl").read_text().splitlines()
        }
        # Non-anonymous clickstream events/summaries carry the participant id;
        # anonymous sessions carry participant_id="" and are excluded here.
        click_pids = {
            rec["participant_id"]
            for line in (tmp_path / "clickstream.jsonl").read_text().splitlines()
            for rec in [json.loads(line)]
            if rec["participant_id"]
        }
        camp_pids = {
            json.loads(line)["participant_id"]
            for line in (tmp_path / "campaigns.jsonl").read_text().splitlines()
        }

        # Both new modalities are individually attributable — same id set as
        # psychographics (the canonical per-participant ordering).
        assert click_pids == psycho_pids
        assert camp_pids == psycho_pids


# ---------------------------------------------------------------------------
# JSONL format
# ---------------------------------------------------------------------------


class TestJsonlFormat:
    def test_each_line_is_valid_json(self, tmp_path: Path):
        run_pipeline(n=2, output_dir=tmp_path, skip_narratives=True)

        for stem in ["traces", "trials", "transactions", "psychographics"]:
            path = tmp_path / f"{stem}.jsonl"
            for i, line in enumerate(path.read_text().splitlines()):
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    pytest.fail(f"{stem}.jsonl line {i} is not valid JSON: {exc}")

    def test_psychographic_has_required_fields(self, tmp_path: Path):
        run_pipeline(n=1, output_dir=tmp_path, skip_narratives=True)
        # Psychographics now fielded at months 1 and 7 → multiple lines; read first
        row = json.loads(
            (tmp_path / "psychographics.jsonl").read_text().splitlines()[0]
        )
        for field in [
            "participant_id",
            "persona_id",
            "price_consciousness",
            "brand_sensitivity",
        ]:
            assert field in row, f"Missing field: {field}"

    def test_trial_has_required_fields(self, tmp_path: Path):
        run_pipeline(n=1, n_trials=2, output_dir=tmp_path, skip_narratives=True)
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[0])
        for field in [
            "participant_id",
            "trial_id",
            "payne_index",
            "prop_cells_inspected",
        ]:
            assert field in row, f"Missing field: {field}"

    def test_trace_has_required_fields(self, tmp_path: Path):
        run_pipeline(n=1, n_trials=2, output_dir=tmp_path, skip_narratives=True)
        row = json.loads((tmp_path / "traces.jsonl").read_text().splitlines()[0])
        for field in [
            "participant_id",
            "trial_id",
            "alternative_id",
            "attribute_id",
            "dwell_ms",
        ]:
            assert field in row, f"Missing field: {field}"

    def test_enums_serialised_as_strings(self, tmp_path: Path):
        run_pipeline(n=1, output_dir=tmp_path, skip_narratives=True)
        row = json.loads((tmp_path / "transactions.jsonl").read_text().splitlines()[0])
        # channel and purchase_type are Enums — should be strings in JSON
        assert isinstance(row["channel"], str)
        assert isinstance(row["purchase_type"], str)


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_same_psychographics(self, tmp_path: Path):
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"

        run_pipeline(n=3, base_seed=99, output_dir=dir_a, skip_narratives=True)
        run_pipeline(n=3, base_seed=99, output_dir=dir_b, skip_narratives=True)

        # Psychographics are fully seeded — byte-identical
        a = (dir_a / "psychographics.jsonl").read_text()
        b = (dir_b / "psychographics.jsonl").read_text()
        assert a == b, "psychographics.jsonl differs between identical-seed runs"

    def test_same_seed_produces_same_trial_stats(self, tmp_path: Path):
        # session_id is uuid4 (OS-random), so trial_id differs between runs.
        # The meaningful fields — payne_index, prop_cells, n_acquisitions — must match.
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"

        run_pipeline(n=2, base_seed=42, output_dir=dir_a, skip_narratives=True)
        run_pipeline(n=2, base_seed=42, output_dir=dir_b, skip_narratives=True)

        _SKIP = {"trial_id", "session_id"}

        def _strip(line: str) -> dict:
            return {k: v for k, v in json.loads(line).items() if k not in _SKIP}

        for stem in ["trials", "transactions"]:
            stripped_a = [
                _strip(line)
                for line in (dir_a / f"{stem}.jsonl").read_text().splitlines()
            ]
            stripped_b = [
                _strip(line)
                for line in (dir_b / f"{stem}.jsonl").read_text().splitlines()
            ]
            assert stripped_a == stripped_b, (
                f"{stem}.jsonl data differs between identical-seed runs"
            )


# ---------------------------------------------------------------------------
# Narrative resilience
# ---------------------------------------------------------------------------


class TestNarrativeResilience:
    def test_pipeline_continues_when_narrative_raises(self, tmp_path: Path):
        """A single narrative API failure must not abort the run."""
        from unittest.mock import patch

        with patch(
            "generator.pipeline.generate_narrative",
            side_effect=RuntimeError("API error"),
        ):
            counts = run_pipeline(n=3, output_dir=tmp_path)

        # All non-narrative modalities must be complete
        # Psychographics fielded at months 1 and 7 → 2 per participant
        assert counts["psychographics"] == 2 * 3
        assert counts["traces"] > 0
        assert counts["transactions"] > 0
        # No narratives written — every call failed
        assert counts["narratives"] == 0

    def test_partial_narrative_failure_writes_successful_ones(self, tmp_path: Path):
        """Participants whose narrative succeeds must still be written."""
        from unittest.mock import patch
        from schemas.text import PersonaNarrative

        call_count = 0

        def flaky_generate(config, category="electronics", participant_id=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("transient API error")
            pid = participant_id if participant_id is not None else config.persona_id
            return PersonaNarrative(
                participant_id=pid,
                persona_id=config.persona_id,
                category=category,
                text="word " * 280,
                word_count=280,
                model_id="deepseek-chat",
                prompt_version="v1",
            )

        with patch("generator.pipeline.generate_narrative", side_effect=flaky_generate):
            counts = run_pipeline(n=3, output_dir=tmp_path)

        # 2 of 3 narratives succeeded
        assert counts["narratives"] == 2
        # Other modalities unaffected (psychographics at months 1 and 7 → 2n)
        assert counts["psychographics"] == 2 * 3

    def test_narrative_failures_reported_in_counts(self, tmp_path: Path):
        """counts dict must include narrative_failures key with failure count."""
        from unittest.mock import patch

        with patch(
            "generator.pipeline.generate_narrative",
            side_effect=RuntimeError("API error"),
        ):
            counts = run_pipeline(n=4, output_dir=tmp_path)

        assert counts["narrative_failures"] == 4

    def test_non_narrative_exception_still_propagates(self, tmp_path: Path):
        """Only narrative errors are swallowed — other generator errors must crash."""
        from unittest.mock import patch

        with patch(
            "generator.pipeline.generate_psychographic",
            side_effect=RuntimeError("psych bug"),
        ):
            with pytest.raises(RuntimeError, match="psych bug"):
                run_pipeline(n=2, output_dir=tmp_path, skip_narratives=True)
