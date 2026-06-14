"""
Generate narratives for all participants in psychographics.jsonl — with exact
PersonaConfig reproduction, retry/backoff, and output validation.

Consistency (the fix): reads the canonical ``random_seed`` the pipeline wrote
into ``participant_configs.jsonl`` and reproduces the EXACT ``PersonaConfig``
via ``sample_persona(archetype, random_seed=...)``. The narrative's z-derived
style description therefore matches the participant's actual generated traces /
transactions / psychographics / clickstream / campaigns — preserving the
project's cross-modal-consistency invariant. (Previously the script re-sampled
with a derived seed ≠ the pipeline's, so narratives could contradict behaviour.)

Robustness: each narrative is retried with exponential backoff on API errors,
and validated (non-empty text, word_count within bounds) before writing. A
silent empty/garbage API response is rejected and retried rather than written.

Resumable: skips participant_ids already present, appends, and flushes after
every write — safe to interrupt and re-run.

Usage:
    PYTHONPATH=. uv run python scripts/generate_missing_narratives.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from generator.persona_sampler import sample_persona
from generator.text_generator import generate_narrative
from schemas.text import PersonaNarrative

DATA_DIR = Path("data/synthetic")
PSYCHO_PATH = DATA_DIR / "psychographics.jsonl"
CONFIGS_PATH = DATA_DIR / "participant_configs.jsonl"
NARR_PATH = DATA_DIR / "narratives.jsonl"

# Output validation. Prompt targets 280–320 words; floor catches silent
# empty/garbage API responses, ceiling catches run-on output.
MIN_WORDS = 120
MAX_WORDS = 500
MAX_RETRIES = 4
BACKOFF_BASE_S = 2.0  # slept backoff = BACKOFF_BASE_S * 2**attempt + jitter


def _load_seed_index(path: Path) -> dict[str, int]:
    """participant_id -> random_seed, read from participant_configs.jsonl.

    The seed is constant across a participant's monthly rows, so the month-1
    row is authoritative. Returns {} for legacy files written before the seed
    was stored (caller handles by skipping).
    """
    seeds: dict[str, int] = {}
    if not path.exists():
        return seeds
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("month", 1) != 1:
                continue
            if "random_seed" in rec:
                seeds[rec["participant_id"]] = int(rec["random_seed"])
    return seeds


def _is_valid(narr: PersonaNarrative) -> bool:
    return bool(narr.text and narr.text.strip()) and (
        MIN_WORDS <= narr.word_count <= MAX_WORDS
    )


def _generate_with_retry(config, category: str, pid: str) -> PersonaNarrative | None:
    """Generate one narrative with exponential backoff + validation.

    Retries on any exception (API timeout / rate-limit / 5xx) and on
    validation failure (empty text or out-of-range word count). Returns None
    if all attempts are exhausted.
    """
    last_reason = "no attempt"
    for attempt in range(MAX_RETRIES):
        try:
            narr = generate_narrative(config, category=category, participant_id=pid)
            if _is_valid(narr):
                return narr
            last_reason = f"invalid word_count={narr.word_count}"
        except Exception as exc:  # noqa: BLE001 — broad retry for API flakiness
            last_reason = f"{type(exc).__name__}: {exc}"
        print(
            f"  retry {attempt + 1}/{MAX_RETRIES} {pid}: {last_reason}",
            file=sys.stderr,
        )
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE_S * (2**attempt) + random.uniform(0, 1))
    print(
        f"  GAVE UP {pid} after {MAX_RETRIES} attempts: {last_reason}", file=sys.stderr
    )
    return None


def _dump_narrative(narr: PersonaNarrative) -> str:
    return json.dumps(
        {
            "participant_id": narr.participant_id,
            "persona_id": narr.persona_id,
            "category": narr.category,
            "text": narr.text,
            "word_count": narr.word_count,
            "model_id": narr.model_id,
            "prompt_version": narr.prompt_version,
            "embedding": narr.embedding,
            "embedding_model_id": narr.embedding_model_id,
        }
    )


def main() -> int:
    # psychographics.jsonl carries 2 rows/participant (months 1 & 7); dedupe.
    seen: set[str] = set()
    participants: list[dict] = []
    for line in PSYCHO_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec["participant_id"]
        if pid not in seen:
            seen.add(pid)
            participants.append(rec)
    print(f"Loaded {len(participants)} unique participants from psychographics.jsonl")

    seeds = _load_seed_index(CONFIGS_PATH)
    if seeds:
        print(
            f"Loaded random_seed for {len(seeds)} participants from participant_configs.jsonl"
        )
    else:
        print(
            "WARNING: no random_seed in participant_configs.jsonl. Regenerate the "
            "dataset (pipeline now writes random_seed) so narratives match behaviour."
        )

    existing: set[str] = set()
    if NARR_PATH.exists():
        for line in NARR_PATH.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line).get("participant_id", ""))
    print(f"Found {len(existing)} existing narratives")

    to_generate = [p for p in participants if p["participant_id"] not in existing]
    print(f"Generating {len(to_generate)} new narratives...")

    failures = 0
    skipped = 0
    written = 0
    with open(NARR_PATH, "a") as f:
        for p in to_generate:
            pid = p["participant_id"]
            archetype_id = p["persona_id"]
            seed = seeds.get(pid)
            if seed is None:
                # Cannot reproduce the exact config — skip rather than write an
                # inconsistent narrative that contradicts the generated behaviour.
                print(
                    f"  SKIP {pid}: no random_seed (regenerate data first)",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            config = sample_persona(archetype_id, random_seed=seed)
            narr = _generate_with_retry(
                config, category=p.get("category", "electronics"), pid=pid
            )
            if narr is None:
                failures += 1
                continue

            f.write(_dump_narrative(narr) + "\n")
            f.flush()
            written += 1

            if (written) % 10 == 0:
                print(
                    f"  {written}/{len(to_generate)} written · "
                    f"{failures} failures · {skipped} skipped"
                )

    print(
        f"Done: {written} written, {failures} failures, {skipped} skipped "
        f"(of {len(to_generate)} pending)"
    )
    # Non-zero if anything pending was not written (failure or missing-seed skip).
    return 0 if (failures == 0 and skipped == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
