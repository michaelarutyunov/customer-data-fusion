"""
Generate narratives for all participants in psychographics.jsonl.

Reads existing psychographic data to get participant_id and persona_id,
re-samples the PersonaConfig using the archetype and a deterministic seed,
then generates a narrative for each participant.

Usage:
    PYTHONPATH=. uv run python scripts/generate_missing_narratives.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from generator.persona_sampler import sample_persona
from generator.text_generator import generate_narrative

DATA_DIR = Path("data/synthetic")
PSYCHO_PATH = DATA_DIR / "psychographics.jsonl"
NARR_PATH = DATA_DIR / "narratives.jsonl"


def main() -> int:
    # Load participant info from psychographics
    participants: list[dict] = []
    with open(PSYCHO_PATH) as f:
        for line in f:
            if line.strip():
                participants.append(json.loads(line))

    print(f"Loaded {len(participants)} participants from psychographics.jsonl")

    # Check existing narratives
    existing_pids: set[str] = set()
    if NARR_PATH.exists():
        with open(NARR_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_pids.add(r.get("participant_id", ""))
    print(f"Found {len(existing_pids)} existing narratives")

    to_generate = [p for p in participants if p["participant_id"] not in existing_pids]
    print(f"Generating {len(to_generate)} new narratives...")

    failures = 0
    with open(NARR_PATH, "a") as f:
        for i, p in enumerate(to_generate):
            pid = p["participant_id"]
            archetype_id = p["persona_id"]

            # Derive deterministic seed from participant index within archetype
            # pid format: archetype_NNNN
            idx = int(pid.rsplit("_", 1)[1])
            seed = sum(ord(c) for c in archetype_id) + idx

            try:
                config = sample_persona(archetype_id, random_seed=seed)
                narrative = generate_narrative(
                    config,
                    category=p.get("category", "electronics"),
                    participant_id=pid,
                )
                f.write(
                    json.dumps(
                        {
                            "participant_id": narrative.participant_id,
                            "persona_id": narrative.persona_id,
                            "category": narrative.category,
                            "text": narrative.text,
                            "word_count": narrative.word_count,
                            "model_id": narrative.model_id,
                            "prompt_version": narrative.prompt_version,
                            "embedding": narrative.embedding,
                            "embedding_model_id": narrative.embedding_model_id,
                        }
                    )
                    + "\n"
                )
                f.flush()
            except Exception as e:
                print(f"  FAILED {pid}: {e}")
                failures += 1
                continue

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(to_generate)} generated, {failures} failures")

    print(f"Done: {len(to_generate)} attempted, {failures} failures")
    return 0 if failures <= len(to_generate) * 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
