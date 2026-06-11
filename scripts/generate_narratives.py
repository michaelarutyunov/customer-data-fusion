"""
Generate narratives.jsonl for all 1000 participants.

Loads persona configs, calls generate_narratives_batch(), and writes
results to data/synthetic/narratives.jsonl.
"""

import dataclasses
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path for absolute imports
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv

load_dotenv()

from generator.persona_sampler import list_archetype_ids, sample_persona
from generator.text_generator import generate_narratives_batch

N_PARTICIPANTS = 1000
OUTPUT_PATH = Path(_REPO_ROOT) / "data" / "synthetic" / "narratives.jsonl"


def main() -> None:
    output_dir = OUTPUT_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    archetypes = list_archetype_ids()
    print(f"Archetypes: {archetypes}")
    print(f"Generating narratives for {N_PARTICIPANTS} participants...")

    # Sample configs for all 1000 participants (cycling archetypes)
    configs = []
    for i in range(N_PARTICIPANTS):
        archetype_id = archetypes[i % len(archetypes)]
        config = sample_persona(archetype_id, random_seed=i)
        configs.append(config)

    # Generate narratives via LLM
    narratives = generate_narratives_batch(configs, category="electronics")

    # Write JSONL
    with open(OUTPUT_PATH, "w") as f:
        for narrative in narratives:
            row = dataclasses.asdict(narrative)
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(narratives)} narratives to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
