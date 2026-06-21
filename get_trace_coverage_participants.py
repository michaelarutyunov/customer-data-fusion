"""Extract participant IDs with trace coverage for M1 training."""

from __future__ import annotations

import json
from pathlib import Path


def get_trace_coverage_participants(
    trials_path: Path = Path("data/synthetic/trials.jsonl"),
    output_path: Path = Path("applications/choice/trace_coverage_participants.txt"),
) -> None:
    """Extract participant IDs that have trace coverage."""
    print(f"Loading trials from {trials_path}")

    participants_with_trace = set()

    with open(trials_path) as f:
        for line in f:
            trial = json.loads(line)
            # Check if trial has trace coverage (has acquisitions)
            if trial.get("total_acquisitions", 0) > 0:
                participants_with_trace.add(trial["participant_id"])

    print(f"Found {len(participants_with_trace)} participants with trace coverage")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for participant_id in sorted(participants_with_trace):
            f.write(f"{participant_id}\n")

    print(f"✅ Saved participant list to {output_path}")


if __name__ == "__main__":
    get_trace_coverage_participants()
