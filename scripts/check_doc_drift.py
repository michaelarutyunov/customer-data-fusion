#!/usr/bin/env python3
"""
check_doc_drift.py — verify documentation cross-references.

Catches the most common doc drift: paths referenced in CLAUDE.md, the Tier 3
context docs (.claude/context/*.md), and the agent docs (.claude/agents/**/AGENT.md)
that no longer exist on disk (e.g. a referenced post-mortem, spec, or script that
was moved or deleted).

Usage:
    uv run python scripts/check_doc_drift.py
    uv run python scripts/check_doc_drift.py --root /path/to/repo

Exit code 0 = no drift, 1 = broken references found.

Scope is deliberately narrow (path existence). It does NOT check semantic
correctness of the docs — only that referenced files are present.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Docs to scan for referenced paths.
DOC_GLOBS = [
    "CLAUDE.md",
    ".claude/context/*.md",
    ".claude/agents/**/AGENT.md",
]

# A backtick-quoted token that looks like a repo-relative path: contains a path
# separator or a known extension, has no spaces, and isn't a URL.
_PATH_RE = re.compile(
    r"`([^`*\s]+\.(?:md|py|yaml|yml|json|jsonl|txt)|[^`*\s]+/[^`*\s]+)`"
)


def collect_doc_files(root: Path) -> list[Path]:
    # Forward-looking docs that legitimately reference not-yet-built paths
    # (the applications layer, the planned choice model / product catalogue).
    # Excluded so the tool's output reflects real drift, not aspirational prose.
    exclude = {
        root / ".claude/context/new-capabilities.md",
        root / ".claude/agents/applications-specialist/AGENT.md",
    }
    files: list[Path] = []
    for glob in DOC_GLOBS:
        files.extend(root.glob(glob))
    return sorted({f for f in set(files) if f not in exclude})


def extract_references(doc_path: Path) -> set[str]:
    refs: set[str] = set()
    text = doc_path.read_text(errors="replace")
    for match in _PATH_RE.finditer(text):
        token = match.group(1)
        # Skip URLs and obvious non-paths.
        if token.startswith(("http://", "https://", "ftp://")):
            continue
        # Skip pip-ish / module-ish tokens like "schemas.EMBEDDING_DIM".
        if "." in token and "/" not in token:
            continue
        # Skip templates/placeholders ({id}, <short-description>) and symbol
        # refs (module.py::Symbol) and git refs (feature/..., fix/...) — these
        # are not repo-relative file paths.
        if any(ch in token for ch in "{}<>") or "::" in token:
            continue
        if token.startswith(("feature/", "fix/", "bugfix/", "hotfix/", "refactor/")):
            continue
        refs.add(token)
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="Repo root (default: cwd)"
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()

    docs = collect_doc_files(root)
    if not docs:
        print(f"No docs found under {root}", file=sys.stderr)
        return 1

    broken: list[tuple[Path, str]] = []
    checked = 0
    for doc in docs:
        for ref in extract_references(doc):
            # Resolve relative to repo root; ignore refs that escape the repo.
            target = (root / ref).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                continue
            checked += 1
            if not target.exists():
                broken.append((doc, ref))

    print(f"Scanned {len(docs)} doc files, checked {checked} path references.")
    if broken:
        print(f"\nBROKEN REFERENCES ({len(broken)}):")
        for doc, ref in sorted(set(broken)):
            print(f"  {doc.relative_to(root)}: `{ref}`")
        return 1
    print("No drift: all referenced paths exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
