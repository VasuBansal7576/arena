#!/usr/bin/env python
"""Program-driven AutoResearch loop for the ETA submission.

This is intentionally closer to Karpathy's AutoResearch shape than the small
ablation harness:

1. read the human research brief in program.md
2. inspect the current best metrics and research ledger
3. choose the next unrun experiment from a code-backed queue
4. run it through autoresearch.py with promotion enabled
5. append a compact research note explaining keep/reject

The actual code edits still happen in this repo, not inside this script. The
point of this runner is to make the loop repeatable and auditable: every
hypothesis maps to a named experiment, every experiment maps to train.py flags
or code already checked in, and promotion is metric-gated.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
PROGRAM_PATH = ROOT / "program.md"
METRICS_PATH = ROOT / "metrics.json"
LEDGER_PATH = ROOT / "research_log.csv"
NOTES_PATH = ROOT / "research_notes.md"


@dataclass(frozen=True)
class ResearchItem:
    name: str
    hypothesis: str
    risk: str


RESEARCH_QUEUE = [
    ResearchItem(
        "squared_error_no_cap_1m_500_lr04",
        "A slower 500-iteration squared-error learner should reduce underfit without changing the inference contract.",
        "May overfit noisy late-2023 patterns if pushed further.",
    ),
    ResearchItem(
        "squared_error_no_cap_1m_420_lr045",
        "Lower learning rate with more boosting rounds may improve the existing best model.",
        "Small gains can disappear on hidden Eval if they only fit Dev noise.",
    ),
    ResearchItem(
        "squared_error_no_cap_1m_420_leaf95",
        "A wider tree can capture airport and outer-borough interactions that the current leaf budget misses.",
        "Higher capacity may overfit sparse route pairs.",
    ),
    ResearchItem(
        "squared_error_no_cap_1m_340_seed7",
        "Changing the recency-weighted sample can reveal whether the current winner is sample-lucky.",
        "If it loses, the original sample is likely genuinely stronger.",
    ),
    ResearchItem(
        "squared_error_no_cap_1500k_420_lr045",
        "More sampled rows may help the slower learner see rare airport and outer-borough routes.",
        "More rows may pull in older/noisier regimes and worsen 2024 drift.",
    ),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def completed() -> dict[str, dict]:
    if not LEDGER_PATH.exists():
        return {}
    with open(LEDGER_PATH, newline="") as f:
        return {row["experiment"]: row for row in csv.DictReader(f)}


def choose_next() -> ResearchItem | None:
    done = completed()
    for item in RESEARCH_QUEUE:
        if item.name not in done:
            return item
    return None


def append_note(item: ResearchItem, before: float, after: float, row: dict | None) -> None:
    promoted = bool(row and row.get("promoted") == "True")
    status = "PROMOTED" if promoted else "rejected"
    NOTES_PATH.write_text(
        (
            NOTES_PATH.read_text()
            if NOTES_PATH.exists()
            else "# Research Notes\n\n"
        )
        + f"## {item.name}\n\n"
        + f"- Hypothesis: {item.hypothesis}\n"
        + f"- Risk: {item.risk}\n"
        + f"- Best before: {before:.6f}\n"
        + f"- Best after: {after:.6f}\n"
        + f"- Decision: {status}\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run a specific queued experiment")
    parser.add_argument("--dry-run", action="store_true", help="print the next experiment without running it")
    args = parser.parse_args()

    if not PROGRAM_PATH.exists():
        raise SystemExit("Missing program.md. AutoResearch needs a human research brief.")

    item = next((x for x in RESEARCH_QUEUE if x.name == args.only), None) if args.only else choose_next()
    if item is None:
        print("No queued experiments left. Add a ResearchItem after reading program.md and metrics.json.")
        return

    before = float(load_json(METRICS_PATH).get("dev_mae", float("inf")))
    print(PROGRAM_PATH.read_text().splitlines()[0])
    print(f"Next experiment: {item.name}")
    print(f"Hypothesis: {item.hypothesis}")
    print(f"Risk: {item.risk}")
    print(f"Current best: {before:.6f}")

    if args.dry_run:
        return

    subprocess.run(
        [sys.executable, "autoresearch.py", "--promote", "--only", item.name],
        cwd=ROOT,
        check=True,
    )
    after = float(load_json(METRICS_PATH).get("dev_mae", float("inf")))
    row = completed().get(item.name)
    append_note(item, before, after, row)
    print(f"Best after: {after:.6f}")
    print(f"Notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
