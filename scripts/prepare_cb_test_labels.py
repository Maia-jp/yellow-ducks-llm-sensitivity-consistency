#!/usr/bin/env python3
"""Add CB test labels from the original public annotator responses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

LABEL_RANGES = {
    "entailment": range(1, 4),
    "neutral": range(0, 1),
    "contradiction": range(-3, 0),
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_items(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    items: dict[tuple[str, str], list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                normalize(f"{row['Context']} {row['Target']}"),
                normalize(row["Prompt"]),
            )
            items.setdefault(key, []).append(row)
    return items


def label_from_responses(raw_responses: str, item_id: str) -> str:
    try:
        responses = [int(value.strip()) for value in raw_responses.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid annotator response for {item_id}") from exc
    if not responses or any(value not in range(-3, 4) for value in responses):
        raise ValueError(f"Missing or invalid annotator responses for {item_id}")

    threshold = math.ceil(0.8 * len(responses))
    matches = [
        label
        for label, values in LABEL_RANGES.items()
        if sum(value in values for value in responses) >= threshold
    ]
    if len(matches) != 1:
        raise ValueError(f"The 80% label rule does not select one label for {item_id}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--items-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    items = read_items(args.items_csv)
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    counts: Counter[str] = Counter()

    with args.test_jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (normalize(row["premise"]), normalize(row["hypothesis"]))
            if key in seen:
                raise ValueError(f"Duplicate test example on line {line_number}")
            seen.add(key)
            item = items.get(key)
            if item is None:
                raise ValueError(f"No source annotation match for test line {line_number}")
            if len(item) != 1:
                raise ValueError(f"Ambiguous source annotation match for test line {line_number}")
            source_item = item[0]
            row["label"] = label_from_responses(
                source_item["Reponses"], source_item.get("uID", str(line_number))
            )
            counts[row["label"]] += 1
            rows.append(row)

    if len(rows) != 250:
        raise ValueError(f"Expected 250 CB test records, found {len(rows)}")
    expected_counts = {"entailment": 113, "neutral": 16, "contradiction": 121}
    if dict(counts) != expected_counts:
        raise ValueError(
            f"Expected published CB test label counts {expected_counts}, found {dict(counts)}"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} labeled CB test records to {args.output_jsonl}")
    print("Label counts: " + ", ".join(f"{label}={counts[label]}" for label in LABEL_RANGES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
