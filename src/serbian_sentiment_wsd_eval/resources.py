from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .constants import (
    DEFAULT_POLARITY_SOURCE,
    DEFAULT_RESOURCE_DIR,
    DEFAULT_SENTIMENT_SOURCE_DIR,
    SYNSET_SOURCE_FILES,
)
from .csvio import write_dicts


def normalize_polarity_lexicon(source: str | Path, out: str | Path) -> dict[str, object]:
    source_path = Path(source)
    values_by_lemma: dict[str, set[int]] = defaultdict(set)
    rows_read = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if not row:
                continue
            rows_read += 1
            lemma = row[0].strip().strip('"')
            if not lemma:
                continue
            pos_flag = len(row) > 1 and row[1].strip() != ""
            neg_flag = len(row) > 2 and row[2].strip() != ""
            if pos_flag and not neg_flag:
                polarity = 1
            elif neg_flag and not pos_flag:
                polarity = -1
            else:
                polarity = 0
            values_by_lemma[lemma].add(polarity)

    duplicate_conflicts = sorted(
        lemma for lemma, polarities in values_by_lemma.items() if len(polarities) > 1
    )
    rows: list[dict[str, object]] = []
    for lemma in sorted(values_by_lemma):
        polarities = values_by_lemma[lemma]
        polarity = 0 if len(polarities) > 1 else next(iter(polarities))
        rows.append({"lemma": lemma, "polarity": polarity})

    out_path = write_dicts(out, rows, ["lemma", "polarity"])
    polarity_counts = Counter(str(row["polarity"]) for row in rows)
    return {
        "source": str(source_path),
        "output": str(out_path),
        "rows_read": rows_read,
        "rows_written": len(rows),
        "duplicate_conflicts": duplicate_conflicts,
        "polarity_counts": dict(sorted(polarity_counts.items())),
    }


def normalize_sentiment_table(
    source: str | Path,
    out: str | Path,
    *,
    label: str,
) -> dict[str, object]:
    source_path = Path(source)
    rows: list[dict[str, object]] = []
    rows_read = 0
    seen_ids: set[str] = set()

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"ID", "POS", "NEG"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            rows_read += 1
            synset_id = (row.get("ID") or "").strip()
            if not synset_id or synset_id in seen_ids:
                continue
            pos = float(row.get("POS") or 0)
            neg = float(row.get("NEG") or 0)
            rows.append(
                {
                    "ID": synset_id,
                    "POS": str(pos),
                    "NEG": str(neg),
                    "Lemme": (row.get("Lemme") or "").strip(),
                    "Vrsta": (row.get("Vrsta") or "").strip(),
                }
            )
            seen_ids.add(synset_id)

    out_path = write_dicts(out, rows, ["ID", "POS", "NEG", "Lemme", "Vrsta"])
    return {
        "label": label,
        "source": str(source_path),
        "output": str(out_path),
        "rows_read": rows_read,
        "rows_written": len(rows),
    }


def prepare_resources(
    *,
    polarity_source: str | Path = DEFAULT_POLARITY_SOURCE,
    sentiment_source_dir: str | Path = DEFAULT_SENTIMENT_SOURCE_DIR,
    out_dir: str | Path = DEFAULT_RESOURCE_DIR,
) -> dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    synset_out = out_path / "synset_sentiment"
    synset_out.mkdir(parents=True, exist_ok=True)

    reports: dict[str, object] = {
        "polarity": normalize_polarity_lexicon(
            polarity_source,
            out_path / "lemma_polarity.csv",
        ),
        "synset_tables": {},
    }
    source_root = Path(sentiment_source_dir)
    for label, filename in SYNSET_SOURCE_FILES.items():
        reports["synset_tables"][label] = normalize_sentiment_table(
            source_root / filename,
            synset_out / f"{label}.csv",
            label=label,
        )

    manifest = {
        "resource_dir": str(out_path),
        "lemma_polarity": str(out_path / "lemma_polarity.csv"),
        "synset_sentiment_dir": str(synset_out),
        "tables": {label: str(synset_out / f"{label}.csv") for label in SYNSET_SOURCE_FILES},
    }
    (out_path / "resource_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_path / "normalization_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reports["manifest"] = manifest
    return reports
