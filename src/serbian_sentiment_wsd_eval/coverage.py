from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .csvio import read_delimited, write_dicts
from .scoring import load_sentiment_table


def write_sentiment_coverage_report(
    wsd_path: str | Path,
    resources_dir: str | Path,
    out_path: str | Path,
) -> dict[str, int]:
    selected: dict[str, set[str]] = defaultdict(set)
    selected_order: list[str] = []
    for row in read_delimited(wsd_path, delimiter="\t"):
        kbid = (row.get("KBid") or "").strip()
        if not kbid or kbid == "NEW_SENSE":
            continue
        if kbid not in selected:
            selected_order.append(kbid)
        selected[kbid].add(row.get("sentence_id", ""))

    synset_dir = Path(resources_dir) / "synset_sentiment"
    tables = [
        (path.stem, load_sentiment_table(path))
        for path in sorted(synset_dir.glob("*.csv"))
    ]

    rows: list[dict[str, object]] = []
    for table, sentiments in tables:
        for kbid in selected_order:
            if kbid in sentiments:
                continue
            sentences = sorted(value for value in selected[kbid] if value)
            rows.append(
                {
                    "table": table,
                    "KBid": kbid,
                    "count": len(sentences),
                    "sentences": ";".join(sentences[:25]),
                }
            )

    write_dicts(
        out_path,
        rows,
        ["table", "KBid", "count", "sentences"],
        delimiter="\t",
    )
    return {"selected_ids": len(selected), "missing_rows": len(rows)}
