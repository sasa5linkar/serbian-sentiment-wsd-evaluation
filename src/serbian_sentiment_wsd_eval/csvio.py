from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def read_delimited(path: str | Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8-sig", newline="") as handle:
        if delimiter is None:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = "\t" if in_path.suffix.lower() == ".tsv" else ","
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_dicts(
    path: str | Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
    *,
    delimiter: str = ",",
    utf8_sig: bool = True,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if utf8_sig else "utf-8"
    with out_path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out_path
