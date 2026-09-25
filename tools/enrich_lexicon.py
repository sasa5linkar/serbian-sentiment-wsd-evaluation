"""Join public ID/POS/NEG scores to a permitted ID/Lemme/Vrsta inventory."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scores", type=Path, required=True)
parser.add_argument("--inventory", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.resolve() in {args.scores.resolve(), args.inventory.resolve()}:
    parser.error("Use a separate output path; source inputs are preserved.")
with args.inventory.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    if not {"ID", "Lemme", "Vrsta"} <= set(reader.fieldnames or []):
        parser.error("Inventory requires ID,Lemme,Vrsta columns (one row per synset).")
    inventory = {}
    for row in reader:
        identifier = row["ID"].strip()
        if not identifier or identifier in inventory:
            parser.error("Inventory has an empty or duplicate ID.")
        inventory[identifier] = row
with args.scores.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    if not {"ID", "POS", "NEG"} <= set(reader.fieldnames or []):
        parser.error("Scores require ID,POS,NEG columns.")
    rows = []
    missing = 0
    seen = set()
    for row in reader:
        identifier = row["ID"].strip()
        if not identifier or identifier in seen:
            parser.error("Scores have an empty or duplicate ID.")
        seen.add(identifier)
        lexical = inventory.get(identifier)
        if lexical is None:
            missing += 1
        rows.append({"ID": identifier, "POS": row["POS"], "NEG": row["NEG"], "Lemme": lexical["Lemme"] if lexical else "", "Vrsta": lexical["Vrsta"] if lexical else ""})
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["ID", "POS", "NEG", "Lemme", "Vrsta"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} score rows; {missing} IDs have no inventory match.")
